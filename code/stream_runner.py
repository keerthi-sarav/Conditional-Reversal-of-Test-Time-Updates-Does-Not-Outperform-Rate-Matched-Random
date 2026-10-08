"""
Corruption-stream construction and per-condition runner for the Adapt-or-Abstain
study. Stream spec follows preregistration.md Sec. 5: severity fixed at 5,
corruptions presented in a cyclic non-IID block order (all batches of one
corruption type before switching), repeated for >=2 full cycles.
"""
import numpy as np
import torch
import copy
import time

import corruptions as CORR
import tta_methods as M
import cifar_resnet as cr

MEAN = torch.tensor(cr.CIFAR10_MEAN).view(1, 3, 1, 1)
STD = torch.tensor(cr.CIFAR10_STD).view(1, 3, 1, 1)


def preprocess(x_uint8):
    x = torch.from_numpy(x_uint8).float().permute(0, 3, 1, 2) / 255.0
    return (x - MEAN) / STD


def build_stream(xtest, ytest, seed, batch_size, batches_per_block, n_cycles,
                  corruption_types=None, severity=5):
    """Returns a list of (images_uint8, labels) batch tuples, cyclic block order,
    corrupted on the fly per-block from the shared test pool (sampled without
    replacement within a block; blocks may reuse images across cycles -- this
    is a continual *stream*, not a train/test split, so reuse across cycles is
    not a leakage concern)."""
    corruption_types = corruption_types or CORR.MAIN_15
    rng = np.random.default_rng(seed)
    order = list(corruption_types)
    rng.shuffle(order)
    batches = []
    n_pool = len(xtest)
    for cyc in range(n_cycles):
        for cname in order:
            fn = CORR.CORRUPTIONS[cname]
            idx_pool = rng.choice(n_pool, size=batches_per_block * batch_size, replace=False)
            for b in range(batches_per_block):
                idx = idx_pool[b * batch_size:(b + 1) * batch_size]
                imgs = np.stack([fn(xtest[i], severity) for i in idx])
                labs = ytest[idx]
                batches.append((imgs, labs, cname))
    return batches, order


def make_calib_buffer(xtrain, seed, n=200):
    rng = np.random.default_rng(seed + 9999)
    idx = rng.choice(len(xtrain), n, replace=False)
    return preprocess(xtrain[idx])


def run_condition(condition, model_fn, ckpt_path, arch, batches, calib_x, seed,
                   tau_h=None, tau_d=None, cotta_restore_prob=0.01, snap_every=50,
                   lr=1e-3, random_revert_accept_prob=None):
    """condition in {"no_adapt", "tent", "gated_tent", "random_revert", "cotta"}.
    Returns per-batch dict list: {t, corruption, acc, accepted, gate_event}."""
    model, missing, unexpected = model_fn(arch, ckpt_path)
    assert not missing and not unexpected
    source_snapshot = M.Snapshot(model) if condition != "no_adapt" else None

    if condition != "no_adapt":
        M.configure_tent(model)
        opt = torch.optim.Adam(list(M.bn_affine_params(model)), lr=lr)
    else:
        model.eval()
        opt = None

    gate = None
    rng_np = np.random.default_rng(seed + 555)
    if condition == "gated_tent":
        gate = M.ReversibilityGate(model, calib_x, tau_h, tau_d, snap_every=snap_every)
    elif condition == "random_revert":
        assert random_revert_accept_prob is not None, (
            "random_revert requires random_revert_accept_prob matched to the gated run's "
            "empirical accept_rate on this same seed/stream (preregistration.md Sec. 4) -- "
            "a hardcoded default here silently breaks the decisive comparison (see Addendum 5)."
        )
        gate = M.RandomRevertGate(model, calib_x, accept_prob=random_revert_accept_prob,
                                   rng=rng_np, snap_every=snap_every)

    log = []
    for t, (imgs, labs, cname) in enumerate(batches):
        x = preprocess(imgs)
        if condition == "no_adapt":
            with torch.no_grad():
                pred = model(x).argmax(1).numpy()
        elif condition == "tent":
            model.train()
            for m in model.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.train()
            logits = M.tent_step(model, x, opt)
            pred = logits.argmax(1).numpy()
        elif condition == "cotta":
            model.train()
            logits = M.tent_step(model, x, opt)
            pred = logits.argmax(1).numpy()
            M.cotta_stochastic_restore(model, source_snapshot, cotta_restore_prob, rng_np)
        elif condition in ("gated_tent", "random_revert"):
            pre_snapshot = M.Snapshot(model)
            ent_before, dis_before, _ = M.calib_signal(model, calib_x)
            model.train()
            for m in model.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.train()
            logits = M.tent_step(model, x, opt)
            pred = logits.argmax(1).numpy()
            ent_after, dis_after, _ = M.calib_signal(model, calib_x)
            accept = gate.decide(model, ent_before, dis_before, ent_after, dis_after)
            if not accept:
                pre_snapshot.restore(model)
            gate_event = gate.periodic_check(model)
        else:
            raise ValueError(condition)

        acc = float((pred == labs).mean())
        row = {"t": t, "corruption": cname, "acc": acc}
        if gate is not None:
            row["accepted"] = accept
            row["gate_event"] = gate_event
        log.append(row)
    result = {"log": log}
    if gate is not None:
        result["accept_rate"] = gate.accept_rate
    return result
