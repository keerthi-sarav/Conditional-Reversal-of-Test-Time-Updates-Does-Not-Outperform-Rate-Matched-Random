"""
Corrected re-run of random_revert only (see preregistration.md Addendum 5).

Bug fixed: random_revert's per-batch accept probability must match the
gated_tent run's own empirical accept_rate on the same seed/stream
(preregistration.md Sec. 4); the original main_grid_runner.py silently used a
fixed 0.5 default instead. This script reads each seed's already-completed
gated_tent_seed{seed}.json accept_rate and passes it through explicitly.
Only random_revert is affected -- no_adapt/tent/cotta/gated_tent results from
the original run stand as-is.
"""
import os, sys, json, pickle, time
import numpy as np

RESULTS_DIR = "results/main_grid"  # overwrite the buggy random_revert_seed*.json in place
SEEDS = [1, 2, 3]
ARCH = "resnet20"
CKPT = "/tmp/ckpts/cifar10_resnet20-4118986f.pt"
BATCH_SIZE = 32
BATCHES_PER_BLOCK = 63
N_CYCLES = 1


def load_batch(p):
    with open(p, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    x = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y = np.array(d[b"labels"])
    return x, y


def run_one(seed, matched_accept_prob):
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, os.getcwd())
    import cifar_resnet as cr
    import tta_methods as M
    import stream_runner as SR

    tol = json.load(open("gate_tolerance.json"))
    tau_h, tau_d, lr_main = tol["tau_h"], tol["tau_d"], tol["lr_main"]
    calib_x = torch.load("calib_x.pt")
    xte, yte = load_batch("data/cifar-10-batches-py/test_batch")

    batches, order = SR.build_stream(xte, yte, seed=seed, batch_size=BATCH_SIZE,
                                      batches_per_block=BATCHES_PER_BLOCK, n_cycles=N_CYCLES)

    t0 = time.time()
    out = SR.run_condition("random_revert", cr.load_pretrained, CKPT, ARCH, batches, calib_x,
                            seed=seed, tau_h=tau_h, tau_d=tau_d, lr=lr_main,
                            random_revert_accept_prob=matched_accept_prob)
    elapsed = time.time() - t0

    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = os.path.join(RESULTS_DIR, f"random_revert_seed{seed}.json")
    payload = {
        "condition": "random_revert", "seed": seed, "arch": ARCH,
        "n_batches": len(batches), "elapsed_s": elapsed,
        "accept_rate": out.get("accept_rate"),
        "matched_target_accept_rate": matched_accept_prob,
        "log": out["log"],
    }
    json.dump(payload, open(fname, "w"))
    return fname, elapsed, np.mean([r["acc"] for r in out["log"]])


def _worker(args):
    seed, matched_accept_prob = args
    try:
        fname, elapsed, mean_acc = run_one(seed, matched_accept_prob)
        return {"condition": "random_revert", "seed": seed, "status": "ok",
                "fname": fname, "elapsed_s": elapsed, "mean_acc": mean_acc}
    except Exception as e:
        import traceback
        return {"condition": "random_revert", "seed": seed, "status": "error",
                "error": str(e), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    import multiprocessing as mp
    matched_rates = {}
    for s in SEEDS:
        gt = json.load(open(os.path.join(RESULTS_DIR, f"gated_tent_seed{s}.json")))
        matched_rates[s] = gt["accept_rate"]
    print("Matched target accept rates (from gated_tent):", matched_rates, flush=True)

    jobs = [(s, matched_rates[s]) for s in SEEDS]
    n_workers = min(3, mp.cpu_count())  # only 3 jobs, keep contention low
    print(f"Dispatching {len(jobs)} corrected random_revert jobs across {n_workers} workers", flush=True)
    t0 = time.time()
    results = []
    with mp.Pool(n_workers) as pool:
        for r in pool.imap_unordered(_worker, jobs):
            r["t_since_start_s"] = time.time() - t0
            print(json.dumps(r if r["status"] == "error" else
                              {k: v for k, v in r.items() if k != "traceback"}), flush=True)
            results.append(r)
    json.dump(results, open(os.path.join(RESULTS_DIR, "_manifest_random_revert_rerun.json"), "w"), indent=1)
    print(f"DONE all {len(jobs)} corrected random_revert jobs in {time.time()-t0:.0f}s", flush=True)
