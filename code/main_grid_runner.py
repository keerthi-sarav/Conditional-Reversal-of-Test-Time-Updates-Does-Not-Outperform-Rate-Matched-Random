"""
Main experiment grid — reduced scope (see preregistration.md Addendum 3).
Runs (condition x seed) jobs in parallel worker processes, single-threaded
each, saving one JSON per job to results/main_grid/.

Scope actually run here (disclosed deviation from the pre-registration):
  - 1 architecture (resnet20), not 2 (resnet20 + resnet56)
  - 3 seeds, not 5
  - 1 stream cycle through the 15 corruption types, not >=2
Everything else (gate implementation, fixed tolerance, conditions, stream
construction, batch size) matches preregistration.md exactly.
"""
import os, sys, json, pickle, time
import numpy as np

RESULTS_DIR = "results/main_grid"
CONDITIONS = ["no_adapt", "tent", "random_revert", "cotta", "gated_tent"]
SEEDS = [1, 2, 3]
ARCH = "resnet20"
CKPT = "/tmp/ckpts/cifar10_resnet20-4118986f.pt"
BATCH_SIZE = 32
BATCHES_PER_BLOCK = 63  # ~2016 images/corruption-type/cycle, matches pilot's per-block sampling logic
N_CYCLES = 1


def load_batch(p):
    with open(p, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    x = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y = np.array(d[b"labels"])
    return x, y


def run_one(condition, seed):
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
    out = SR.run_condition(condition, cr.load_pretrained, CKPT, ARCH, batches, calib_x,
                            seed=seed, tau_h=tau_h, tau_d=tau_d, lr=lr_main)
    elapsed = time.time() - t0

    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = os.path.join(RESULTS_DIR, f"{condition}_seed{seed}.json")
    payload = {
        "condition": condition, "seed": seed, "arch": ARCH,
        "n_batches": len(batches), "elapsed_s": elapsed,
        "accept_rate": out.get("accept_rate"),
        "log": out["log"],
    }
    json.dump(payload, open(fname, "w"))
    return fname, elapsed, np.mean([r["acc"] for r in out["log"]])


def _worker(args):
    condition, seed = args
    try:
        fname, elapsed, mean_acc = run_one(condition, seed)
        return {"condition": condition, "seed": seed, "status": "ok",
                "fname": fname, "elapsed_s": elapsed, "mean_acc": mean_acc}
    except Exception as e:
        import traceback
        return {"condition": condition, "seed": seed, "status": "error",
                "error": str(e), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    import multiprocessing as mp
    jobs = [(c, s) for c in CONDITIONS for s in SEEDS]
    n_workers = min(6, mp.cpu_count())
    print(f"Dispatching {len(jobs)} jobs across {n_workers} workers", flush=True)
    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        results = []
        for r in pool.imap_unordered(_worker, jobs):
            r["t_since_start_s"] = time.time() - t0
            print(json.dumps(r if r["status"] == "error" else
                              {k: v for k, v in r.items() if k != "traceback"}), flush=True)
            results.append(r)
    json.dump(results, open(os.path.join(RESULTS_DIR, "_manifest.json"), "w"), indent=1)
    print(f"DONE all {len(jobs)} jobs in {time.time()-t0:.0f}s", flush=True)
