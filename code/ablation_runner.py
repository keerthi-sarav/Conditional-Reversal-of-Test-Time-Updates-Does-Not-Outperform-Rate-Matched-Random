"""
Ablation & sensitivity grid (see preregistration.md Sec.5 "threshold sweep is
an ablation, reported separately" + plan step "Ablations and sensitivity").

SCOPE REDUCTION (disclosed, matches the Addendum-3 pattern): this ablation
runs at seed=1 only and a shortened 300-batch stream (~32% of the 945-batch
main-grid stream), not the full 5-seed design, because of the same CPU-only
sandbox ceiling that forced the main grid's scope cut. This is an ablation
for mechanism understanding, not a replacement for the pre-registered primary
comparison (which used the full main grid). Findings here are reported as
single-seed, reduced-stream sensitivity checks, not as additional confirmatory
evidence at the same standard as the main result.

Four axes, each varied one-at-a-time off the main-grid gated_tent config
(tau_h=0.00218815, tau_d=0.01000001, snap_every=50, lr=0.01, resnet20):

  A. Tolerance multiplier sweep on (tau_h, tau_d): 0.25x, 0.5x, 1x, 2x, 4x
     -> the accuracy-vs-revert-rate frontier (does the gate merely revert more,
        or does it revert *well*?).
  B. Acceptance-criterion component: entropy-only (tau_d=inf), disagreement-only
     (tau_h=inf), vs both (=the 1x point from axis A, not re-run).
  C. Snapshot/anchor refresh cadence: snap_every in {25, 100} vs the main
     value of 50 (not re-run).
  D. Architecture generalization (resnet56): ATTEMPTED, then DROPPED. Fresh
     tolerance calibration on resnet56 (75 warm-up batches, deeper net) did
     not finish within a reasonable wait under the CPU contention already
     present from the corrected random_revert re-run occupying the sandbox's
     cores, so it was interrupted. This is disclosed as a further
     compute-driven scope cut (Addendum 6) -- consistent with the earlier
     cuts (single-seed main-run scope, no_cycles=1) -- rather than run at an
     even-more-reduced scale that would make the comparison uninformative.
     The ablation below therefore covers axes A-C on resnet20 only; the
     pre-registration's "both architectures" success threshold was already
     unmet after Addendum 3 and remains unmet here.

Batches further cut to 60 (a single cycle through all 15 corruption types at
4 batches/type) to fit the remaining compute budget alongside the
still-running corrected random_revert re-run.
"""
import os, sys, json, time, glob
import numpy as np
import torch

sys.path.append(os.getcwd())
import cifar_resnet as cr
import tta_methods as M
import stream_runner as SR

RESULTS_DIR = "results/ablation"
os.makedirs(RESULTS_DIR, exist_ok=True)

CKPT = {"resnet20": "/tmp/ckpts/cifar10_resnet20-4118986f.pt",
        "resnet56": "/tmp/ckpts/cifar10_resnet56-187c023a.pt"}
SEED = 1
N_BATCHES = 300
BATCH_SIZE = 32

_state = torch.load("ablation_state.pt", weights_only=False)
calib_x = _state["calib_x"]
warmup_batches = _state["warmup_batches"]
ablation_batches = _state["ablation_batches"]  # fixed 300-batch cyclic stream, seed=1
TAU_H0 = _state["tau_h0"]
TAU_D0 = _state["tau_d0"]


def run_one(job_id, arch, condition, tau_h, tau_d, snap_every, lr=0.01,
            random_revert_accept_prob=None):
    t0 = time.time()
    fname = os.path.join(RESULTS_DIR, f"{job_id}.json")
    if os.path.exists(fname):
        print(f"[skip] {job_id} already done", flush=True)
        return fname, 0.0, None
    torch.manual_seed(SEED)
    model, _, _ = cr.load_pretrained(arch, CKPT[arch])
    res = SR.run_condition(condition, cr.load_pretrained, CKPT[arch], arch,
                            ablation_batches, calib_x, seed=SEED,
                            tau_h=tau_h, tau_d=tau_d, snap_every=snap_every, lr=lr,
                            random_revert_accept_prob=random_revert_accept_prob)
    accs = [r["acc"] for r in res["log"]]
    out = {"job_id": job_id, "arch": arch, "condition": condition,
           "tau_h": tau_h, "tau_d": tau_d, "snap_every": snap_every, "lr": lr,
           "accept_rate": res.get("accept_rate"), "mean_acc": float(np.mean(accs)),
           "log": res["log"]}
    json.dump(out, open(fname, "w"))
    elapsed = time.time() - t0
    print(f"[done] {job_id} mean_acc={out['mean_acc']:.4f} "
          f"accept_rate={out['accept_rate']} ({elapsed:.0f}s)", flush=True)
    return fname, elapsed, out["mean_acc"]


def calibrate_for_arch(arch):
    """Fresh tau_h/tau_d for `arch`, calibrated on the SAME warm-up batches
    used for resnet20 (protocol: warm-up only, never eval data)."""
    tau_h, tau_d = calibrate_tolerance(cr.load_pretrained, CKPT[arch], arch,
                                        warmup_batches, calib_x, seed=SEED)
    return float(tau_h), float(tau_d)


def _worker(j):
    try:
        fname, elapsed, mean_acc = run_one(**j)
        return {"job_id": j["job_id"], "status": "ok", "file": fname,
                "elapsed_s": elapsed, "mean_acc": mean_acc}
    except Exception as e:
        import traceback
        return {"job_id": j["job_id"], "status": "error", "error": str(e),
                "traceback": traceback.format_exc()}


if __name__ == "__main__":
    jobs = []

    # Axis A: tolerance multiplier sweep (resnet20, gated_tent)
    for mult in [0.25, 0.5, 1.0, 2.0, 4.0]:
        jobs.append(dict(job_id=f"threshold_mult_{mult}", arch="resnet20",
                          condition="gated_tent", tau_h=TAU_H0 * mult, tau_d=TAU_D0 * mult,
                          snap_every=50))

    # Axis B: acceptance-criterion components (resnet20, gated_tent)
    jobs.append(dict(job_id="criterion_entropy_only", arch="resnet20", condition="gated_tent",
                      tau_h=TAU_H0, tau_d=float("inf"), snap_every=50))
    jobs.append(dict(job_id="criterion_disagreement_only", arch="resnet20", condition="gated_tent",
                      tau_h=float("inf"), tau_d=TAU_D0, snap_every=50))

    # Axis C: snapshot/anchor refresh cadence (resnet20, gated_tent)
    for snap in [25, 100]:
        jobs.append(dict(job_id=f"snap_every_{snap}", arch="resnet20", condition="gated_tent",
                          tau_h=TAU_H0, tau_d=TAU_D0, snap_every=snap))

    # Axis D (architecture generalization) DROPPED -- see module docstring
    # ("Addendum 6: compute-driven scope cut"). Not run.

    # matched no_adapt / ungated tent on resnet20 for reference at this stream length
    jobs.append(dict(job_id="ref_resnet20_no_adapt", arch="resnet20", condition="no_adapt",
                      tau_h=TAU_H0, tau_d=TAU_D0, snap_every=50))
    jobs.append(dict(job_id="ref_resnet20_tent", arch="resnet20", condition="tent",
                      tau_h=TAU_H0, tau_d=TAU_D0, snap_every=50, lr=0.01))

    manifest = {"tau_h0": TAU_H0, "tau_d0": TAU_D0, "seed": SEED, "n_batches": len(ablation_batches),
                "architecture_axis": "dropped (Addendum 6, compute-driven)", "jobs": []}

    import multiprocessing as mp
    n_workers = min(2, mp.cpu_count())  # keep contention low alongside the rerun job
    print(f"Dispatching {len(jobs)} ablation jobs across {n_workers} workers", flush=True)
    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        for r in pool.imap_unordered(_worker, jobs):
            r["t_since_start_s"] = time.time() - t0
            print(json.dumps(r), flush=True)
            manifest["jobs"].append(r)
    json.dump(manifest, open(os.path.join(RESULTS_DIR, "_manifest.json"), "w"), indent=1)
    print(f"DONE all {len(jobs)} ablation jobs in {time.time()-t0:.0f}s", flush=True)
