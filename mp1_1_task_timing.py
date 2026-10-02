"""
Mini Project 1.1 - Simulating and Measuring Real-Time Task Timing Behavior in Python
Real-Time Systems, Unit 1

Covers procedure steps 1-9 and Lab Tasks 1-4.

Run:  python mp1_1_task_timing.py          (takes about a minute; figures go to ./figures)
Note: timings come from a general-purpose OS, so numbers differ from run to run and machine
      to machine. The structure of the results (tail > mean, jitter > 0, loaded > unloaded)
      is what the lab is about.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rt_common import (perf, calibrate, busy_ms, BackgroundLoad, run_periodic, analyze,
                       hard_utility, firm_utility, soft_utility)

os.makedirs("figures", exist_ok=True)
SEED = 2026
MS = 1e3

# ----------------------------------------------------------------------------
# Step 1: the job function. Run time depends on the input x in [0, 1].
#   common path : 8 + 8*x ms           (8 - 16 ms)
#   long path   : x > 0.9 adds 220*(x-0.9) ms, up to +22 ms (data-dependent, rarely taken)
# ----------------------------------------------------------------------------
LONG_PATH_X = 0.9


def job(x):
    busy_ms(8.0 + 8.0 * x)
    if x > LONG_PATH_X:
        busy_ms(220.0 * (x - LONG_PATH_X))


def measure_job(inputs):
    """Step 2: time every call individually with perf_counter()."""
    out = np.zeros(len(inputs))
    for i, x in enumerate(inputs):
        t = perf()
        job(x)
        out[i] = perf() - t
    return out


def summary(a):
    return dict(mean=float(a.mean()), std=float(a.std()), max=float(a.max()), p95=float(np.percentile(a, 95)),
                median=float(np.median(a)))


def periodic_experiment(T, n_jobs, inputs, duty=0.0, D_frac=0.8):
    """Steps 3-6 for one configuration. Deadline is measured from the ACTUAL release (D = D_frac*T)."""
    D = D_frac * T
    with BackgroundLoad(duty):
        rec = run_periodic(lambda k: job(inputs[k]), n_jobs, T)
    res = analyze(rec, D, "actual")
    res.update(rec=rec, T=T, D=D)
    return res


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    ipm = calibrate()
    print(f"Calibration: {ipm:,.0f} loop iterations per millisecond")
    results = {}

    # ------------------------------------------------------------------------
    # Step 2 + Lab Task 2: execution time in isolation, ordinary vs adversarial inputs
    # ------------------------------------------------------------------------
    ordinary_inputs = rng.uniform(0.0, 0.85, 300)       # representative test profile: long path never exercised
    adversarial_inputs = np.full(300, 0.999)            # hand-picked to hit the longest path
    c_ord = measure_job(ordinary_inputs)
    c_adv = measure_job(adversarial_inputs)
    s_ord, s_adv = summary(c_ord), summary(c_adv)
    print("\nSTEP 2 - isolated execution time (ms), 300 calls each")
    print(f"  ordinary inputs   : mean {s_ord['mean']*MS:6.2f}  std {s_ord['std']*MS:5.2f}  max {s_ord['max']*MS:6.2f}  p95 {s_ord['p95']*MS:6.2f}  median {s_ord['median']*MS:6.2f}")
    print(f"  adversarial inputs: mean {s_adv['mean']*MS:6.2f}  std {s_adv['std']*MS:5.2f}  max {s_adv['max']*MS:6.2f}  p95 {s_adv['p95']*MS:6.2f}  median {s_adv['median']*MS:6.2f}")
    print(f"  -> the ordinary campaign's maximum ({s_ord['max']*MS:.1f} ms) is a MEASURED, not proven, WCET estimate;")
    print(f"     the adversarial input exposes {s_adv['max']*MS:.1f} ms ({s_adv['max']/s_ord['max']:.2f}x larger); comparing medians "
          f"({s_adv['median']*MS:.1f} vs {s_ord['median']*MS:.1f} ms) separates the real path cost from OS-noise spikes in the maxima.")
    results["exec_ordinary_ms"] = {k: v * MS for k, v in s_ord.items()}
    results["exec_adversarial_ms"] = {k: v * MS for k, v in s_adv.items()}

    plt.figure(figsize=(8.5, 4.8))
    bins = np.linspace(0, max(c_adv.max(), c_ord.max()) * MS * 1.05, 60)
    plt.hist(c_ord * MS, bins=bins, color="tab:blue", alpha=0.75, label="Ordinary inputs (x in [0, 0.85))")
    plt.hist(c_adv * MS, bins=bins, color="tab:red", alpha=0.75, label="Adversarial input (x = 0.999)")
    plt.axvline(s_ord["mean"] * MS, color="navy", ls=":", label=f"ordinary mean {s_ord['mean']*MS:.1f} ms")
    plt.axvline(s_ord["p95"] * MS, color="navy", ls="-.", label=f"ordinary p95 {s_ord['p95']*MS:.1f} ms")
    plt.axvline(s_ord["max"] * MS, color="navy", ls="--", label=f"ordinary max {s_ord['max']*MS:.1f} ms (measured WCET estimate)")
    plt.axvline(s_adv["max"] * MS, color="darkred", ls="--", label=f"adversarial max {s_adv['max']*MS:.1f} ms")
    plt.xlabel("Execution time of one job (ms)")
    plt.ylabel("Number of calls")
    plt.title("Step 2 and Lab Task 2: execution-time distribution of the job function")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/rts1_fig3_exec_hist.png", dpi=150)
    plt.close()

    # ------------------------------------------------------------------------
    # Steps 3-6: unloaded periodic run, T = 40 ms, D = 0.8 T = 32 ms, 120 jobs
    # Field inputs cover the whole range, so the long path occasionally occurs.
    # ------------------------------------------------------------------------
    T, N = 0.04, 120
    field_inputs = rng.uniform(0.0, 1.0, N)
    base = periodic_experiment(T, N, field_inputs)
    D = base["D"]
    R = base["response"]
    print(f"\nSTEPS 3-6 - periodic run: T = {T*MS:.0f} ms, D = {D*MS:.0f} ms, {N} jobs (unloaded)")
    print(f"  response time   : mean {R.mean()*MS:6.2f}  max {R.max()*MS:6.2f}  p95 {np.percentile(R, 95)*MS:6.2f} ms")
    print(f"  release jitter  : std {base['rel_std']*MS:6.3f} ms, max |.| {base['rel_max']*MS:6.3f} ms")
    print(f"  completion jitter: std {base['comp_std']*MS:6.3f} ms, max |.| {base['comp_max']*MS:6.3f} ms")
    print(f"  deadline hits   : {base['hit'].sum()}/{N}  (hit fraction {base['hit_fraction']*100:.1f} %)")
    tau = 0.25 * D
    u_hard, u_firm, u_soft = hard_utility(R, D), firm_utility(R, D), soft_utility(R, D, tau)
    print(f"  mean utility    : hard {u_hard.mean():7.2f} | firm {u_firm.mean():5.3f} | soft(tau=0.25D) {u_soft.mean():5.3f}")
    results["base"] = dict(T_ms=T*MS, D_ms=D*MS, resp_mean_ms=R.mean()*MS, resp_max_ms=R.max()*MS,
                           resp_p95_ms=float(np.percentile(R, 95))*MS, rel_std_ms=base["rel_std"]*MS,
                           rel_max_ms=base["rel_max"]*MS, comp_std_ms=base["comp_std"]*MS, comp_max_ms=base["comp_max"]*MS,
                           hit_fraction=base["hit_fraction"], misses=int((~base["hit"]).sum()),
                           util_hard=float(u_hard.mean()), util_firm=float(u_firm.mean()), util_soft=float(u_soft.mean()),
                           exec_mean_vs_T=s_ord["mean"]/T)

    # Step 7: fig1_3-style timing diagram around the slowest job (12 consecutive jobs)
    kmax = int(np.argmax(R))
    k0 = int(np.clip(kmax - 6, 0, N - 12))
    ks = range(k0, k0 + 12)
    rec = base["rec"]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for row, k in enumerate(ks):
        a, c = rec["actual"][k] * MS, rec["done"][k] * MS
        dl = a + D * MS
        ax.broken_barh([(a, c - a)], (row - 0.35, 0.7), facecolors="tab:red" if not base["hit"][k] else "tab:blue")
        ax.plot([dl, dl], [row - 0.45, row + 0.45], color="k", lw=2)
        ax.plot(rec["ideal"][k] * MS, row, marker="^", color="green", ms=7)
    ax.set_yticks(range(12))
    ax.set_yticklabels([f"job {k}" for k in ks])
    ax.invert_yaxis()
    ax.set_xlabel("Time since r0 (ms)")
    ax.set_title("Step 7: timing diagram (blue = deadline met, red = missed, black bar = deadline, green triangle = ideal release)")
    ax.title.set_fontsize(10)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/rts1_fig1_timing_diagram.png", dpi=150)
    plt.close()

    # Step 8: response time and the three utility sequences
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True)
    ax[0].plot(R * MS, marker="o", ms=3, lw=1, color="tab:blue", label="Response time R")
    ax[0].axhline(D * MS, color="red", ls="--", label=f"Deadline D = {D*MS:.0f} ms")
    ax[0].set_ylabel("R (ms)")
    ax[0].set_title("Step 8: same timing behaviour, three deadline classifications")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    ax[1].plot(u_hard, drawstyle="steps-mid", color="tab:red", label="Hard (miss = -100)")
    ax[1].plot(u_firm, drawstyle="steps-mid", color="tab:orange", ls="--", label="Firm (miss = 0)")
    ax[1].plot(u_soft, marker=".", lw=1, color="tab:green", label=f"Soft (exp. decay, tau = {tau*MS:.0f} ms)")
    ax[1].set_yscale("symlog", linthresh=1)
    ax[1].set_ylabel("Utility (symlog)")
    ax[1].set_xlabel("Job index")
    ax[1].legend(loc="center right", fontsize=8)
    ax[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/rts1_fig2_response_utility.png", dpi=150)
    plt.close()

    # Expected observation 3: shrink D, watch the miss fraction (re-evaluated on the measured R values)
    d_grid = np.linspace(0.1, 1.0, 91) * T
    miss_frac = np.array([(R > d).mean() for d in d_grid])
    plt.figure(figsize=(8.5, 4.6))
    plt.plot(d_grid * MS, miss_frac * 100, color="tab:red", lw=2)
    plt.axvline(R.mean() * MS, color="navy", ls=":", label=f"mean R = {R.mean()*MS:.1f} ms")
    plt.axvline(np.percentile(R, 95) * MS, color="navy", ls="-.", label=f"95th percentile R = {np.percentile(R, 95)*MS:.1f} ms")
    plt.axvline(R.max() * MS, color="navy", ls="--", label=f"max R = {R.max()*MS:.1f} ms")
    plt.axvline(D * MS, color="k", ls="-", alpha=0.4, label=f"chosen D = {D*MS:.0f} ms")
    plt.xlabel("Relative deadline D (ms)")
    plt.ylabel("Deadline-miss fraction (%)")
    plt.title("Expected observation 3: the tail, not the mean, governs the miss fraction")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/rts1_fig4_miss_vs_deadline.png", dpi=150)
    plt.close()
    results["miss_at_D_equals_p95_pct"] = float((R > np.percentile(R, 95)).mean() * 100)
    results["miss_at_D_equals_mean_pct"] = float((R > R.mean()).mean() * 100)

    # ------------------------------------------------------------------------
    # Lab Task 1: background interference thread
    # ------------------------------------------------------------------------
    loaded = periodic_experiment(T, N, field_inputs, duty=1.0)
    RL = loaded["response"]
    print("\nLAB TASK 1 - background CPU-burning thread (duty 100 %)")
    print(f"  {'':10s} {'rel jitter std':>15s} {'rel max':>9s} {'mean R':>9s} {'max R':>9s} {'hit fraction':>13s}")
    for name, r in (("unloaded", base), ("loaded", loaded)):
        print(f"  {name:10s} {r['rel_std']*MS:12.3f} ms {r['rel_max']*MS:6.2f} ms {r['resp_mean']*MS:6.2f} ms "
              f"{r['resp_max']*MS:6.2f} ms {r['hit_fraction']*100:11.1f} %")
    results["loaded"] = dict(rel_std_ms=loaded["rel_std"]*MS, rel_max_ms=loaded["rel_max"]*MS,
                             resp_mean_ms=loaded["resp_mean"]*MS, resp_max_ms=loaded["resp_max"]*MS,
                             hit_fraction=loaded["hit_fraction"], comp_std_ms=loaded["comp_std"]*MS)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    bins = np.linspace(min(base["release_jitter"].min(), loaded["release_jitter"].min()) * MS,
                       max(base["release_jitter"].max(), loaded["release_jitter"].max()) * MS, 40)
    ax[0].hist(base["release_jitter"] * MS, bins=bins, alpha=0.75, color="tab:blue", label="Unloaded")
    ax[0].hist(loaded["release_jitter"] * MS, bins=bins, alpha=0.75, color="tab:red", label="With background thread")
    ax[0].set_xlabel("Release jitter (ms)"); ax[0].set_ylabel("Number of jobs")
    ax[0].set_title("Release jitter distribution"); ax[0].legend(); ax[0].grid(True, alpha=0.3)
    ax[1].plot(R * MS, color="tab:blue", label="Unloaded")
    ax[1].plot(RL * MS, color="tab:red", label="With background thread")
    ax[1].axhline(D * MS, color="k", ls="--", label="Deadline D")
    ax[1].set_xlabel("Job index"); ax[1].set_ylabel("Response time (ms)")
    ax[1].set_title("Response time"); ax[1].legend(fontsize=8); ax[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/rts1_fig5_loaded_vs_unloaded.png", dpi=150)
    plt.close()

    # ------------------------------------------------------------------------
    # Lab Task 3: soft-utility decay rate (cost functions of fig1_1 and three decay constants)
    # ------------------------------------------------------------------------
    taus = {"fast (tau = 0.05 D)": 0.05 * D, "default (tau = 0.25 D)": 0.25 * D, "slow (tau = 1.0 D)": 1.0 * D}
    print("\nLAB TASK 3 - mean soft utility on the unloaded run for three decay rates")
    soft_means = {}
    for name, tv in taus.items():
        soft_means[name] = float(soft_utility(R, D, tv).mean())
        print(f"  {name:24s}: {soft_means[name]:.4f}")
    results["soft_means"] = soft_means
    x = np.linspace(0, 2.0, 400)
    plt.figure(figsize=(8.5, 4.8))
    plt.plot(x, np.where(x <= 1, 1.0, -1.0), color="tab:red", lw=2, label="Hard (shown as -1 after D; -100 in the code)")
    plt.plot(x, np.where(x <= 1, 1.0, 0.0), color="tab:orange", lw=2, ls="--", label="Firm")
    for (name, tv), col in zip(taus.items(), ["tab:olive", "tab:green", "tab:cyan"]):
        plt.plot(x, soft_utility(x * D, D, tv), color=col, lw=2, label=f"Soft, {name}")
    plt.axvline(1.0, color="k", lw=1, alpha=0.5)
    plt.xlabel("Response time R / D"); plt.ylabel("Utility")
    plt.title("Lab Task 3: fig1_1 cost functions and the effect of the soft decay rate")
    plt.legend(fontsize=8); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig("figures/rts1_fig6_cost_functions.png", dpi=150)
    plt.close()

    # ------------------------------------------------------------------------
    # Lab Task 4: vary T with D = 0.8 T, 100 jobs each (unloaded)
    # ------------------------------------------------------------------------
    print("\nLAB TASK 4 - vary the period, D = 0.8 T, 100 jobs each")
    print(f"  {'T (ms)':>7s} {'rel jitter std (ms)':>20s} {'as % of T':>10s} {'rel max (ms)':>13s} {'hit fraction':>13s}")
    sweep = []
    for Tv in (0.02, 0.04, 0.08, 0.16):
        r = periodic_experiment(Tv, 100, rng.uniform(0, 1, 100))
        row = dict(T_ms=Tv*MS, rel_std_ms=r["rel_std"]*MS, rel_std_pct=r["rel_std"]/Tv*100,
                   rel_max_ms=r["rel_max"]*MS, hit_fraction=r["hit_fraction"])
        sweep.append(row)
        print(f"  {Tv*MS:7.0f} {row['rel_std_ms']:20.3f} {row['rel_std_pct']:9.2f}% {row['rel_max_ms']:13.2f} {row['hit_fraction']*100:11.1f} %")
    results["period_sweep"] = sweep
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot([s["T_ms"] for s in sweep], [s["rel_std_pct"] for s in sweep], "o-", color="tab:blue", lw=2, label="Release jitter std (% of T)")
    ax.set_xscale("log"); ax.set_xlabel("Period T (ms, log scale)"); ax.set_ylabel("Release jitter std as % of T", color="tab:blue")
    ax.grid(True, which="both", alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot([s["T_ms"] for s in sweep], [s["hit_fraction"] * 100 for s in sweep], "s--", color="tab:red", label="Deadline-hit fraction (%)")
    ax2.set_ylabel("Deadline-hit fraction (%)", color="tab:red"); ax2.set_ylim(0, 105)
    ax.set_title("Lab Task 4: jitter relative to the period (D = 0.8 T)")
    plt.tight_layout()
    plt.savefig("figures/rts1_fig7_period_sweep.png", dpi=150)
    plt.close()

    with open("results_mp1_1.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nDone. Figures in ./figures, numbers in results_mp1_1.json")
