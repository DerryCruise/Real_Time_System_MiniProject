"""
rt_common.py - shared timing utilities for the Real-Time Systems Unit 1 mini projects.

Provides
  * calibrate() / busy_ms()   CPU-bound work that takes a known number of milliseconds
  * BackgroundLoad            duty-cycle CPU-burning thread (Section 1.6 interference)
  * run_periodic()            drift-free periodic release loop with full instrumentation
  * analyze()                 release/completion jitter, response time, deadline hits
  * hard/firm/soft_utility()  the three cost functions of Section 1.2 (fig1_1)

Notes
  * Everything is a pure-software simulation: it illustrates timing concepts on a
    general-purpose OS and does NOT produce certifiable timing guarantees.
  * time.perf_counter() is used for every timestamp (monotonic, high resolution).
"""
import threading
import time
import numpy as np

perf = time.perf_counter

# ----------------------------------------------------------------------------
# Calibrated busy work
# ----------------------------------------------------------------------------
_ITERS_PER_MS = None


def _spin(n):
    """Pure-Python arithmetic loop; run time grows linearly with n."""
    s = 0
    for i in range(n):
        s += (i * i) % 7
    return s


def calibrate(n=200_000, repeats=7):
    """Measure how many loop iterations take 1 ms on this machine (idle CPU)."""
    global _ITERS_PER_MS
    durations = []
    for _ in range(repeats):
        t = perf()
        _spin(n)
        durations.append(perf() - t)
    _ITERS_PER_MS = n / (float(np.median(durations)) * 1000.0)
    return _ITERS_PER_MS


def busy_ms(ms):
    """Burn roughly `ms` milliseconds of CPU (calibrated on an idle machine)."""
    if _ITERS_PER_MS is None:
        calibrate()
    _spin(max(1, int(ms * _ITERS_PER_MS)))


# ----------------------------------------------------------------------------
# Background interference (Section 1.6: general interference / OS noise)
# ----------------------------------------------------------------------------
class BackgroundLoad:
    """Thread that burns CPU for `duty` of every `slice_s` seconds (duty=1.0 -> continuous)."""

    def __init__(self, duty, slice_s=0.005):
        self.duty, self.slice_s = duty, slice_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            if self.duty >= 1.0:
                _spin(2000)                                  # ~0.2 ms chunks so stop is noticed
            else:
                end = perf() + self.duty * self.slice_s
                while perf() < end:
                    _spin(500)
                time.sleep((1.0 - self.duty) * self.slice_s)

    def __enter__(self):
        if self.duty > 0:
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self.duty > 0:
            self._thread.join()


# ----------------------------------------------------------------------------
# Periodic release loop (drift-free: every target is r0 + k*T, never "previous + T")
# ----------------------------------------------------------------------------
def run_periodic(body, n_jobs, period, pre_job=None, start_delay=0.1):
    """
    Run `body(k)` for k = 0..n_jobs-1, nominally released at r0 + k*period.

    pre_job(k), if given, runs after the loop wakes up and BEFORE the release instant is
    recorded; it models an interrupt/DMA-like stall that delays the job's effective start.

    Returns arrays (seconds, relative to r0): ideal release, actual release, completion.
    """
    ideal = np.arange(n_jobs) * period
    actual = np.zeros(n_jobs)
    done = np.zeros(n_jobs)
    r0 = perf() + start_delay
    for k in range(n_jobs):
        remaining = (r0 + ideal[k]) - perf()           # absolute target recomputed each iteration
        if remaining > 0:
            time.sleep(remaining)
        if pre_job is not None:
            pre_job(k)
        actual[k] = perf() - r0
        body(k)
        done[k] = perf() - r0
    return dict(ideal=ideal, actual=actual, done=done, period=period)


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------
def analyze(rec, deadline, deadline_ref="actual"):
    """
    release jitter    = actual release - ideal release
    completion jitter = (completion - ideal release) minus its mean, i.e. deviation of each
                        completion instant from a perfectly periodic completion pattern
    response time     = completion - actual release          (deadline_ref="actual",  Project 1.1)
                      = completion - ideal release           (deadline_ref="ideal",   Project 1.3)
    A job meets its deadline when response time <= deadline.
    """
    ideal, actual, done = rec["ideal"], rec["actual"], rec["done"]
    rel_j = actual - ideal
    offset = done - ideal
    comp_j = offset - offset.mean()
    ref = actual if deadline_ref == "actual" else ideal
    resp = done - ref
    hit = resp <= deadline
    return dict(
        release_jitter=rel_j, completion_jitter=comp_j, response=resp, hit=hit,
        rel_std=float(rel_j.std()), rel_max=float(np.abs(rel_j).max()),
        comp_std=float(comp_j.std()), comp_max=float(np.abs(comp_j).max()),
        hit_fraction=float(hit.mean()), resp_mean=float(resp.mean()), resp_max=float(resp.max()),
    )


# ----------------------------------------------------------------------------
# Cost functions of Section 1.2 (fig1_1)
# ----------------------------------------------------------------------------
def hard_utility(R, D, penalty=-100.0):
    return np.where(np.asarray(R) <= D, 1.0, penalty)


def firm_utility(R, D):
    return np.where(np.asarray(R) <= D, 1.0, 0.0)


def soft_utility(R, D, tau):
    """1 while R <= D, then exponential decay exp(-(R-D)/tau)."""
    R = np.asarray(R)
    return np.where(R <= D, 1.0, np.exp(-(R - D) / tau))
