"""Generate result graphs from the measured values (taken from the run screenshots)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "results/"
C_BASE, C_OPT = "#d9534f", "#2e8b57"

# ---------------- measured values (RTX 4500 Ada, 200 batches) ----------------
total = [2.498, 1.502]            # s
per_batch = [12.49, 7.51]         # ms
throughput = [80.1, 133.2]        # batches/s

# ---- Figure 1: total time, per-batch time, throughput ----
fig, ax = plt.subplots(1, 3, figsize=(11, 3.8))
for a, vals, title, unit in zip(
        ax, [total, per_batch, throughput],
        ["Total time (200 batches)", "Time per batch", "Throughput"],
        ["s", "ms", "batches/s"]):
    bars = a.bar(["Baseline", "Optimized"], vals, color=[C_BASE, C_OPT], width=0.55)
    a.set_title(title); a.set_ylabel(unit)
    for b, v in zip(bars, vals):
        a.text(b.get_x() + b.get_width() / 2, v, f"{v}", ha="center", va="bottom")
    a.set_ylim(0, max(vals) * 1.18)
fig.suptitle("Figure 1: Run time and throughput, baseline vs optimized")
fig.tight_layout(); fig.savefig(OUT + "fig1_time_throughput.png", dpi=150); plt.close(fig)

# ---- Figure 2: CUDA time breakdown (10 profiled batches) ----
cats = ["Memory copy\n(aten::copy_)", "GEMM\n(aten::mm)", "Element-wise\n(add, mul, clamp)", "Reduction\n(sum)"]
base = [52.498, 42.130, 11.006 + 10.915 + 10.446, 1.921]
opt  = [28.767, 40.257, 11.323 + 11.288 + 10.851, 1.919]
x = np.arange(len(cats)); w = 0.38
fig, a = plt.subplots(figsize=(9, 4.2))
b1 = a.bar(x - w / 2, base, w, label="Baseline", color=C_BASE)
b2 = a.bar(x + w / 2, opt, w, label="Optimized", color=C_OPT)
for bars in (b1, b2):
    for b in bars:
        a.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=9)
a.set_xticks(x); a.set_xticklabels(cats); a.set_ylabel("Self CUDA time (ms, 10 batches)")
a.set_title("Figure 2: GPU time by operation type"); a.legend()
fig.tight_layout(); fig.savefig(OUT + "fig2_cuda_time_breakdown.png", dpi=150); plt.close(fig)

# ---- Figure 3: estimated GPU utilization ----
kernel_base = 42.130 + 11.006 + 10.915 + 10.446 + 1.921   # ms per 10 batches
kernel_opt  = 40.257 + 11.323 + 11.288 + 10.851 + 1.919
util_base = kernel_base / (per_batch[0] * 10) * 100
util_opt = min(kernel_opt / (per_batch[1] * 10) * 100, 100)
fig, a = plt.subplots(figsize=(5.5, 4))
bars = a.bar(["Baseline", "Optimized"], [util_base, util_opt], color=[C_BASE, C_OPT], width=0.5)
for b, v in zip(bars, [util_base, util_opt]):
    a.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f} %", ha="center", va="bottom")
a.set_ylim(0, 115); a.set_ylabel("Estimated GPU kernel busy (%)")
a.set_title("Figure 3: Estimated GPU utilization")
fig.tight_layout(); fig.savefig(OUT + "fig3_gpu_utilization_estimate.png", dpi=150); plt.close(fig)
print("util estimates:", util_base, kernel_opt / (per_batch[1] * 10) * 100)

# ---- Figure 4: roofline ----
peak, bw = 39.6e12, 432e9            # RTX 4500 Ada: spec-sheet values (verify)
ridge = peak / bw
ai = np.logspace(-2, 3, 200)
perf = np.minimum(peak, ai * bw) / 1e12
fig, a = plt.subplots(figsize=(7.5, 4.6))
a.loglog(ai, perf, color="black")
pts = {"Element-wise (AI 0.125)": 0.125, "Sum reduction (AI 0.25)": 0.25, "GEMM (AI 249)": 249}
for n, v in pts.items():
    a.plot(v, min(peak, v * bw) / 1e12, "o", ms=9, label=n)
a.axvline(ridge, ls="--", color="gray"); a.text(ridge * 1.05, 0.02, f"ridge ≈ {ridge:.0f}", color="gray")
a.set_xlabel("Arithmetic intensity (FLOP/byte)"); a.set_ylabel("Attainable performance (TFLOP/s)")
a.set_title("Figure 4: Roofline model (RTX 4500 Ada, spec-sheet values)")
a.legend(); a.grid(True, which="both", alpha=0.25)
fig.tight_layout(); fig.savefig(OUT + "fig4_roofline.png", dpi=150); plt.close(fig)

# ---- Figure 5: schematic timeline ----
fig, ax = plt.subplots(2, 1, figsize=(10, 3.8), sharex=True)
cp, kc = "#f0ad4e", "#337ab7"
for i in range(3):
    s = i * 12.5
    ax[0].barh(1, 5.0, left=s, color=cp, edgecolor="k"); ax[0].barh(0, 7.5, left=s + 5.0, color=kc, edgecolor="k")
    ax[1].barh(1, 5.0, left=i * 7.5, color=cp, edgecolor="k"); ax[1].barh(0, 7.5, left=i * 7.5 + 5.0, color=kc, edgecolor="k")
for a, t in zip(ax, ["BEFORE (baseline): copy and compute take turns", "AFTER (optimized): copy of batch i+1 hidden behind compute of batch i"]):
    a.set_yticks([0, 1]); a.set_yticklabels(["Compute", "Copy"]); a.set_title(t, fontsize=10)
ax[1].set_xlabel("time (ms) - schematic only, not a measured trace")
fig.suptitle("Figure 5: Schematic timeline")
fig.tight_layout(); fig.savefig(OUT + "fig5_schematic_timeline.png", dpi=150); plt.close(fig)
