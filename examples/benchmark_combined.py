"""Generate combined benchmark figure: forecast speed and training time side by side."""

import time

import jax
import matplotlib.pyplot as plt
import numpy as np

import orc
from reservoirpy.jax.nodes import Reservoir, Ridge

jax.config.update("jax_enable_x64", True)

N_REPEATS = 5
RES_DIMS = [500, 1000, 2000, 4000]
TRAIN_SIZES = [500, 1000, 2000, 4000, 8000, 16000]
FIXED_RES_DIM = 2000

leak_rate = 0.6
bias = 1.6
embedding_scaling = 0.08
Wr_density = 0.02
Wr_spectral_radius = 0.8
beta = 8e-8
spinup = 200
data_dim = 3

max_train = max(TRAIN_SIZES) + spinup
U_long, _ = orc.data.lorenz63(tN=max_train * 0.03 + 1, dt=0.03)
fcast_steps = int((300 / 0.03) / 2)
U_fcast = U_long[:fcast_steps]

def run_forecast_benchmarks():
    orc_fcast_times, rpy_fcast_times = [], []

    for res_dim in RES_DIMS:
        # --- ORC ---
        esn = orc.forecaster.ESNForecaster(
            data_dim=data_dim, res_dim=res_dim, leak_rate=leak_rate, bias=bias,
            embedding_scaling=embedding_scaling, Wr_density=Wr_density,
            Wr_spectral_radius=Wr_spectral_radius,
        )
        esn, res_states = orc.forecaster.train_RCForecaster(
            esn, U_fcast, beta=beta, spinup=spinup
        )
        init_state = res_states[-1]
        _ = esn.forecast(10, init_state)

        times = []
        for _ in range(N_REPEATS):
            t0 = time.perf_counter()
            out = esn.forecast(fcast_steps, init_state)
            jax.block_until_ready(out)
            times.append(time.perf_counter() - t0)
        orc_fcast_times.append(np.median(times) / fcast_steps * 1000) 

        reservoir = Reservoir(
            lr=leak_rate, units=res_dim,
            Win=esn.embedding.win[0], W=esn.driver.wr[0],
            bias=np.ones(res_dim) * bias,
        )
        rpy_esn = (reservoir >> Ridge(beta)).fit(U_fcast[:-1], U_fcast[1:], warmup=spinup)

        def rpy_forecast(model, steps, x0):
            x = x0
            for _ in range(steps):
                x = model(x)

        rpy_forecast(rpy_esn, 10, U_fcast[-1])  # warmup

        times = []
        for _ in range(N_REPEATS):
            t0 = time.perf_counter()
            rpy_forecast(rpy_esn, fcast_steps, U_fcast[-1])
            times.append(time.perf_counter() - t0)
        rpy_fcast_times.append(np.median(times) / fcast_steps * 1000)

        print(f"  [forecast] res_dim={res_dim}: ORC {orc_fcast_times[-1]:.4f} ms/step, "
              f"RPy {rpy_fcast_times[-1]:.4f} ms/step")

    return orc_fcast_times, rpy_fcast_times

def run_training_benchmarks():

    esn = orc.forecaster.ESNForecaster(
        data_dim=data_dim, res_dim=FIXED_RES_DIM, leak_rate=leak_rate, bias=bias,
        embedding_scaling=embedding_scaling, Wr_density=Wr_density,
        Wr_spectral_radius=Wr_spectral_radius,
    )
    reservoir = Reservoir(
        lr=leak_rate, units=FIXED_RES_DIM,
        Win=esn.embedding.win[0], W=esn.driver.wr[0],
        bias=np.ones(FIXED_RES_DIM) * bias,
    )
    rpy_esn = reservoir >> Ridge(beta)

    orc_train_times, rpy_train_times = [], []

    for n_steps in TRAIN_SIZES:
        U_train = U_long[:n_steps]


        _e, _s = orc.forecaster.train_RCForecaster(esn, U_train, beta=beta, spinup=spinup)
        jax.block_until_ready(_s)

        times = []
        for _ in range(N_REPEATS):
            t0 = time.perf_counter()
            _, rs = orc.forecaster.train_RCForecaster(esn, U_train, beta=beta, spinup=spinup)
            jax.block_until_ready(rs)
            times.append(time.perf_counter() - t0)
        orc_train_times.append(np.median(times))

        rpy_esn.fit(U_train[:-1], U_train[1:], warmup=spinup)

        times = []
        for _ in range(N_REPEATS):
            t0 = time.perf_counter()
            rpy_esn.fit(U_train[:-1], U_train[1:], warmup=spinup)
            times.append(time.perf_counter() - t0)
        rpy_train_times.append(np.median(times))

        print(f"  [training] n_steps={n_steps}: ORC {orc_train_times[-1]:.3f} s, "
              f"RPy {rpy_train_times[-1]:.3f} s")

    return orc_train_times, rpy_train_times

print("Running forecast benchmarks...")
orc_fcast_cpu, rpy_fcast_cpu = run_forecast_benchmarks()

print("\nRunning training benchmarks...")
orc_train_cpu, rpy_train_cpu = run_training_benchmarks()

# HARDCODED GPU FOR PLOTS; ignore/comment out when running on your own computer
orc_fcast_gpu = [0.0134, 0.0153, 0.0189, 0.0370]  # ms/step
rpy_fcast_gpu = [0.1626, 0.1539, 0.1472, 0.1633]  # ms/step

orc_train_gpu = [0.041, 0.056, 0.087, 0.151, 0.271, 0.516]
rpy_train_gpu = [0.525, 0.610, 0.599, 0.687, 0.893, 1.337]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "text.usetex": True,
    "figure.dpi": 300,
})

COLORS = {
    "orc_cpu": "#2563eb",
    "orc_gpu": "#16a34a",
    "rpy_cpu": "#dc2626",
    "rpy_gpu": "#f97316",
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
ax1.plot(RES_DIMS, orc_fcast_cpu, "o-",  color=COLORS["orc_cpu"], markersize=6, linewidth=1.5, label="ORC (CPU)")
ax1.plot(RES_DIMS, orc_fcast_gpu, "^-",  color=COLORS["orc_gpu"], markersize=6, linewidth=1.5, label="ORC (GPU)")
ax1.plot(RES_DIMS, rpy_fcast_cpu, "s--", color=COLORS["rpy_cpu"], markersize=6, linewidth=1.5, label="ReservoirPy (CPU)")
ax1.plot(RES_DIMS, rpy_fcast_gpu, "D--", color=COLORS["rpy_gpu"], markersize=6, linewidth=1.5, label="ReservoirPy (GPU)")
ax1.set_xlabel(r"Reservoir dimension $N_r$")
ax1.set_ylabel(r"Time per forecast step (ms)")
ax1.set_xticks(RES_DIMS)
ax1.grid(True, alpha=0.2, linewidth=0.5)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.text(0.04, 0.96, r"\textbf{(a)}", transform=ax1.transAxes,
         va="top", ha="left", fontsize=13)


ax2.plot(TRAIN_SIZES, orc_train_cpu, "o-",  color=COLORS["orc_cpu"], markersize=6, linewidth=1.5, label="ORC (CPU)")
ax2.plot(TRAIN_SIZES, orc_train_gpu, "^-",  color=COLORS["orc_gpu"], markersize=6, linewidth=1.5, label="ORC (GPU)")
ax2.plot(TRAIN_SIZES, rpy_train_cpu, "s--", color=COLORS["rpy_cpu"], markersize=6, linewidth=1.5, label="ReservoirPy (CPU)")
ax2.plot(TRAIN_SIZES, rpy_train_gpu, "D--", color=COLORS["rpy_gpu"], markersize=6, linewidth=1.5, label="ReservoirPy (GPU)")
ax2.set_xlabel(r"Training samples $N_{\mathrm{train}}$")
ax2.set_ylabel(r"Training time (s)")
ax2.set_xticks(TRAIN_SIZES)
ax2.tick_params(axis="x", rotation=45)
ax2.grid(True, alpha=0.2, linewidth=0.5)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.text(0.04, 0.96, r"\textbf{(b)}", transform=ax2.transAxes,
         va="top", ha="left", fontsize=13)


handles, labels = ax1.get_legend_handles_labels()
order = [2, 3, 0, 1]
fig.legend(
    [handles[i] for i in order], [labels[i] for i in order],
    loc="lower center", ncol=4, frameon=True, fancybox=False,
    edgecolor="black", bbox_to_anchor=(0.5, -0.12),
)

fig.tight_layout()
fig.savefig("imgs/benchmark_results_combined.png", dpi=300, bbox_inches="tight")
fig.savefig("imgs/benchmark_results_combined.pdf", bbox_inches="tight")
print("\nPlot saved to imgs/benchmark_results_combined.png and .pdf")
