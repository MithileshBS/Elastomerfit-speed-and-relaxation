import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# ============================================================
# FIXED AB + FIT 3 PRONY BRANCHES ON FULL STRESS HISTORY
# + TIME-SCALING STUDY
# + NUMERICAL SAFEGUARD FOR SMALL dt/tau
# ============================================================

FILE_PATH = "loading_relaxation_file1.csv"

TIME_COL = "time"
LAMBDA_COL = "lambda"
STRESS_COL = "true_stress"

# ------------------------------------------------------------+
# Fixed Arruda-Boyce parameters from loading fit
# ------------------------------------------------------------
mu_fix = 0.5503
N_fix = 1.5330

# ------------------------------------------------------------
# Initial guesses from your 3-branch g(t) fit
# ------------------------------------------------------------
g1_init = 0.0828208039
#tau1_init = 1.58437879
tau1_init = 0.01

g2_init = 0.0871650399
#tau2_init = 53.1987869761
tau2_init = 0.1

g3_init = 0.0692907579
#tau3_init = 1610.6328528685
tau3_init = 1.0

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(FILE_PATH)

time = df[TIME_COL].to_numpy(dtype=float)
lam = df[LAMBDA_COL].to_numpy(dtype=float)
stress_exp = df[STRESS_COL].to_numpy(dtype=float)

if np.any(np.diff(time) <= 0):
    raise ValueError("Time must be strictly increasing.")

# ============================================================
# ARRUDA-BOYCE MODEL
# ============================================================
a1 = 1.0 / 10.0
a2 = 11.0 / 350.0
a3 = 19.0 / 1750.0
a4 = 519.0 / 134750.0

def sigma_AB5_cauchy_muN(lam, mu, N):
    lam = np.asarray(lam, dtype=float)
    lam = np.maximum(lam, 1e-12)
    N = max(float(N), 1.0 + 1e-12)

    I1 = lam**2 + 2.0 / lam
    Sfac = lam**2 - lam**(-1.0)

    B = (
        0.5
        + a1 * (I1 / N)
        + a2 * (I1**2 / N**2)
        + a3 * (I1**3 / N**3)
        + a4 * (I1**4 / N**4)
    )

    return 2.0 * mu * Sfac * B

sigma0 = sigma_AB5_cauchy_muN(lam, mu_fix, N_fix)

# ============================================================
# DETECT LOADING / RELAXATION AND PEAK-EMPHASIS WEIGHTS
# ============================================================
dlam = np.diff(lam, prepend=lam[0])
loading_mask = dlam > 1e-8

loading_indices = np.where(loading_mask)[0]
peak_mask = np.zeros_like(time, dtype=bool)

if len(loading_indices) > 0:
    n_peak = max(5, int(0.15 * len(loading_indices)))
    peak_mask[loading_indices[-n_peak:]] = True

weights = np.ones_like(time, dtype=float)
weights[loading_mask] = 3.0
weights[peak_mask] = 10.0

# ============================================================
# HELPER FOR STABLE COMPUTATION OF b = g*(1-exp(-x))/x
# ============================================================
def stable_branch_factors(g, dt, tau):
    tau_eff = max(float(tau), 1e-12)
    x = dt / tau_eff
    a = np.exp(-x)

    if abs(x) < 1e-12:
        b = g
    else:
        b = g * (1.0 - a) / x

    return a, b

# ============================================================
# 3-BRANCH COMBINED MODEL
# params = [g1, g2, g3, tau1, tau2, tau3]
# ============================================================
def combined_model_3branch(params, time, sigma0):
    g1, g2, g3, tau1, tau2, tau3 = params
    g_inf = 1.0 - (g1 + g2 + g3)

    sigma_pred = np.zeros_like(time, dtype=float)

    Q1 = 0.0
    Q2 = 0.0
    Q3 = 0.0

    sigma_pred[0] = g_inf * sigma0[0]

    for n in range(1, len(time)):
        dtn = time[n] - time[n - 1]
        dsigma0 = sigma0[n] - sigma0[n - 1]

        a_1, b_1 = stable_branch_factors(g1, dtn, tau1)
        a_2, b_2 = stable_branch_factors(g2, dtn, tau2)
        a_3, b_3 = stable_branch_factors(g3, dtn, tau3)

        Q1 = a_1 * Q1 + b_1 * dsigma0
        Q2 = a_2 * Q2 + b_2 * dsigma0
        Q3 = a_3 * Q3 + b_3 * dsigma0

        sigma_pred[n] = g_inf * sigma0[n] + Q1 + Q2 + Q3

    return sigma_pred

# ============================================================
# RESIDUALS
# ============================================================
def residuals(params, time, sigma0, stress_exp, weights):
    g1, g2, g3, tau1, tau2, tau3 = params

    sigma_pred = combined_model_3branch(params, time, sigma0)

    scale = np.maximum(np.abs(stress_exp), 0.05 * np.max(np.abs(stress_exp)))
    res = np.sqrt(weights) * (sigma_pred - stress_exp) / scale

    penalty = np.zeros(1, dtype=float)
    gsum = g1 + g2 + g3
    if gsum >= 0.999999:
        penalty[0] = 1e3 * (gsum - 0.999999)

    return np.concatenate([res, penalty])

# ============================================================
# FIT
# ============================================================
x0 = np.array([
    g1_init, g2_init, g3_init,
    tau1_init, tau2_init, tau3_init
], dtype=float)

lower_bounds = np.array([0, 0, 0, 1e-8, 1e-8, 1e-8], dtype=float)
upper_bounds = np.array([0.999999, 0.999999, 0.999999, 1e8, 1e8, 1e8], dtype=float)

result = least_squares(
    residuals,
    x0,
    bounds=(lower_bounds, upper_bounds),
    args=(time, sigma0, stress_exp, weights),
    method="trf",
    max_nfev=50000,
    verbose=2
)

g1_fit, g2_fit, g3_fit, tau1_fit, tau2_fit, tau3_fit = result.x

# ------------------------------------------------------------
# Sort branches by tau
# ------------------------------------------------------------
order = np.argsort([tau1_fit, tau2_fit, tau3_fit])

taus = np.array([tau1_fit, tau2_fit, tau3_fit])[order]
gs = np.array([g1_fit, g2_fit, g3_fit])[order]

tau1_fit, tau2_fit, tau3_fit = taus
g1_fit, g2_fit, g3_fit = gs

g_inf_fit = 1.0 - (g1_fit + g2_fit + g3_fit)

params_sorted = np.array([
    g1_fit, g2_fit, g3_fit,
    tau1_fit, tau2_fit, tau3_fit
], dtype=float)

stress_fit = combined_model_3branch(params_sorted, time, sigma0)

rmse = np.sqrt(np.mean((stress_fit - stress_exp) ** 2))
mae = np.mean(np.abs(stress_fit - stress_exp))

# ============================================================
# PRINT RESULTS
# ============================================================
print("\n==============================")
print("FIXED AB + 3-BRANCH STRESS FIT")
print("==============================")
print("Success:", result.success)
print("Message:", result.message)

print(f"\ng1    = {g1_fit:.12g}")
print(f"tau1  = {tau1_fit:.12g} s")
print(f"g2    = {g2_fit:.12g}")
print(f"tau2  = {tau2_fit:.12g} s")
print(f"g3    = {g3_fit:.12g}")
print(f"tau3  = {tau3_fit:.12g} s")
print(f"g_inf = {g_inf_fit:.12g}")

print(f"\nRMSE  = {rmse:.12g}")
print(f"MAE   = {mae:.12g}")

# ============================================================
# SAVE MAIN FIT
# ============================================================
pd.DataFrame({
    "time": time,
    "lambda": lam,
    "true_stress_exp": stress_exp,
    "sigma0_AB_fixed": sigma0,
    "stress_fit_3branch": stress_fit
}).to_csv("fixedAB_3branch_fullstress_fit.csv", index=False)

# ============================================================
# PLOT: MAIN FIT - STRESS VS TIME
# ============================================================
plt.figure(figsize=(10, 6))
plt.plot(time, stress_exp, ".", ms=3, label="Experimental")
plt.plot(time, sigma0, "--", lw=2, label="Fixed AB sigma0")
plt.plot(time, stress_fit, "-", lw=2, label="Fixed AB + 3-branch fit")
plt.xlabel("Time (s)")
plt.ylabel("True Stress (MPa)")
plt.title("Fixed AB + 3-Branch Fit: Stress vs Time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# PLOT: MAIN FIT - STRESS VS STRETCH
# ============================================================
plt.figure(figsize=(10, 6))
plt.plot(lam, stress_exp, ".", ms=3, label="Experimental")
plt.plot(lam, sigma0, "--", lw=2, label="Fixed AB sigma0")
plt.plot(lam, stress_fit, "-", lw=2, label="Fixed AB + 3-branch fit")
plt.xlabel("Stretch λ")
plt.ylabel("True Stress (MPa)")
plt.title("Fixed AB + 3-Branch Fit: Stress vs Stretch")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# TIME-SCALING STUDY
# same lambda history, same fitted parameters, only change time
# ============================================================
time_factors = [0.1, 0.5, 1.0, 2.0, 10.0]   # 0.1 = 10x faster, 10 = 10x slower

time_scale_results = {
    "lambda": lam,
    "time_original": time
}

# ------------------------------------------------------------
# PLOT: EFFECT OF TIME SCALING ON STRESS VS STRETCH
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(lam, stress_exp, "k.", ms=3, label="Experimental")

for fac in time_factors:
    time_scaled = time * fac
    stress_scaled = combined_model_3branch(params_sorted, time_scaled, sigma0)

    time_scale_results[f"stress_time_x_{fac}"] = stress_scaled
    plt.plot(lam, stress_scaled, lw=2, label=f"time x {fac}")

plt.plot(lam, sigma0, "k--", lw=2, label="AB only")
plt.xlabel("Stretch λ")
plt.ylabel("True Stress (MPa)")
plt.title("Effect of Time Scaling on AB + 3-Branch Response")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# PLOT: EFFECT OF TIME SCALING ON STRESS VS TIME
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))

for fac in time_factors:
    time_scaled = time * fac
    stress_scaled = combined_model_3branch(params_sorted, time_scaled, sigma0)
    plt.plot(time_scaled, stress_scaled, lw=2, label=f"time x {fac}")

plt.xlabel("Scaled Time (s)")
plt.ylabel("True Stress (MPa)")
plt.title("Stress vs Time for Different Time Scales")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# SAVE TIME-SCALING STUDY
# ------------------------------------------------------------
pd.DataFrame(time_scale_results).to_csv("time_scaling_study.csv", index=False)

# ============================================================
# OPTIONAL: PRINT INTERPRETATION GUIDE
# ============================================================
print("\n==============================")
print("TIME-SCALING INTERPRETATION")
print("==============================")
print("time x 0.1  -> much faster loading, usually stiffer / higher stress")
print("time x 0.5  -> faster loading")
print("time x 1.0  -> original fitted response")
print("time x 2.0  -> slower loading")
print("time x 10.0 -> much slower loading, usually softer / lower stress")