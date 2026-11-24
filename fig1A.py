import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -----------------------------
# Parameters
# -----------------------------
r = 1.0   # intrinsic growth rate
K = 1.0   # carrying capacity
A = 0.5   # Allee threshold; bistability requires 0 < A < K

# -----------------------------
# Model: single-species Allee effect
# dN/dt = r * N * (1 - N/K) * (N/A - 1)
# Produces two stable equilibria (N=0 and N=K) separated by
# an unstable threshold at N=A
# -----------------------------
def allee(t, N):
    return r * N * (1 - N / K) * (N / A - 1)

# -----------------------------
# Initial condition: above Allee threshold, heading to K
# -----------------------------
N0 = [0.65]

# -----------------------------
# Phase 1: simulate until perturbation
# -----------------------------
t_span1 = (0, 10)
t_eval1 = np.linspace(*t_span1, 500)
sol1 = solve_ivp(allee, t_span1, N0, t_eval=t_eval1)
N_end = sol1.y[0, -1]

# -----------------------------
# Pulse perturbation: push population below Allee threshold
# -----------------------------
N_end -= 0.52
N_perturbed = [max(N_end, 0)]  # enforce non-negativity

# -----------------------------
# Phase 2: simulate post-perturbation dynamics
# System is now below threshold and collapses to N=0
# -----------------------------
t_span2 = (10, 25)
t_eval2 = np.linspace(*t_span2, 500)
sol2 = solve_ivp(allee, t_span2, N_perturbed, t_eval=t_eval2)

# -----------------------------
# Concatenate phases for continuous time series
# -----------------------------
t = np.concatenate([sol1.t, sol2.t])
N = np.concatenate([sol1.y[0], sol2.y[0]])

# -----------------------------
# Plot: time series with perturbation marked
# -----------------------------
plt.figure(figsize=(6, 4.5))
plt.axvline(10, linestyle="--", color="gray", linewidth=2.5)  # perturbation time
plt.plot(t, N, color="teal", linewidth=2.5)
plt.axhline(0, color="gray", linewidth=0.5)
plt.xlabel("Time")
plt.ylabel("Abundance")
plt.tight_layout()
plt.show()

exit()

# ======================================================
# Phase line (1D phase portrait)
# Shows flow direction and stability of equilibria
# ======================================================
N_vals = np.linspace(0, 1.2, 400)
dN = allee(0, N_vals)

plt.figure(figsize=(5.5, 4))
plt.axhline(0, color="black", linewidth=1)

# Arrows indicating direction of flow along the N axis
for n in np.linspace(0.05, 1.15, 20):
    growth = allee(0, n)
    direction = np.sign(growth)
    plt.arrow(
        n, 0,
        0.04 * direction, 0,
        head_width=0.02,
        head_length=0.03,
        fc="gray",
        ec="gray"
    )

# Equilibria: filled = stable, open = unstable
equilibria = [0, A, K]
for eq in equilibria:
    stability = allee(0, eq + 1e-4)
    if stability < 0:
        plt.scatter(eq, 0, color="black", s=60, zorder=3)       # stable
    else:
        plt.scatter(eq, 0, facecolors="white",
                    edgecolors="black", s=60, zorder=3)          # unstable

plt.xlabel(r"$N$")
plt.yticks([])
plt.xlim(-0.05, 1.25)
plt.tight_layout()
plt.show()
