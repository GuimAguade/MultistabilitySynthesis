import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ======================================================
# Two-species competitive Lotka–Volterra model
# As presented in:
# Petraitis (2013) — "Multiple Stable States in Natural Ecosystems"
#
# This classic model captures competitive exclusion and
# alternative stable states depending on interaction strengths.
# ======================================================

# -----------------------------
# Parameters
# -----------------------------
r1 = r2 = 1.0     # intrinsic growth rates
K1 = K2 = 1.0     # carrying capacities

alpha12 = 1.1     # effect of species 2 on species 1
alpha21 = 1.2     # effect of species 1 on species 2

# -----------------------------
# Model: competitive dynamics
#
# dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
# dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)
#
# Interpretation:
# - Logistic growth with interspecific competition
# - alpha_ij > 1 implies strong competition
#
# This parameter regime can produce bistability:
# each species excludes the other depending on initial conditions
# -----------------------------
def competition(t, N):
    N1, N2 = N

    dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
    dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)

    return [dN1dt, dN2dt]

# -----------------------------
# Initial conditions
#
# Slight asymmetry determines which species dominates
# -----------------------------
N0 = [0.4, 0.6]

# -----------------------------
# Phase 1: simulate pre-perturbation dynamics
#
# System approaches one of the alternative stable states
# (competitive exclusion of one species)
# -----------------------------
t_span1 = (0, 75)
t_eval1 = np.linspace(*t_span1, 500)

sol1 = solve_ivp(competition, t_span1, N0, t_eval=t_eval1)

N1_end, N2_end = sol1.y[0, -1], sol1.y[1, -1]

# -----------------------------
# Pulse perturbation:
# reduce the currently dominant species
#
# This tests whether the system can switch basins
# of attraction (i.e., flip dominance)
# -----------------------------
if N1_end > N2_end:
    N1_end -= 0.8
else:
    N2_end -= 0.8

N_perturbed = [N1_end, N2_end]

# -----------------------------
# Phase 2: simulate post-perturbation dynamics
#
# Possible outcome:
# - recovery of original dominant species
# - or switch to the alternative stable state
# -----------------------------
t_span2 = (75, 100)
t_eval2 = np.linspace(*t_span2, 500)

sol2 = solve_ivp(competition, t_span2, N_perturbed, t_eval=t_eval2)

# -----------------------------
# Concatenate trajectories
# -----------------------------
t  = np.concatenate([sol1.t, sol2.t])
N1 = np.concatenate([sol1.y[0], sol2.y[0]])
N2 = np.concatenate([sol1.y[1], sol2.y[1]])

# -----------------------------
# Plot: time series with perturbation
# -----------------------------
plt.figure(figsize=(6, 4.5))

plt.axvline(75, linestyle="--", color="gray", linewidth=2.5)

plt.plot(t, N1, color="firebrick", linewidth=2.5, label="Species 1")
plt.plot(t, N2, color="teal",      linewidth=2.5, label="Species 2")

plt.axhline(0, color="gray", linewidth=0.5)

plt.xlabel("Time")
plt.ylabel("Abundance")

# plt.legend()  # optional

plt.tight_layout()
plt.show()


# ======================================================
# Phase plane (2D phase portrait)
#
# Visualizes:
# - flow direction
# - basins of attraction
# - alternative stable equilibria
# ======================================================

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Reuse parameters
# -----------------------------
r1 = r2 = 1.0
K1 = K2 = 1.0
alpha12 = 1.1
alpha21 = 1.2

# -----------------------------
# Grid for phase plane
# -----------------------------
n_points = 14   # sparse grid for clarity

N1grid = np.linspace(0, 1.2, n_points)
N2grid = np.linspace(0, 1.2, n_points)

X, Y = np.meshgrid(N1grid, N2grid)

# -----------------------------
# Vector field
#
# Computes direction of change at each point
# -----------------------------
dN1 = r1 * X * (1 - (X + alpha12 * Y) / K1)
dN2 = r2 * Y * (1 - (Y + alpha21 * X) / K2)

# -----------------------------
# Normalize vectors for visualization
# (only direction matters, not magnitude)
# -----------------------------
mag = np.sqrt(dN1**2 + dN2**2)
mag[mag == 0] = 1

dN1n = dN1 / mag
dN2n = dN2 / mag

# -----------------------------
# Plot: phase plane
# -----------------------------
plt.figure(figsize=(5.5, 5))

plt.quiver(
    X, Y, dN1n, dN2n,
    color="lightgray",
    angles="xy",
    scale_units="xy",
    scale=15,
    width=0.004,
    zorder=1
)



plt.xlabel(r"$N_1$")
plt.ylabel(r"$N_2$")

plt.xlim(-0.1, 1.3)
plt.ylim(-0.1, 1.3)

plt.tight_layout()
plt.show()
