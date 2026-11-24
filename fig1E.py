import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ======================================================
# Multi-species plant–pollinator community
# Based on the same 2N-dimensional mutualistic model as:
# Lever et al. (2014) Ecology Letters
#
# Here, interactions are NOT nested:
# instead, they are sampled (approximately) at random
# for simplicity and illustrative purposes.
# ======================================================

# -----------------------------
# Parameters
# -----------------------------
d       = 1.215   # pollinator mortality offset
h       = 0.5     # handling time (saturating mutualism)
C       = 1.0     # intraspecific competition strength
mu      = 1e-4    # immigration (avoids exact extinction)
delta_A = -0.3    # perturbation applied to pollinators

n_P     = 4       # number of plant species
n_A     = 4       # number of pollinator species

# -----------------------------
# Species-specific growth rates
#
# Drawn from a uniform distribution to introduce heterogeneity
# -----------------------------
rng = np.random.default_rng(seed=42)

r_P = rng.uniform(0.15, 0.25, n_P)  # plants
r_A = rng.uniform(0.15, 0.25, n_A)  # pollinators

# -----------------------------
# Interaction matrix (gamma)
#
# gamma[i,j] = strength of interaction between plant i and pollinator j
#
# Here:
# - Mostly weak, homogeneous interactions
# - Slightly stronger self-pairing (diagonal)
#
# This is NOT a nested structure — interactions are simplified
# and only loosely structured for illustration.
# -----------------------------
gamma = 1.2 * np.ones((n_P, n_A))
np.fill_diagonal(gamma, 0.5)

# -----------------------------
# Model: multi-species mutualistic dynamics
#
# For each plant i:
# dP_i/dt = growth + mutualistic gain - competition + immigration
#
# For each pollinator j:
# dA_j/dt = (growth - mortality) + mutualistic gain - competition + immigration
#
# Mutualistic gain saturates via:
#     interaction / (1 + h * interaction)
#
# where interaction = sum of partner abundances weighted by gamma
# -----------------------------
def community(t, y):

    P = y[:n_P]     # plant abundances
    A = y[n_P:]     # pollinator abundances

    dydt = np.zeros(n_P + n_A)

    # Plant dynamics
    for i in range(n_P):
        interaction_P = np.sum(gamma[i, :] * A)

        dydt[i] = (
            r_P[i] * P[i]
            + P[i] * (interaction_P / (1 + h * interaction_P))
            - C * P[i]**2
            + mu
        )

    # Pollinator dynamics
    for j in range(n_A):
        interaction_A = np.sum(gamma[:, j] * P)

        dydt[n_P + j] = (
            (r_A[j] - d) * A[j]
            + A[j] * (interaction_A / (1 + h * interaction_A))
            - C * A[j]**2
            + mu
        )

    return dydt

# -----------------------------
# Initial conditions
#
# All species start at intermediate abundance
# -----------------------------
P0 = np.random.uniform(0.4, 0.6, n_P)
A0 = np.random.uniform(0.4, 0.6, n_A)

y0 = np.concatenate([P0, A0])

# -----------------------------
# Phase 1: simulate pre-perturbation dynamics
#
# System evolves toward a coexistence state
# shaped by mutualism and competition
# -----------------------------
t_span1 = (0, 30)
t_eval1 = np.linspace(*t_span1, 1000)

sol1 = solve_ivp(
    community,
    t_span1,
    y0,
    t_eval=t_eval1,
    method="RK45",
    rtol=1e-8,
    atol=1e-10
)

P_end = sol1.y[:n_P, -1]
A_end = sol1.y[n_P:, -1]

# -----------------------------
# Pulse perturbation:
# reduce ALL pollinator species
#
# This mimics a shock affecting one trophic level
# and can trigger cascading effects on plants
# -----------------------------
A_perturbed = np.maximum(A_end + delta_A, 0)

y1 = np.concatenate([P_end, A_perturbed])

# -----------------------------
# Phase 2: simulate post-perturbation dynamics
#
# Possible outcomes:
# - recovery to coexistence
# - partial collapse (some species extinct)
# - full collapse of mutualistic system
# -----------------------------
t_span2 = (30, 100)
t_eval2 = np.linspace(*t_span2, 1000)

sol2 = solve_ivp(
    community,
    t_span2,
    y1,
    t_eval=t_eval2,
    method="RK45",
    rtol=1e-8,
    atol=1e-10
)

# -----------------------------
# Concatenate time series
# -----------------------------
t     = np.concatenate([sol1.t, sol2.t])
P_all = np.concatenate([sol1.y[:n_P, :], sol2.y[:n_P, :]], axis=1)
A_all = np.concatenate([sol1.y[n_P:, :], sol2.y[n_P:, :]], axis=1)

# -----------------------------
# Plot: time series for all species
#
# Green tones: plants
# Purple tones: pollinators
# -----------------------------
green_shades  = ["#2d6a4f", "#52b788", "#95d5b2", "#d8f3dc"]
purple_shades = ["#3d1a78", "#7b2d8b", "#b05fbf", "#d4a5e0"]

fig, ax = plt.subplots(figsize=(7, 4.5))

# Plant trajectories
for i in range(n_P):
    ax.plot(
        t, P_all[i],
        color=green_shades[i],
        linewidth=2.0,
        alpha=0.8,
        label=f"Plant $P_{i+1}$"
    )

# Pollinator trajectories
for j in range(n_A):
    ax.plot(
        t, A_all[j],
        color=purple_shades[j],
        linewidth=2.0,
        alpha=0.8,
        label=f"Pollinator $A_{j+1}$"
    )

# Mark perturbation time
ax.axvline(30, linestyle="--", color="gray", linewidth=2)
ax.axhline(0, color="gray", linewidth=0.5)

ax.set_xlabel("Time", fontsize=12)
ax.set_ylabel("Abundance", fontsize=12)

plt.tight_layout()
plt.savefig("trajectory_multispecies.png", dpi=150, bbox_inches="tight")
plt.show()
