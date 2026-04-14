import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ======================================================
# Two-species mutualistic system (plant–pollinator)
# Inspired by a 2-D simplification of the model in:
# Lever et al. (2014) Ecology Letters
# ======================================================

# -----------------------------
# Parameters
# -----------------------------
d      = 0.97   # pollinator mortality offset (reduces intrinsic growth)
gamma  = 3.0    # strength of mutualistic interaction

delta_A = -0.35 # perturbation applied to pollinator population

r   = 0.2       # intrinsic growth rate (shared baseline)
h   = 0.5       # handling time (saturating mutualism)
C   = 1.0       # intraspecific competition strength
mu  = 1e-4      # small immigration term (prevents extinction at exactly zero)

# -----------------------------
# Model: plant–pollinator dynamics
#
# dP/dt = rP + mutualistic gain from A - density dependence + immigration
# dA/dt = (r - d)A + mutualistic gain from P - density dependence + immigration
#
# Mutualistic benefit saturates via a Holling type II functional response:
#     gamma * X / (1 + h * gamma * X)
#
# This system can exhibit bistability and tipping points,
# where perturbations may shift the system between
# high-abundance coexistence and low-abundance states.
# -----------------------------
def plant_pollinator(t, y):
    P, A = y

    dP = (
        r * P
        + P * (gamma * A / (1 + h * gamma * A))
        - C * P**2
        + mu
    )

    dA = (
        (r - d) * A
        + A * (gamma * P / (1 + h * gamma * P))
        - C * A**2
        + mu
    )

    return [dP, dA]

# -----------------------------
# Initial conditions: both species established
# -----------------------------
P0, A0 = 0.5, 0.5

# -----------------------------
# Phase 1: simulate pre-perturbation dynamics
# System typically approaches a high-abundance coexistence state
# -----------------------------
t_span1 = (0, 50)
t_eval1 = np.linspace(*t_span1, 1000)

sol1 = solve_ivp(
    plant_pollinator,
    t_span1,
    [P0, A0],
    t_eval=t_eval1,
    method="RK45",
    rtol=1e-8,
    atol=1e-10
)

P_end = sol1.y[0, -1]
A_end = sol1.y[1, -1]

# -----------------------------
# Pulse perturbation:
# reduce pollinator abundance only
# (can push system across a tipping point)
# -----------------------------
A_perturbed = max(A_end + delta_A, 0)  # enforce non-negativity

# -----------------------------
# Phase 2: simulate post-perturbation dynamics
# Depending on perturbation strength, system may:
# - recover to coexistence, or
# - collapse to a low-abundance state
# -----------------------------
t_span2 = (50, 100)
t_eval2 = np.linspace(*t_span2, 1000)

sol2 = solve_ivp(
    plant_pollinator,
    t_span2,
    [P_end, A_perturbed],
    t_eval=t_eval2,
    method="RK45",
    rtol=1e-8,
    atol=1e-10
)

# -----------------------------
# Concatenate phases for continuous trajectories
# -----------------------------
t = np.concatenate([sol1.t, sol2.t])
P = np.concatenate([sol1.y[0], sol2.y[0]])
A = np.concatenate([sol1.y[1], sol2.y[1]])

# -----------------------------
# Plot: time series with perturbation marked
# -----------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(t, P, color="mediumseagreen", linewidth=2.5, alpha=0.8,
        label="Plant (P)")
ax.plot(t, A, color="mediumpurple", linewidth=2.5, alpha=0.8,
        label="Pollinator (A)")

ax.axvline(50, linestyle="--", color="gray", linewidth=2,
           label="Perturbation")
ax.axhline(0, color="gray", linewidth=0.5)

ax.set_xlabel("Time", fontsize=12)
ax.set_ylabel("Abundance", fontsize=12)

# ax.legend(fontsize=11)  # optional

plt.tight_layout()
plt.savefig("trajectory.png", dpi=150, bbox_inches="tight")
plt.show()
