import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ======================================================
# Three-species higher-order interaction (HOI) model
#
# Dynamics follow:
# dN_i/dt = r_i N_i [1 - (N_i - B_ijk N_j N_k)/K_i]
#
# Here we set K_i = 1, so:
# dN_i/dt = r_i N_i (1 - N_i + B_ijk N_j N_k)
#
# This introduces higher-order (triplet) interactions,
# where species i is affected by the PRODUCT of j and k.
#
# Workflow:
# 1. Solve analytically for equilibria (via polynomial)
# 2. Integrate dynamics from random initial condition
# 3. Apply perturbations
# 4. Observe possible tipping between equilibria
# ======================================================

# -----------------------------
# Parameters
# -----------------------------
A, B, C = [-1.112, -0.908, -1.018]  # HOI coefficients
r1 = r2 = r3 = 1.0                  # growth rates
EPS = 1e-10                         # extinction threshold

# -----------------------------
# Analytical step:
# Solve for equilibria
#
# By reducing the system, we obtain a quintic equation
# in N3. Its roots determine candidate equilibria.
# -----------------------------

# Polynomial coefficients for N3
a5 = -A**2 * B**2
a4 =  A**2 * B**2
a3 =  2 * A * B
a2 = -2 * A * B + A * B * C
a1 = -1 + C * (A + B)
a0 =  1 + C

roots = np.roots([a5, a4, a3, a2, a1, a0])

# Keep only real, positive roots
real_positive_roots = [r.real for r in roots if np.isreal(r) and r.real > 0]

# -----------------------------
# Reconstruct full equilibria
# -----------------------------
equilibria = []

for N3 in real_positive_roots:

    denom = 1 - A * B * (N3**2)

    # Avoid singularities
    if np.abs(denom) < 1e-12:
        continue

    N2 = (1 + B * N3) / denom
    N1 = 1 + A * N2 * N3

    # Store equilibrium (positivity can be checked if desired)
    equilibria.append(np.array([N1, N2, N3]))

equilibria = np.array(equilibria)

print("Number of positive equilibria:", len(equilibria))
print("Equilibrium vectors:")
print(equilibria)

# Require multiple equilibria to proceed (bistability/multistability)
if len(equilibria) < 2:
    raise RuntimeError("Not enough equilibria found.")

# -----------------------------
# Dynamical system with extinction threshold
#
# Prevents negative abundances and enforces absorbing state at zero
# -----------------------------
def system(t, x):

    x = x.copy()
    x[x <= EPS] = 0  # enforce extinction

    N1, N2, N3 = x

    dN1 = r1 * N1 * (1 - N1 + A * N2 * N3)
    dN2 = r2 * N2 * (1 - N2 + B * N1 * N3)
    dN3 = r3 * N3 * (1 - N3 + C * N1 * N2)

    dx = np.array([dN1, dN2, dN3])

    # Prevent further decrease once extinct
    dx[x <= EPS] = 0

    return dx

# -----------------------------
# Initial condition
#
# Random positive abundances
# -----------------------------
N0 = np.random.normal(0.4, 0.2, 3)

# -----------------------------
# Phase 1: initial dynamics
# -----------------------------
t_span1 = (0, 50)
t_eval1 = np.linspace(*t_span1, 2000)

sol1 = solve_ivp(system, t_span1, N0, t_eval=t_eval1)

# -----------------------------
# Phase 2: small perturbation
#
# Here we simply re-integrate from final state
# (can modify to include explicit perturbation)
# -----------------------------
final_state = sol1.y[:, -1].copy()
perturbed_state = final_state.copy()

t_span2 = (50, 100)
t_eval2 = np.linspace(*t_span2, 2000)

sol2 = solve_ivp(system, t_span2, perturbed_state, t_eval=t_eval2)

# -----------------------------
# Phase 3: targeted perturbation
#
# Reduce species 1 slightly to test stability
# -----------------------------
final_state = sol2.y[:, -1].copy()

perturbed_state = final_state.copy()
perturbed_state[0] = 0.095   # explicit perturbation on N1

t_span3 = (100, 300)
t_eval3 = np.linspace(*t_span3, 2000)

sol3 = solve_ivp(system, t_span3, perturbed_state, t_eval=t_eval3)

# -----------------------------
# Plot: full time series
#
# Shows response to sequential perturbations
# and possible transitions between equilibria
# -----------------------------
plt.figure(figsize=(8, 5))

# Phase 1
plt.plot(sol1.t, sol1.y[0], color="purple", alpha=0.8)
plt.plot(sol1.t, sol1.y[1], color="teal",   alpha=0.8)
plt.plot(sol1.t, sol1.y[2], color="red",    alpha=0.8)

# Phase 2
plt.plot(sol2.t, sol2.y[0], color="purple", alpha=0.8)
plt.plot(sol2.t, sol2.y[1], color="teal",   alpha=0.8)
plt.plot(sol2.t, sol2.y[2], color="red",    alpha=0.8)

# Phase 3
plt.plot(sol3.t, sol3.y[0], color="purple", alpha=0.8)
plt.plot(sol3.t, sol3.y[1], color="teal",   alpha=0.8)
plt.plot(sol3.t, sol3.y[2], color="red",    alpha=0.8)

plt.axhline(0, color="black", alpha=0.1)

plt.xlabel("Time")
plt.ylabel("Abundance")
plt.title("Higher-order interaction dynamics with extinction threshold")

plt.tight_layout()
plt.show()
