import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

np.random.seed(42)

# -----------------------------
# Parameters (5 species)
# Each r_i, K_i, A_i is drawn from Normal(mean, std=0.1)
# where mean matches the 1-D model values
# -----------------------------
n_species = 5

r = np.random.normal(loc=1.0, scale=0.1, size=n_species)
K = np.random.normal(loc=1.0, scale=0.1, size=n_species)
A = np.random.normal(loc=0.5, scale=0.1, size=n_species)

# Cooperation matrix: 1 on diagonal (self), 0.1 off-diagonal (cross-species facilitation)
C = np.full((n_species, n_species), 0.1)
np.fill_diagonal(C, 1.0)

print("Parameters:")
for i in range(n_species):
    print(f"  Species {i+1}: r={r[i]:.3f}, K={K[i]:.3f}, A={A[i]:.3f}")

# -----------------------------
# Coupled model
# dN_i/dt = r_i * N_i * (1 - N_i/K_i) * ((sum_jC_ijN_j)/A_i - 1)
# The Allee term uses the weighted abundance across all species,
# capturing positive (facilitation) coupling: each species benefits
# from the collective presence of all others through the matrix C.
# -----------------------------
def coupled_allee(t, N):
    N = np.maximum(N, 0)              # guard against tiny negatives
    weighted = C @ N                  # weighted_i = sum_j C_ij * N_j
    dN = r * N * (1 - N / K) * (weighted / A - 1)
    return dN

# -----------------------------
# Initial conditions (same relative starting point as 1-D model)
# -----------------------------
N0 = np.full(n_species, 0.65)

# -----------------------------
# First simulation (0 → 10)
# -----------------------------
t_span1 = (0, 10)
t_eval1 = np.linspace(*t_span1, 500)
sol1 = solve_ivp(coupled_allee, t_span1, N0, t_eval=t_eval1,
                 method='RK45', rtol=1e-8, atol=1e-10)

N_end = sol1.y[:, -1].copy()

# -----------------------------
# Apply press perturbation (same magnitude as 1-D: -0.52)
# Distributed equally across all species
# -----------------------------
perturbation = 4 / n_species        # split evenly so total drop = 0.52
N_end -= perturbation
N_perturbed = np.maximum(N_end, 0)

print(f"\nPost-perturbation N: {N_perturbed}")

# -----------------------------
# Second simulation (10 → 25)
# -----------------------------
t_span2 = (10, 25)
t_eval2 = np.linspace(*t_span2, 500)
sol2 = solve_ivp(coupled_allee, t_span2, N_perturbed, t_eval=t_eval2,
                 method='RK45', rtol=1e-8, atol=1e-10)

# -----------------------------
# Combine time series
# -----------------------------
t = np.concatenate([sol1.t, sol2.t])
N_all = np.concatenate([sol1.y, sol2.y], axis=1)   # shape: (5, T)
N_total = N_all.sum(axis=0)                         # total community abundance

# -----------------------------
# Colour palette — teal shades from dark to light
# -----------------------------
colors = ["#003d3d", "#006666", "#008080", "#33b2b2", "#80d4d4"]
labels = [f"Species {i+1}" for i in range(n_species)]


# ======================================================
# Plot 2All species trajectories together
# ======================================================
fig, ax = plt.subplots(figsize=(7, 4.5))

for i in range(n_species):
    ax.plot(t, N_all[i], color=colors[i], linewidth=2.2, label=labels[i], alpha=0.9)
ax.axvline(10, linestyle="--", color="gray", linewidth=2, alpha=0.7)
ax.axhline(0, color="gray", linewidth=0.5)
ax.set_ylabel("Abundance")
ax.set_xlabel("Time")
#ax.legend(fontsize=9, frameon=False, ncol=2)
ax.set_ylim(-0.05, 1.3)

plt.tight_layout()
plt.savefig("allee_5species_community.png",
            dpi=150, bbox_inches="tight")
plt.show()

