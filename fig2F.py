import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ======================================================
# High-dimensional competitive community dynamics
# Based on simulations in:
# Aguadé-Gorgorió & Kéfi (2024), Journal of Physics: Complexity
#
# Workflow:
# 1. Generate a random interaction matrix
# 2. Identify two alternative stable communities (cliques)
# 3. Combine them into a single system
# 4. Let them compete under different initial conditions
# 5. Apply perturbations to test resilience and tipping behavior
# ======================================================

# -----------------------------
# Global parameters
# -----------------------------
muA   = -0.75   # mean interaction strength (competition-dominated)
stdA  = 0.5     # variability of interactions
S     = 50      # total number of species in pool

temps1 = 2000   # long integration to approach equilibrium
temps2 = 10     # short refinement integration

EPS   = 1e-8    # extinction threshold (numerical)
TOL   = 1e-4    # tolerance for steady-state detection

T_PRE        = 40   # time before perturbation
T_POST       = 20   # time after perturbation
PERTURBATION = 0.2  # pulse perturbation: reduce abundance to 20%

# -----------------------------
# Model: generalized Lotka–Volterra dynamics
#
# dx_i/dt = x_i * (1 - x_i + Σ_j A_ij x_j)
#
# - Logistic self-regulation (1 - x_i)
# - Pairwise interactions via matrix A
# - Competitive structure can generate multiple stable states
# -----------------------------
def run(x0, t0, tmax, matrix, EPS):

    def eqs(t, x):
        dx = x * (1 - x + matrix @ x)

        # Prevent negative drift for extinct species
        dx[x <= EPS] = np.maximum(0, dx[x <= EPS])

        return dx

    sol = solve_ivp(
        fun=eqs,
        t_span=(t0, tmax),
        y0=x0,
        dense_output=True,
        max_step=1.0
    )

    # Dense time grid for smooth trajectories
    t_eval = np.linspace(t0, tmax, int((tmax - t0) * 2) + 2)

    return t_eval, sol.sol(t_eval)

# -----------------------------
# Identify alternative stable states (cliques)
#
# Procedure:
# - Run system from random initial condition
# - Check convergence to steady state
# - Extract surviving species (coexisting subset)
# - Repeat until two distinct communities are found
# -----------------------------
def simulate_once(A, temps1, temps2, EPS, TOL):

    x0 = np.random.random(S)

    # Long transient
    _, traj1 = run(x0, 0, temps1, A, EPS)
    final1   = traj1[:, -5]

    # Short refinement
    _, traj2 = run(final1, 0, temps2, A, EPS)
    final2   = traj2[:, -5]

    # Check convergence
    if np.allclose(final1, final2, atol=TOL):

        survivors = np.where(final1 > TOL)[0]

        # Discard trivial extinction
        if len(survivors) == 0:
            return None, None

        # Extract reduced interaction matrix for surviving species
        A_reduced = A[np.ix_(survivors, survivors)]

        return survivors, A_reduced

    return None, None

# -----------------------------
# Search for two distinct stable communities
# -----------------------------
print("Searching for two alternative stable states…")

A = np.random.normal(muA, stdA, size=(S, S))
np.fill_diagonal(A, 0.0)

results    = []
max_trials = 500

for trial in range(max_trials):

    survivors, A_reduced = simulate_once(A, temps1, temps2, EPS, TOL)

    if survivors is None:
        continue

    if len(results) == 0:
        results.append((survivors, A_reduced))

    else:
        prev_survivors, _ = results[0]

        overlap = len(set(survivors) & set(prev_survivors))
        total   = len(set(survivors) | set(prev_survivors))

        # Accept if communities are not identical
        if overlap < total:
            results.append((survivors, A_reduced))
            print(f"  Accepted second state (overlap {overlap}/{total}) after {trial+1} trials")
            break
        else:
            print(f"  Rejected: full overlap ({overlap}/{total})")

if len(results) < 2:
    raise RuntimeError("Could not find two distinct stable states within max_trials.")

survivors1, _ = results[0]
survivors2, _ = results[1]

print(f"\nState 1: {len(survivors1)} survivors")
print(f"State 2: {len(survivors2)} survivors")
print(f"Overlap: {len(set(survivors1) & set(survivors2))} species")

# -----------------------------
# Build combined system
#
# Merge both communities into a shared species pool
# This allows direct competition between alternative states
# -----------------------------
all_survivors = np.unique(np.concatenate([survivors1, survivors2]))
A_star        = A[np.ix_(all_survivors, all_survivors)]
N             = len(all_survivors)

# Map original species indices → reduced system indices
index_map = {s: i for i, s in enumerate(all_survivors)}

idx1 = np.array([index_map[s] for s in survivors1])
idx2 = np.array([index_map[s] for s in survivors2])

# -----------------------------
# Color palettes for visualization
# -----------------------------
colors1 = [mcolors.to_hex(c) for c in
           [plt.cm.Reds(v) for v in np.linspace(0.40, 0.90, len(idx1))]]

colors2 = [mcolors.to_hex(c) for c in
           [plt.cm.GnBu(v) for v in np.linspace(0.40, 0.90, len(idx2))]]

# Assign a color to each species
color_map = {}
for k, i in enumerate(idx1):
    color_map[i] = colors1[k % len(colors1)]
for k, i in enumerate(idx2):
    color_map[i] = colors2[k % len(colors2)]

# -----------------------------
# Experiment: simulate perturbation dynamics
#
# Steps:
# 1. Run system to quasi-equilibrium
# 2. Apply pulse perturbation to selected species
# 3. Continue simulation to observe recovery or collapse
# -----------------------------
def run_experiment(x0_full, perturb_indices, perturbation_size,
                   T_pre, T_post, A_star, EPS):

    # Pre-perturbation dynamics
    t_pre, y_pre = run(x0_full, 0, T_pre, A_star, EPS)

    # Apply pulse perturbation (multiplicative reduction)
    x_perturbed = y_pre[:, -1].copy()
    x_perturbed[perturb_indices] *= perturbation_size

    # Post-perturbation dynamics
    t_post_raw, y_post = run(x_perturbed, 0, T_POST, A_star, EPS)
    t_post             = t_post_raw + T_PRE  # ensure continuous time

    return t_pre, y_pre, t_post, y_post

# -----------------------------
# Initial conditions
#
# Construct biased initial states:
# - One community dominant
# - The other present at low abundance
# -----------------------------
def make_x0(dominant_idx, weak_idx, N):

    x0 = np.zeros(N)

    x0[dominant_idx] = np.random.uniform(0.8, 1.2, len(dominant_idx))
    x0[weak_idx]     = np.random.uniform(0.1, 0.3, len(weak_idx))

    return x0

np.random.seed(42)

x0_plot1 = make_x0(idx1, idx2, N)   # State 1 dominant
x0_plot2 = make_x0(idx2, idx1, N)   # State 2 dominant

# -----------------------------
# Run two complementary experiments
#
# Experiment 1:
# - State 1 dominates initially
# - Perturb State 1 → test collapse or recovery
#
# Experiment 2:
# - State 2 dominates initially
# - Perturb State 2 → symmetric test
# -----------------------------
print("\nRunning Plot 1 (State 1 dominant, perturb State 1)…")
t1_pre, y1_pre, t1_post, y1_post = run_experiment(
    x0_plot1, idx1, PERTURBATION, T_PRE, T_POST, A_star, EPS)

print("Running Plot 2 (State 2 dominant, perturb State 2)…")
t2_pre, y2_pre, t2_post, y2_post = run_experiment(
    x0_plot2, idx2, PERTURBATION, T_PRE, T_POST, A_star, EPS)

# -----------------------------
# Plotting: time series of all species
#
# Shows:
# - Coexistence within each state
# - Competition between states
# - Response to perturbation (tipping vs recovery)
# -----------------------------
def plot_experiment(ax, t_pre, y_pre, t_post, y_post,
                    idx_dominant, idx_weak, color_map,
                    dominant_label, weak_label):

    # Pre-perturbation trajectories
    for i in idx_dominant:
        ax.plot(t_pre, y_pre[i], color=color_map[i], lw=1.1, alpha=0.85)
    for i in idx_weak:
        ax.plot(t_pre, y_pre[i], color=color_map[i], lw=1.1, alpha=0.85)

    # Post-perturbation trajectories
    for i in idx_dominant:
        ax.plot(t_post, y_post[i], color=color_map[i], lw=1.1, alpha=0.85)
    for i in idx_weak:
        ax.plot(t_post, y_post[i], color=color_map[i], lw=1.1, alpha=0.85)

    # Mark perturbation time
    ax.axvline(T_PRE, color='lightgray', lw=1.4, ls='--')

    ax.set_xlim(0, T_PRE + T_POST)
    ax.set_ylim(bottom=0)

# -----------------------------
# Generate figure
# -----------------------------
fig, axes = plt.subplots(2, 1, figsize=(5, 5), sharey=False)

plot_experiment(
    axes[0], t1_pre, y1_pre, t1_post, y1_post,
    idx_dominant=idx1, idx_weak=idx2,
    color_map=color_map,
    dominant_label="State 1", weak_label="State 2"
)

plot_experiment(
    axes[1], t2_pre, y2_pre, t2_post, y2_post,
    idx_dominant=idx2, idx_weak=idx1,
    color_map=color_map,
    dominant_label="State 2", weak_label="State 1"
)

# Global title summarizing system structure
plt.suptitle(
    f"Bistability temporal dynamics  |  S={S}, μ={muA}, σ={stdA}\n"
    f"State 1: {len(survivors1)} sp  |  State 2: {len(survivors2)} sp  |  "
    f"Overlap: {len(set(survivors1)&set(survivors2))} sp",
    fontsize=11, y=1.02
)

# Clean axis formatting
axes[0].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
axes[1].tick_params(axis='x', which='both', bottom=True, labelbottom=True)

plt.tight_layout()



plt.show()
