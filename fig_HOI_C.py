import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

# ======================================================
# Monte Carlo comparison: GLV vs HOI dynamics
#
# Goal:
# Explore how feasibility, stability, and multistability
# depend on interaction statistics (μ, σ)
#
# Models:
# 1. GLV (pairwise interactions)
# 2. HOI (higher-order interactions)
#
# For each (μ, σ):
# - Sample random interaction structures
# - Compute equilibria
# - Test feasibility (N_i > 0)
# - Test stability (Jacobian eigenvalues)
# - Detect multistability (coexistence + boundary stable states)
# ======================================================

# -----------------------------
# System size and parameter grid
# -----------------------------
S = 3
n_grid = 300

mu_vals    = np.linspace(-3, 0.5, n_grid)
sigma_vals = np.linspace(0.0, 1.5, n_grid)

# -----------------------------
# Storage arrays
# -----------------------------
glv_feasible = np.zeros((n_grid, n_grid))
hoi_feasible = np.zeros((n_grid, n_grid))

glv_stable = np.zeros((n_grid, n_grid))
hoi_stable = np.zeros((n_grid, n_grid))

glv_1feasible = np.zeros((n_grid, n_grid))  # boundary (N1=0)
hoi_1feasible = np.zeros((n_grid, n_grid))

glv_1stable = np.zeros((n_grid, n_grid))
hoi_1stable = np.zeros((n_grid, n_grid))

glv_multistable = np.zeros((n_grid, n_grid))
hoi_multistable = np.zeros((n_grid, n_grid))

# -----------------------------
# Monte Carlo parameters
# -----------------------------
n_trials = 500

# ======================================================
# Parameter scan
# ======================================================
for i, mu in enumerate(mu_vals):
    for j, sigma in enumerate(sigma_vals):

        # Counters
        glv_feas_count = hoi_feas_count = 0
        glv_stab_count = hoi_stab_count = 0
        glv_1feas_count = hoi_1feas_count = 0
        glv_1stab_count = hoi_1stab_count = 0
        glv_multistab = hoi_multistab = 0

        # -------------------------
        # Monte Carlo sampling
        # -------------------------
        for trial in range(n_trials):

            glv_temp = 0  # flag: stable interior equilibrium
            hoi_temp = 0

            # ======================================================
            # 1. GLV MODEL
            # ======================================================

            # Random symmetric interaction matrix
            A = np.zeros((S, S))
            for a in range(S):
                for b in range(S):
                    if a != b:
                        A[a, b] = np.random.normal(mu, sigma)

            # Impose symmetry (optional structural constraint)
            for a in range(S):
                for b in range(S):
                    if a < b:
                        A[a, b] = A[b, a]

            # Solve equilibrium: (I - A) N = 1
            M   = np.eye(S) - A
            rhs = np.ones(S)

            try:
                N_glv = np.linalg.solve(M, rhs)

                # ---- Feasibility ----
                if np.all(N_glv > 0):
                    glv_feas_count += 1

                    # ---- Jacobian ----
                    J = np.zeros((S, S))
                    for a in range(S):
                        for b in range(S):
                            if a == b:
                                J[a, b] = 1 - 2*N_glv[a] + np.dot(A[a], N_glv)
                            else:
                                J[a, b] = N_glv[a] * A[a, b]

                    eigvals = np.linalg.eigvals(J)

                    # ---- Stability ----
                    if np.all(np.real(eigvals) < 0):
                        glv_stab_count += 1
                        glv_temp = 1

            except np.linalg.LinAlgError:
                pass

            # -----------------------------
            # GLV boundary equilibrium (N1 = 0)
            # -----------------------------
            M_red = np.array([
                [1, -A[1,2]],
                [-A[2,1], 1]
            ])
            rhs_red = np.ones(2)

            try:
                N2_red, N3_red = np.linalg.solve(M_red, rhs_red)

                if (N2_red > 0) and (N3_red > 0):
                    glv_1feas_count += 1

                    N_boundary = np.array([0, N2_red, N3_red])

                    # Jacobian at boundary
                    J = np.zeros((S, S))
                    for a in range(S):
                        for b in range(S):
                            if a == b:
                                J[a, b] = 1 - 2*N_boundary[a] + np.dot(A[a], N_boundary)
                            else:
                                J[a, b] = N_boundary[a] * A[a, b]

                    eigvals = np.linalg.eigvals(J)

                    if np.all(np.real(eigvals) < 0):
                        glv_1stab_count += 1

                        # Detect multistability
                        if glv_temp == 1:
                            glv_multistab += 1

            except np.linalg.LinAlgError:
                pass

            # ======================================================
            # 2. HOI MODEL
            # ======================================================

            # Random HOI coefficients
            B = np.random.normal(mu, sigma, size=S)

            # -----------------------------
            # Boundary equilibrium (N1 = 0)
            # -----------------------------
            N1, N2, N3 = 0, 1, 1  # analytical simplification

            hoi_1feas_count += 1

            J = np.zeros((3,3))

            # Diagonal
            J[0,0] = 1 - 2*N1 + B[0]*N2*N3
            J[1,1] = 1 - 2*N2 + B[1]*N1*N3
            J[2,2] = 1 - 2*N3 + B[2]*N1*N2

            # Off-diagonal
            J[0,1] = N1 * B[0] * N3
            J[0,2] = N1 * B[0] * N2
            J[1,0] = N2 * B[1] * N3
            J[1,2] = N2 * B[1] * N1
            J[2,0] = N3 * B[2] * N2
            J[2,1] = N3 * B[2] * N1

            eigvals = np.linalg.eigvals(J)

            if np.all(np.real(eigvals) < 0):
                hoi_1stab_count += 1
                hoi_temp = 1

            # -----------------------------
            # Interior equilibrium (numerical)
            # -----------------------------
            def hoi_equilibrium(N):
                N1, N2, N3 = N
                return np.array([
                    1 - N1 + B[0]*N2*N3,
                    1 - N2 + B[1]*N1*N3,
                    1 - N3 + B[2]*N1*N2
                ])

            sol = root(hoi_equilibrium, np.ones(3))

            if sol.success:
                N_hoi = sol.x

                if np.all(N_hoi > 0):
                    hoi_feas_count += 1

                    N1, N2, N3 = N_hoi

                    # Jacobian
                    J = np.zeros((3,3))
                    J[0,0] = 1 - 2*N1 + B[0]*N2*N3
                    J[1,1] = 1 - 2*N2 + B[1]*N1*N3
                    J[2,2] = 1 - 2*N3 + B[2]*N1*N2

                    J[0,1] = N1 * B[0] * N3
                    J[0,2] = N1 * B[0] * N2
                    J[1,0] = N2 * B[1] * N3
                    J[1,2] = N2 * B[1] * N1
                    J[2,0] = N3 * B[2] * N2
                    J[2,1] = N3 * B[2] * N1

                    eigvals = np.linalg.eigvals(J)

                    if np.all(np.real(eigvals) < 0):
                        hoi_stab_count += 1

                        if hoi_temp == 1:
                            hoi_multistab += 1

        # ======================================================
        # Store probabilities
        # ======================================================
        glv_feasible[i, j] = glv_feas_count / n_trials
        hoi_feasible[i, j] = hoi_feas_count / n_trials

        glv_1feasible[i, j] = glv_1feas_count / n_trials
        hoi_1feasible[i, j] = hoi_1feas_count / n_trials

        glv_stable[i, j] = glv_stab_count / glv_feas_count if glv_feas_count > 0 else 0
        hoi_stable[i, j] = hoi_stab_count / hoi_feas_count if hoi_feas_count > 0 else 0

        glv_1stable[i, j] = glv_1stab_count / glv_1feas_count if glv_1feas_count > 0 else 0
        hoi_1stable[i, j] = hoi_1stab_count / hoi_1feas_count if hoi_1feas_count > 0 else 0

        glv_multistable[i, j] = glv_multistab / n_trials
        hoi_multistable[i, j] = hoi_multistab / n_trials

print("Monte Carlo scan complete.")

# ======================================================
# Visualization: multistability maps
# ======================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

extent = [mu_vals[0], mu_vals[-1], sigma_vals[0], sigma_vals[-1]]

# GLV
im0 = axes[0].imshow(glv_multistable.T, origin='lower',
                     aspect='auto', extent=extent,
                     cmap='Greens', vmin=0, vmax=1)

axes[0].set_title("GLV Multistability")
axes[0].set_xlabel("μ")
axes[0].set_ylabel("σ")

# HOI
im1 = axes[1].imshow(hoi_multistable.T, origin='lower',
                     aspect='auto', extent=extent,
                     cmap='Blues', vmin=0, vmax=1)

axes[1].set_title("HOI Multistability")
axes[1].set_xlabel("μ")

fig.colorbar(im0, ax=axes[0])
fig.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.show()
