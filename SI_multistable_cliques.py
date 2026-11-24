r"""
Internal multistability in assembled cliques of the generalized Lotka-Volterra (GLV) model.

Supporting Information, Section S2, of "Multistability in species-rich communities"
G. Aguade-Gorgorio & S. Kefi.  https://github.com/GuimAguade/MultistabilitySynthesis
November 2026

Model   dN_i/dt = N_i (1 + sum_j A_ij N_j),  A_ii = -1,  A_ij ~ Normal(mu, sigma^2) for i != j,
        i.e. Eq. (2) of the main text with r_i = K_i = 1 and no immigration (closed model).

Protocol
  1. Assembly. For each pool, (mu, sigma) are drawn uniformly in MU_RANGE x SIGMA_RANGE and the
     S species are integrated from random initial abundances, without extinction threshold. The run
     is kept if it ends within 0.1% of a clique C: a feasible, linearly stable equilibrium that no
     absent species of the pool can invade.
  2. Random pulses. From then on only the species of C are considered. Each clique receives N_PULSES
     pulses: a random number of randomly chosen species are each multiplied by a factor drawn
     log-uniformly in [PULSE_FLOOR, 1], so that no species is removed. Each pulsed state is integrated
     until it returns to C, settles on another stable state T (some species of C go extinct), or
     does not settle.
  3. Analytical search. A pulse can only end on a support T within C with N*_T = -A_TT^-1 1 > 0,
     diag(N*_T) A_TT stable, and negative invasion growth rates 1 + (A N*_T)_i < 0 for every species
     i in C \ T. All such T are listed: every support is tested if |C| <= N_EXHAUST, and a
     mixed-integer enumeration is used otherwise. Cliques with a diagonal Lyapunov function
     (D A + A^T D negative definite for some positive diagonal D) have a globally stable equilibrium,
     hence no alternative state, and are not pulsed.
  4. Lost species. For each alternative state T, the lost species are reintroduced at a fraction f
     of their abundance in C, to test whether and when C is recovered.
  5. Robustness of C. Linear stability margin and feasibility margin of multistable cliques, compared
     with monostable cliques of the same size.

Usage   python multistable_cliques.py [N_POOLS]      (Ctrl-C stops the search and prints the results)
Requires numpy and scipy (>= 1.9, for milp).
"""
import sys
import time
from collections import Counter
from itertools import combinations
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, milp, LinearConstraint, Bounds

# ------------------------------------------------------------------ parameters
S = 50                         # species pool size
N_POOLS = 10000              # random pools, one assembly each
MU_RANGE = (-1.5, 0.0)         # pool mean interaction, mu ~ U(MU_RANGE)
SIGMA_RANGE = (0.0, 1.0)       # pool interaction sd, sigma ~ U(SIGMA_RANGE)
SEED = 1
if len(sys.argv) > 1:
    N_POOLS = int(sys.argv[1])

N_PULSES = 1000                # random pulses per clique
PULSE_FLOOR = 1e-3             # no species is reduced below PULSE_FLOOR x its abundance in C
REINTRO = [1e-3, 0.1, 0.5, 1.0]   # reintroduction of lost species, as a fraction of their abundance in C

TOL = 1e-9                     # eigenvalues and invasion rates must be < -TOL
CONV = 1e-3                    # converged: max |N - N*| < CONV x max N*
PROPOSALS = [1e-10, 1e-8, 1e-6, 1e-4, 1e-3]   # thresholds used only to propose a support
T_ASSEMBLY, N_CHUNKS = 500.0, 10              # assembly integrated up to t = 5000
T_CHECK = 20.0                 # pulses: outcomes checked every T_CHECK time units
T_SCREEN_MIN, T_SCREEN_MAX = 2000.0, 10000.0
SETTLE = 1e-2                  # a state is tested when all present species have |growth rate| < SETTLE
SLOW = 0.05                    # after the screen, within 5% of C and approaching: counted as a return
N_FOLLOW = 50                  # unsettled pulses per clique followed with an adaptive solver
T_FOLLOW, N_FOLLOW_CHUNKS = 1000.0, 20
WALL = 60.0                    # wall-clock limit (s) per adaptive integration
N_EXHAUST = 14                 # cliques up to this size: all 2^n supports tested
MILP_EPS, MILP_MX, MILP_TIME, MILP_MAX_SOL = 1e-6, 10.0, 20.0, 200

CERT_CHECK_FRAC = 0.02         # share of certified cliques pulsed anyway, as a check (none should shift)

rng = np.random.default_rng()   # single stream, consumed in the same order as in earlier versions


# ------------------------------------------------------------------ dynamics
class Timeout(Exception):
    pass


def integrate(A, y0, t_end, method="RK45", rtol=1e-8, atol=1e-10):
    """GLV in log-abundances y = log N: dy/dt = 1 + A exp(y). Species can decline towards zero but
    are never removed."""
    t0 = time.time()

    def f(t, y):
        if time.time() - t0 > WALL:
            raise Timeout
        return 1.0 + A @ np.exp(y)

    def blowup(t, y):
        return np.max(y) - np.log(1e6)
    blowup.terminal = True
    try:
        with np.errstate(over="ignore", under="ignore"):
            sol = solve_ivp(f, (0, t_end), y0, method=method, rtol=rtol, atol=atol, events=blowup)
    except Timeout:
        return None, "timeout"
    if sol.status == 1:
        return sol, "blowup"
    return sol, ("ok" if sol.success else "failed")


def equilibrium(A, idx, invaders):
    """Equilibrium on the support idx. Returns (N*, ok), with ok = feasible, linearly stable and
    not invadable by the species in `invaders`."""
    idx, invaders = np.asarray(idx, int), np.asarray(invaders, int)
    if len(idx) == 0:
        return None, False
    try:
        N = np.linalg.solve(A[np.ix_(idx, idx)], -np.ones(len(idx)))
    except np.linalg.LinAlgError:
        return None, False
    if N.min() <= 0:
        return N, False
    stable = np.linalg.eigvals(N[:, None] * A[np.ix_(idx, idx)]).real.max() < -TOL
    uninvadable = len(invaders) == 0 or (1 + A[np.ix_(invaders, idx)] @ N).max() < -TOL
    return N, stable and uninvadable


def identify(A, N):
    """Support (tuple) of a feasible, stable and uninvadable equilibrium lying within CONV of the
    state N, or None. Thresholds only propose supports; acceptance is algebraic."""
    if not np.all(np.isfinite(N)):
        return None
    everyone = np.arange(len(N))
    for th in PROPOSALS:
        sup = np.where(N > th * N.max())[0]
        x, ok = equilibrium(A, sup, np.setdiff1d(everyone, sup))
        if ok:
            full = np.zeros(len(N))
            full[sup] = x
            if np.max(np.abs(N - full)) < CONV * full.max():
                return tuple(int(i) for i in sup)
    return None


def assemble(A, N0):
    """Integrate the whole pool; return the clique reached (array of indices) or a failure flag."""
    y = np.log(N0)
    for _ in range(N_CHUNKS):
        sol, flag = integrate(A, y, T_ASSEMBLY, rtol=1e-6, atol=1e-8)
        if flag != "ok":
            return flag
        y = sol.y[:, -1]
        sup = identify(A, np.exp(y))
        if sup is not None:
            return np.array(sup)
    return "not settled"


def settle(Ac, y):
    """Adaptive (LSODA) integration of the clique system from log-abundances y until it settles.
    Returns the support of the state reached, or a failure flag."""
    for _ in range(N_FOLLOW_CHUNKS):
        sol, flag = integrate(Ac, y, T_FOLLOW, "LSODA")
        if flag != "ok":
            return flag
        y = sol.y[:, -1]
        if not np.all(np.isfinite(y)):
            return "failed"
        sup = identify(Ac, np.exp(y))
        if sup is not None:
            return sup
    return "not settled"


def random_pulses(n, m):
    """m pulses (rows of multiplicative factors): k ~ U{1..n} randomly chosen species are each
    multiplied by a factor log-uniform in [PULSE_FLOOR, 1]; the other species are untouched."""
    k = rng.integers(1, n + 1, m)
    rank = rng.random((m, n)).argsort(axis=1).argsort(axis=1)
    fac = 10 ** rng.uniform(np.log10(PULSE_FLOOR), 0, (m, n))
    return np.where(rank < k[:, None], fac, 1.0)


def run_pulses(Ac, NC, F):
    """Fate of the pulsed states F * N*_C. All pulses are integrated together with a fixed-step RK4
    scheme in log-abundances and checked every T_CHECK. Pulses where the fixed step diverges are
    redone adaptively. At the end of the screen, pulses within SLOW of C and still approaching it are
    counted as returns, and the remaining ones are followed adaptively (at most N_FOLLOW).
    Returns, for each pulse, the support reached (tuple) or a flag (str), and the number of slow returns."""
    m, n = F.shape
    full = tuple(range(n))
    ev = np.linalg.eigvals(NC[:, None] * Ac)
    steps = int(np.ceil(T_CHECK / min(0.2, 0.3 / np.abs(ev).max())))
    dt = T_CHECK / steps
    t_screen = np.clip(20.0 / -ev.real.max(), T_SCREEN_MIN, T_SCREEN_MAX)
    ymax = np.log(1e6 * max(1.0, NC.max()))

    def f(Y):
        return 1.0 + np.exp(Y) @ Ac.T

    Y = np.log(F * NC)
    out = [None] * m
    active = np.arange(m)
    dist = np.full(m, np.inf)              # relative distance to C at the previous check
    approaching = np.zeros(m, bool)
    redo, t, n_slow = [], 0.0, 0
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        while len(active) and t < t_screen:
            Ya = Y[active]
            for _ in range(steps):
                k1 = f(Ya)
                k2 = f(Ya + 0.5 * dt * k1)
                k3 = f(Ya + 0.5 * dt * k2)
                k4 = f(Ya + dt * k3)
                Ya = Ya + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            Y[active] = Ya
            t += T_CHECK
            N = np.exp(Ya)
            blown = ~np.all(np.isfinite(Ya), axis=1)
            blown[~blown] = Ya[~blown].max(axis=1) > ymax
            d = np.full(len(active), np.inf)
            d[~blown] = np.max(np.abs(N[~blown] - NC), axis=1) / NC.max()
            back = d < CONV
            redo += list(active[blown])
            for j in np.where(back)[0]:
                out[active[j]] = full
            done = blown | back
            # pulses that have stopped moving: do they sit on another stable state?
            G = np.where(N > 1e-6 * N.max(axis=1, keepdims=True), np.abs(1 + N @ Ac.T), 0)
            for j in np.where(~done & (G.max(axis=1) < SETTLE))[0]:
                sup = identify(Ac, N[j])
                if sup is not None:
                    out[active[j]] = sup
                    done[j] = True
            approaching = (d < dist[active])[~done]
            dist[active] = d
            active = active[~done]
    if len(active):
        slow = (dist[active] < SLOW) & approaching
        for i in active[slow]:
            out[i] = full
        n_slow = int(slow.sum())
        active = active[~slow]
    follow = active if len(active) <= N_FOLLOW else rng.choice(active, N_FOLLOW, replace=False)
    for i in redo:
        out[i] = settle(Ac, np.log(F[i] * NC))
    for i in follow:
        out[i] = settle(Ac, Y[i])
    for i in active:
        if out[i] is None:
            out[i] = "unresolved"
    return out, n_slow


# ------------------------------------------------------------------ algebra
def lyapunov_certificate(Ac):
    """True if D Ac + Ac^T D is negative definite for some positive diagonal D (D = I tried first).
    The feasible equilibrium of the clique is then globally stable."""
    if np.linalg.eigvalsh(Ac + Ac.T).max() < -TOL:
        return True

    def worst(z):
        M = np.exp(z - z.mean())[:, None] * Ac
        return np.linalg.eigvalsh(M + M.T).max()
    res = minimize(worst, np.zeros(len(Ac)), method="Nelder-Mead",
                   options=dict(maxiter=400 * len(Ac), xatol=1e-8, fatol=1e-12))
    return res.fun < -TOL


def alternatives_exhaustive(Ac):
    """All alternative stable states T (proper supports of C), testing every support."""
    n = len(Ac)
    alts = []
    for k in range(1, n):
        idx = np.array(list(combinations(range(n), k)))
        M = Ac[idx[:, :, None], idx[:, None, :]]
        N = np.linalg.solve(M, -np.ones((len(idx), k, 1)))[..., 0]
        for j in np.where(np.all(N > 0, axis=1))[0]:
            lost = np.setdiff1d(np.arange(n), idx[j])
            if (1 + Ac[np.ix_(lost, idx[j])] @ N[j]).max() < -TOL and \
                    np.linalg.eigvals(N[j][:, None] * M[j]).real.max() < -TOL:
                alts.append(tuple(int(i) for i in idx[j]))
    return alts, True


def alternatives_milp(Ac, NC):
    """All alternative stable states T by mixed-integer enumeration. Variables: abundances x and
    presence z. z_i = 1: 1 + (A x)_i = 0, x_i in [MILP_EPS, Mx]; z_i = 0: 1 + (A x)_i <= -MILP_EPS,
    x_i = 0; sum z <= n - 1 excludes C. Each solution is verified exactly and then excluded, until
    no solution remains (complete) or MILP_MAX_SOL solutions have been found (incomplete)."""
    n = len(Ac)
    I = np.eye(n)
    Mx = MILP_MX * max(1.0, NC.max())
    Mw = 1 + np.abs(Ac).sum(axis=1) * Mx
    rows = [np.hstack([Ac, -MILP_EPS * I]), np.hstack([Ac, -np.diag(Mw)]),
            np.hstack([I, -Mx * I]), np.hstack([I, -MILP_EPS * I]),
            np.r_[np.zeros(n), np.ones(n)][None]]
    inf = np.full(n, np.inf)
    lb = [-inf, -1 - Mw, -inf, np.zeros(n), np.array([-np.inf])]
    ub = [np.full(n, -1 - MILP_EPS), inf, np.zeros(n), inf, np.array([n - 1.0])]
    bounds = Bounds(np.zeros(2 * n), np.r_[np.full(n, Mx), np.ones(n)])
    integrality = np.r_[np.zeros(n), np.ones(n)]
    alts, complete = [], False
    for _ in range(MILP_MAX_SOL):
        res = milp(np.zeros(2 * n), integrality=integrality, bounds=bounds,
                   constraints=LinearConstraint(np.vstack(rows), np.concatenate(lb), np.concatenate(ub)),
                   options=dict(time_limit=MILP_TIME, disp=False))
        if res.status == 2:            # infeasible: every saturated support has been listed
            complete = True
            break
        if res.status != 0 or res.x is None:
            break
        z = res.x[n:] > 0.5
        sup = np.where(z)[0]
        _, ok = equilibrium(Ac, sup, np.where(~z)[0])
        if ok:
            alts.append(tuple(int(i) for i in sup))
        rows.append(np.r_[np.zeros(n), np.where(z, 1.0, -1.0)][None])   # exclude this support
        lb.append(np.array([-np.inf]))
        ub.append(np.array([len(sup) - 1.0]))
    return alts, complete


def alternative_states(Ac, NC, certified):
    """Analytical list of alternative stable states of the clique, and whether the list is complete."""
    if certified:
        return [], True
    if len(Ac) <= N_EXHAUST:
        return alternatives_exhaustive(Ac)
    return alternatives_milp(Ac, NC)


def margins(Ac, NC):
    """Linear stability margin, -max Re eig(diag(N*) A), and feasibility margin: the smallest relative
    change of the growth-rate vector r = (1, ..., 1) that drives a species of C to zero."""
    stab = -np.linalg.eigvals(NC[:, None] * Ac).real.max()
    feas = np.min(NC / np.linalg.norm(np.linalg.inv(Ac), axis=1)) / np.sqrt(len(NC))
    return stab, feas


# ------------------------------------------------------------------ search
print(f"GLV pools: S = {S}, A_ii = -1, A_ij ~ N(mu, sigma^2), mu ~ U{MU_RANGE}, sigma ~ U{SIGMA_RANGE}, "
      f"{N_POOLS} pools, seed {SEED}")
pools, cliques = Counter(), []
t_start = time.time()
try:
    for p in range(N_POOLS):
        mu, sigma = rng.uniform(*MU_RANGE), rng.uniform(*SIGMA_RANGE)
        A = rng.normal(mu, sigma, (S, S))
        np.fill_diagonal(A, -1.0)
        res = assemble(A, rng.uniform(0.01, 1.0, S))
        if isinstance(res, str):
            pools[res] += 1
            continue
        pools["clique"] += 1
        n = len(res)
        Ac = A[np.ix_(res, res)]
        NC = np.linalg.solve(Ac, -np.ones(n))
        c = dict(n=n, mu=mu, sigma=sigma, Ac=Ac, NC=NC, alts_num=[], n_unsettled=0, n_slow=0)
        c["cert"] = n == 1 or lyapunov_certificate(Ac)
        c["pulsed"] = n > 1 and ((not c["cert"]) or rng.random() < CERT_CHECK_FRAC)
        c["stab"], c["feas"] = margins(Ac, NC)
        if c["pulsed"]:
            F = random_pulses(n, N_PULSES)
            outs, c["n_slow"] = run_pulses(Ac, NC, F)
            switch = np.array([isinstance(o, tuple) and len(o) < n for o in outs])
            c["n_unsettled"] = sum(isinstance(o, str) for o in outs)
            c["frac_switch"] = switch.mean()
            if switch.any():
                D = ((1 - F) * NC).sum(axis=1) / NC.sum()     # fraction of total abundance removed
                i = np.where(switch)[0][np.argmin(D[switch])]  # mildest switching pulse
                c["D_min"], c["depth_min"] = D[i], F[i].min()  # depth: lowest remaining N0_i / N*_i
                c["depth_all"] = F[switch].min(axis=1)         # depth of every switching pulse
                c["alts_num"] = sorted({o for o, s in zip(outs, switch) if s})
        c["alts_an"], c["complete"] = alternative_states(Ac, NC, c["cert"])
        cliques.append(c)
        if (p + 1) % 100 == 0:
            el = time.time() - t_start
            print(f"  {p + 1}/{N_POOLS} pools, {len(cliques)} cliques, "
                  f"{sum(bool(c['alts_num']) for c in cliques)} internally multistable | "
                  f"{el / 60:.1f} min elapsed, {el / (p + 1) * (N_POOLS - p - 1) / 60:.1f} min left",
                  flush=True)
except KeyboardInterrupt:
    print(f"\n*** interrupted after {sum(pools.values())} pools")


# ------------------------------------------------------------------ results
def pct(k, n):
    return f"{k}/{n} ({100 * k / max(n, 1):.1f}%)"


def med_iqr(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if not len(v):
        return "NA"
    q = np.percentile(v, [25, 50, 75])
    return f"{q[1]:.3g} [IQR {q[0]:.3g}-{q[2]:.3g}]"


def label_num(c):
    if c["alts_num"]:
        return "multistable"
    return "not settled" if c["n_unsettled"] else "monostable"


def label_an(c):
    if c["alts_an"]:
        return "multistable"
    return "monostable" if c["complete"] else "undetermined"


multi = [c for c in cliques if c["alts_num"]]
mono = [c for c in cliques if label_num(c) == "monostable" and c["n"] >= 2]

print("\n=== 1. Assembly ===")
print(f"  pools: {sum(pools.values())}")
print(f"  assembly reached a clique: {pct(pools['clique'], sum(pools.values()))}")
for k, v in pools.items():
    if k != "clique":
        print(f"  outcome '{k}': {v}")
print(f"  clique size: {med_iqr([c['n'] for c in cliques])}")
print(f"  cliques with a diagonal Lyapunov function (globally stable, no alternative state): "
      f"{pct(sum(c['cert'] for c in cliques), len(cliques))}")
print(f"  cliques pulsed ({N_PULSES} random pulses each): {sum(c['pulsed'] for c in cliques)} "
      f"(of which {sum(c['pulsed'] and c['cert'] for c in cliques)} certified, as a check; "
      f"certified cliques that shifted: {sum(c['cert'] and bool(c['alts_num']) for c in cliques)}, must be 0)")

print("\n=== 2. How many cliques are internally multistable? ===")
print(f"  random pulses: cliques where at least one pulse ends on another stable state: "
      f"{pct(len(multi), len(cliques))}")
print(f"  random pulses: cliques with pulses that did not settle (no alternative state found): "
      f"{sum(label_num(c) == 'not settled' for c in cliques)} "
      f"({sum(c['n_unsettled'] for c in cliques)} pulses; {sum(c['n_slow'] for c in cliques)} "
      f"further pulses counted as slow returns to C)")
print(f"  analytical: cliques with at least one alternative stable state: "
      f"{pct(sum(label_an(c) == 'multistable' for c in cliques), len(cliques))}; "
      f"undetermined: {sum(label_an(c) == 'undetermined' for c in cliques)}")
labs_an = ["multistable", "monostable", "undetermined"]
print("  " + "pulses / analytical".ljust(24) + "".join(f"{l:>14s}" for l in labs_an))
for ln in ["multistable", "monostable", "not settled"]:
    print(f"  {ln:24s}" + "".join(
        f"{sum(label_num(c) == ln and label_an(c) == la for c in cliques):14d}" for la in labs_an))
n_states = sum(len(c["alts_num"]) for c in multi)
n_in = sum(T in c["alts_an"] for c in multi for T in c["alts_num"])
print(f"  states reached by pulses that are in the analytical list: {n_in}/{n_states}")
print(f"  size of multistable cliques |C|: {med_iqr([c['n'] for c in multi])} "
      f"(all cliques: {med_iqr([c['n'] for c in cliques])})")
lost = [c["n"] - len(T) for c in multi for T in c["alts_num"]]
print(f"  alternative states per multistable clique: {med_iqr([len(c['alts_num']) for c in multi])}")
print(f"  size of alternative states |T|: {med_iqr([len(T) for c in multi for T in c['alts_num']])}")
if lost:
    print(f"  species lost when shifting, |C| - |T|: {med_iqr(lost)}, range {min(lost)}-{max(lost)}")

print("\n=== 3. How large must a pulse be? (mildest switching pulse of each multistable clique) ===")
print(f"  fraction of total abundance removed: {med_iqr([c['D_min'] for c in multi])}")
print(f"  lowest remaining abundance of any species, min N0_i/N*_i: "
      f"{med_iqr([c['depth_min'] for c in multi])}")
if multi:
    print(f"  (same quantity over all switching pulses: "
          f"{med_iqr(np.concatenate([c['depth_all'] for c in multi]))})")
print(f"  share of the {N_PULSES} random pulses that switch: {med_iqr([c['frac_switch'] for c in multi])}")

print("\n=== 4. Do the lost species come back? ===")
pairs = []
for c in multi:
    full = tuple(range(c["n"]))
    for T in c["alts_num"]:
        T_, R = np.array(T), np.setdiff1d(np.arange(c["n"]), T)
        NT = np.linalg.solve(c["Ac"][np.ix_(T_, T_)], -np.ones(len(T_)))
        out = {}
        for f in REINTRO:
            x = np.zeros(c["n"])
            x[T_], x[R] = NT, f * c["NC"][R]
            o = settle(c["Ac"], np.log(x))
            out[f] = "C" if o == full else ("T" if o == T else "other")
        pairs.append(dict(inv=(1 + c["Ac"][np.ix_(R, T_)] @ NT).max(), out=out))
print(f"  invasion growth rate of the lost species at T (largest per state): "
      f"{med_iqr([q['inv'] for q in pairs])}; negative in {sum(q['inv'] < 0 for q in pairs)}/{len(pairs)}")
print(f"  lost species reintroduced at a fraction f of their abundance in C:")
for f in REINTRO:
    cnt = Counter(q["out"][f] for q in pairs)
    print(f"    f = {f:<6g} back to C: {cnt['C']}/{len(pairs)}, stays at T: {cnt['T']}, other: {cnt['other']}")

print("\n=== 5. Are multistable cliques less robust than monostable ones? ===")
print(f"  monostable reference: {len(mono)} cliques with n >= 2 (certified, or all pulses returned)")
for key, lab in [("feas", "feasibility margin"), ("stab", "linear stability margin")]:
    p = [100 * np.mean([r[key] < c[key] for r in mono if r["n"] == c["n"]]) for c in multi
         if sum(r["n"] == c["n"] for r in mono) >= 10]
    print(f"  {lab}: multistable {med_iqr([c[key] for c in multi])}, "
          f"monostable {med_iqr([c[key] for c in mono])}")
    print(f"    percentile among monostable cliques of the same size: median {np.median(p) if p else np.nan:.0f} "
          f"({len(p)} multistable cliques compared; 50 = typical)")
