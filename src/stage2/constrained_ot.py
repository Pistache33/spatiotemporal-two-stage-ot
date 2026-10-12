"""Stage-2 exact-hard unbalanced FGW (constrained by the Stage-1 feasible set).

Extracted verbatim from `pairA&C_stage1&2.ipynb` (cell 3, the "PARETO STAGE2"
cell). Objective, parameters, constraints and the sparse sinkhorn solver are
unchanged. Only `DCR_SCRIPT` has been repointed to the bundled
`domain_concentration_fgw.py` (the original expression-cost builder).

Original absolute paths (ROOT / PAIR_CONFIG / CURATED_PATH) are preserved below.
"""
# ============================================================
# PARETO STAGE2 — ORIGINAL EXACT-HARD UNBALANCED FGW
# Helper cell only; nothing runs automatically.
# ============================================================

import os
GPU_ID = "0"   # edit before first JAX import if needed
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from pathlib import Path
import gc, re, json, math, time, tempfile, importlib.util
import numpy as np
import pandas as pd
import anndata as ad
from IPython.display import display

import jax
import jax.numpy as jnp
from ott.geometry import pointcloud, costs
from ott.problems.quadratic import quadratic_problem
from moscot.backends.ott._utils import alpha_to_fused_penalty

# ============================================================
# CONFIG
# ============================================================

ROOT = Path("/mnt/disk2/lirui/mosta_e105_e115_baselines/pareto")
CURATED_PATH = Path("/mnt/disk2/lirui/baseline_scripts/mosta_flatvi_fgw/MOSTA_curated_transitions.tsv")
DCR_SCRIPT = Path(__file__).with_name("domain_concentration_fgw.py")  # originally /mnt/disk2/lirui/baseline_scripts/run_domain_concentration_fgw.py

PAIR_CONFIG = {
    "PairA": {
        "source": Path("/mnt/disk2/lirui/domain_concentration_repro/input/PairA_source_E9.5_E1S1.h5ad"),
        "target": Path("/mnt/disk2/lirui/domain_concentration_repro/input/PairA_target_E10.5_E2S1.h5ad"),
    },
    "PairC": {
        "source": Path("/mnt/disk2/lirui/mosta_e105_e115_nbvae30/data/E10.5_E2S1.nbvae30.h5ad"),
        "target": Path("/mnt/disk2/lirui/mosta_e105_e115_nbvae30/data/E11.5_E1S1.nbvae30.h5ad"),
    },
}

FEATURE, SPATIAL, ANN = "X_nbvae_30", "spatial", "annotation"

ALPHA2 = 0.4
EPSILON = 1e-4
TAU_A, TAU_B = 1.0, 0.05

OUTER_MAX = 10
INNER_MIN, INNER_MAX = 20, 500
INNER_CHECK_EVERY, INNER_TOL = 10, 0.01

BATCH_SIZE = 1024
COST_CHUNK = 128


# ============================================================
# PATH / SUPPORT HELPERS
# ============================================================

def res_tag(x):
    return f"{float(x):g}".replace(".", "p")

def stage1_dir(pair_name, source_resolution, target_resolution):
    return ROOT / pair_name / f"src{res_tag(source_resolution)}_tgt{res_tag(target_resolution)}"

def resolve_support_path(run_dir):
    candidates = [
        run_dir / "hard_support.npz",
        run_dir / "feasible_set" / "hard_support.npz",
    ]
    for p in candidates:
        if p.exists(): return p
    hits = sorted(list(run_dir.glob("*hard_support*.npz")) + list((run_dir / "feasible_set").glob("*hard_support*.npz")) if (run_dir / "feasible_set").exists() else list(run_dir.glob("*hard_support*.npz")))
    if hits: return hits[0]
    raise FileNotFoundError(f"No hard-support npz found under {run_dir}")

def load_hard_support(path, ns, nt):
    S = np.load(path, allow_pickle=False)
    keys = set(S.files)

    # Current Stage1 format
    if {"source_domain_code", "support_indptr", "support_indices"} <= keys:
        codes = np.asarray(S["source_domain_code"], dtype=np.int32)
        indptr = np.asarray(S["support_indptr"], dtype=np.int64)
        indices = np.asarray(S["support_indices"], dtype=np.int32)

    # Earlier compact Stage1 format
    elif {"source_spot_leiden_code", "support_offsets", "support_target_indices"} <= keys:
        codes = np.asarray(S["source_spot_leiden_code"], dtype=np.int32)
        indptr = np.asarray(S["support_offsets"], dtype=np.int64)
        indices = np.asarray(S["support_target_indices"], dtype=np.int32)

    else:
        raise KeyError(f"Unknown hard-support format. keys={sorted(keys)}")

    n_domains = len(indptr) - 1
    if len(codes) != ns: raise ValueError(f"support source length={len(codes)} != ns={ns}")
    if codes.min() < 0 or codes.max() >= n_domains: raise ValueError("invalid source domain code")
    if len(indices) and (indices.min() < 0 or indices.max() >= nt): raise ValueError("invalid target index")

    sizes = np.diff(indptr)
    if np.any(sizes <= 0): raise RuntimeError("At least one source domain has empty support")

    return {"codes": codes, "indptr": indptr, "indices": indices, "n_domains": n_domains}


# ============================================================
# HARD SUPPORT -> SPARSE SPOT EDGES
# ============================================================

def build_stage2_edges(S, ns, nt):
    codes, indptr, indices = S["codes"], S["indptr"], S["indices"]
    domain_degree = np.diff(indptr)
    row_degree = domain_degree[codes]
    if np.any(row_degree <= 0): raise RuntimeError("empty feasible source row")

    row_offsets = np.zeros(ns + 1, dtype=np.int64)
    row_offsets[1:] = np.cumsum(row_degree)
    n_edges = int(row_offsets[-1])

    src_edge = np.repeat(np.arange(ns, dtype=np.int32), row_degree)
    tgt_edge = np.empty(n_edges, dtype=np.int32)

    for i in range(ns):
        d = int(codes[i])
        lo, hi = row_offsets[i], row_offsets[i + 1]
        tgt_edge[lo:hi] = indices[indptr[d]:indptr[d + 1]]

    return {
        "src_edge_np": src_edge, "tgt_edge_np": tgt_edge,
        "row_degree": row_degree, "row_offsets": row_offsets,
        "n_edges": n_edges, "density": n_edges / float(ns * nt),
    }


# ============================================================
# ORIGINAL NBVAE30 EXPRESSION COST
#
# Cexpr_ij = ||z_i-z_j||_2 / GLOBAL MAX over ALL i,j.
# Only allowed entries are retained afterwards.
# ============================================================

_DCR = None

def load_dcr_builder():
    global _DCR
    if _DCR is not None: return _DCR
    spec = importlib.util.spec_from_file_location("dcr_stage2_exact", DCR_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _DCR = mod
    return mod

def build_expression_edges(pair_name, zs, zt, E):
    dcr = load_dcr_builder()
    parent = "/mnt/disk2/lirui" if os.path.isdir("/mnt/disk2/lirui") else None

    with tempfile.TemporaryDirectory(prefix=f"_tmp_pareto_stage2_{pair_name}_", dir=parent) as tmp:
        c0_path = Path(tmp) / "C0.npy"
        C0, info = dcr.build_c0(c0_path, zs, zt, COST_CHUNK)
        expr = np.empty(E["n_edges"], dtype=np.float32)

        for i in range(zs.shape[0]):
            lo, hi = E["row_offsets"][i], E["row_offsets"][i + 1]
            expr[lo:hi] = C0[i, E["tgt_edge_np"][lo:hi]]
            if i == 0 or (i + 1) % 1000 == 0 or i == zs.shape[0] - 1:
                print(f"expression edges: {i+1}/{zs.shape[0]}")

        del C0
        gc.collect()

    return expr, info


# ============================================================
# ORIGINAL STAGE2 SPATIAL GEOMETRY
# dimension-wise standardization on RAW slice coordinates
# ============================================================

def standardize_spatial(x):
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    return ((x - mu) / sd).astype(np.float32)


# ============================================================
# EXACT SPARSE LOG-SINKHORN
# ============================================================

def make_sparse_solver(src_edge_np, tgt_edge_np, ns, nt):
    src_edge = jnp.asarray(src_edge_np, dtype=jnp.int32)
    tgt_edge = jnp.asarray(tgt_edge_np, dtype=jnp.int32)

    active_target_np = np.bincount(tgt_edge_np, minlength=nt) > 0
    active_target = jnp.asarray(active_target_np)

    loga = jnp.full((ns,), -jnp.log(float(ns)), dtype=jnp.float32)
    logb = jnp.full((nt,), -jnp.log(float(nt)), dtype=jnp.float32)

    def segment_lse(values, ids, n_segments):
        mx = jnp.full((n_segments,), -jnp.inf, dtype=values.dtype).at[ids].max(values)
        shifted = values - mx[ids]
        sm = jnp.zeros((n_segments,), dtype=values.dtype).at[ids].add(jnp.exp(shifted))
        return jnp.where(sm > 0, mx + jnp.log(sm), -jnp.inf)

    @jax.jit
    def sinkhorn_step(logK, logu, logv):
        row_lse = segment_lse(logK + logv[tgt_edge], src_edge, ns)
        new_logu = TAU_A * (loga - row_lse)

        col_lse = segment_lse(logK + new_logu[src_edge], tgt_edge, nt)
        proposed_v = TAU_B * (logb - col_lse)
        new_logv = jnp.where(active_target, proposed_v, 0.0)
        return new_logu, new_logv

    @jax.jit
    def source_error(logK, logu, logv):
        row_lse = segment_lse(logK + logv[tgt_edge], src_edge, ns)
        row_mass = jnp.exp(logu + row_lse)
        expected = 1.0 / ns
        return jnp.max(jnp.abs(row_mass - expected) / expected)

    def solve(edge_cost, epsilon):
        logK = -edge_cost / float(epsilon)
        logu = jnp.zeros(ns, dtype=jnp.float32)
        logv = jnp.zeros(nt, dtype=jnp.float32)
        final_err, used_iter = np.inf, INNER_MAX

        for it in range(1, INNER_MAX + 1):
            logu, logv = sinkhorn_step(logK, logu, logv)
            if it >= INNER_MIN and it % INNER_CHECK_EVERY == 0:
                final_err = float(jax.device_get(source_error(logK, logu, logv)))
                if final_err <= INNER_TOL:
                    used_iter = it
                    break

        edge_mass = jnp.exp(logu[src_edge] + logK + logv[tgt_edge])
        return edge_mass, used_iter, final_err

    return src_edge, tgt_edge, solve


# ============================================================
# OTT WRAPPER FOR GW LINEARIZATION
# ============================================================

class DenseTransport:
    def __init__(self, matrix):
        self.matrix = matrix

    def marginal(self, axis):
        return self.matrix.sum(axis=axis)


# ============================================================
# ORIGINAL STAGE2 CURATED EVALUATION
# self + curated descendants; Cavity IS NOT removed.
# ============================================================

def load_stage2_allowed():
    df = pd.read_csv(CURATED_PATH, sep=None, engine="python")

    def find_col(candidates):
        for c in candidates:
            if c in df.columns: return c
        raise KeyError(f"Cannot find columns {candidates}")

    src_col = find_col(["Cell_type", "cell_type", "source_annotation"])
    desc_col = find_col(["Known_descendants", "known_descendants", "descendants"])

    def split(x):
        x = str(x)
        if x.lower() == "nan": return []
        return [y.strip() for y in re.split(r"[,;|\n]+", x) if y.strip() and y.strip() != "/"]

    allowed = {}
    for _, r in df.iterrows():
        s = str(r[src_col]).strip()
        if not s or s.lower() == "nan": continue
        viable = set(split(r[desc_col]))
        viable.add(s)
        allowed[s] = viable

    return allowed


def evaluate_stage2(pair_name, source, target, E, edge_mass, allowed):
    src_ann = source.obs[ANN].astype(str).to_numpy()
    tgt_ann = target.obs[ANN].astype(str).to_numpy()
    ns = source.n_obs

    top1 = np.empty(ns, dtype=np.int32)
    top1_mass = np.empty(ns, dtype=np.float32)

    for i in range(ns):
        lo, hi = E["row_offsets"][i], E["row_offsets"][i + 1]
        local = edge_mass[lo:hi]
        k = int(np.argmax(local))
        top1[i] = E["tgt_edge_np"][lo + k]
        top1_mass[i] = local[k]

    endpoint_ann = tgt_ann[top1]
    top1_correct = np.asarray([
        endpoint_ann[i] in allowed.get(src_ann[i], {src_ann[i]})
        for i in range(ns)
    ], dtype=bool)

    rows = []
    for s in sorted(np.unique(src_ann)):
        ii = np.flatnonzero(src_ann == s)
        rows.append({
            "source_annotation": s,
            "n_source_cells": len(ii),
            "top1_accuracy": float(top1_correct[ii].mean()),
        })
    per_source = pd.DataFrame(rows)

    src_types = np.asarray(sorted(np.unique(src_ann)), dtype=str)
    tgt_types = np.asarray(sorted(np.unique(tgt_ann)), dtype=str)
    src_map = {x:i for i,x in enumerate(src_types)}
    tgt_map = {x:i for i,x in enumerate(tgt_types)}

    src_code = np.asarray([src_map[x] for x in src_ann], dtype=np.int32)
    tgt_code = np.asarray([tgt_map[x] for x in tgt_ann], dtype=np.int32)

    valid = np.zeros((len(src_types), len(tgt_types)), dtype=bool)
    for s, si in src_map.items():
        for t in allowed.get(s, {s}):
            if t in tgt_map: valid[si, tgt_map[t]] = True

    edge_correct = valid[src_code[E["src_edge_np"]], tgt_code[E["tgt_edge_np"]]]

    total_mass = float(edge_mass.sum(dtype=np.float64))
    correct_mass = float(edge_mass[edge_correct].sum(dtype=np.float64))
    mass_total = np.bincount(src_code[E["src_edge_np"]], weights=edge_mass, minlength=len(src_types))
    mass_correct = np.bincount(src_code[E["src_edge_np"]], weights=edge_mass * edge_correct, minlength=len(src_types))
    mass_acc = np.divide(mass_correct, mass_total, out=np.full(len(src_types), np.nan), where=mass_total > 0)

    per_source = per_source.merge(
        pd.DataFrame({"source_annotation": src_types, "transport_mass_accuracy": mass_acc}),
        on="source_annotation", how="left"
    )

    details = pd.DataFrame({
        "source_index": np.arange(ns),
        "source_obs_name": source.obs_names.astype(str),
        "source_annotation": src_ann,
        "target_index_top1": top1,
        "target_obs_name_top1": target.obs_names.astype(str).to_numpy()[top1],
        "target_annotation_top1": endpoint_ann,
        "top1_mass": top1_mass,
        "top1_correct": top1_correct,
    })

    summary = {
        "pair": pair_name,
        "top1_weighted": float(top1_correct.mean()),
        "top1_macro": float(per_source["top1_accuracy"].mean()),
        "mass_weighted": correct_mass / total_mass,
        "mass_macro": float(np.nanmean(mass_acc)),
        "transport_mass": total_mass,
        "n_allowed_edges": int(E["n_edges"]),
        "support_density": float(E["density"]),
        "forbidden_mass": 0.0,
    }

    return summary, per_source, details, top1


# ============================================================
# SAVE
# ============================================================

def save_stage2(stage2_dir, prefix, support_path, E, edge_mass, history, summary, per_source, details, top1, fused_penalty):
    stage2_dir.mkdir(parents=True, exist_ok=True)

    # Compact sparse coupling:
    # row i = targets[row_offsets[i]:row_offsets[i+1]]
    np.savez_compressed(
        stage2_dir / f"{prefix}_transport_sparse.npz",
        row_offsets=E["row_offsets"],
        target_indices=E["tgt_edge_np"],
        mass=edge_mass.astype(np.float32),
    )

    np.save(stage2_dir / f"{prefix}_top1_target_index.npy", top1)
    history.to_csv(stage2_dir / f"{prefix}_iteration_history.csv", index=False)
    per_source.to_csv(stage2_dir / f"{prefix}_by_annotation.csv", index=False)
    details.to_csv(stage2_dir / f"{prefix}_top1_details.csv", index=False)
    pd.DataFrame([summary]).to_csv(stage2_dir / f"{prefix}_summary.csv", index=False)

    config = {
        "alpha2": ALPHA2,
        "fused_penalty": float(fused_penalty),
        "epsilon": EPSILON,
        "tau_a": TAU_A,
        "tau_b": TAU_B,
        "outer_iterations": OUTER_MAX,
        "inner_min": INNER_MIN,
        "inner_max": INNER_MAX,
        "inner_check_every": INNER_CHECK_EVERY,
        "inner_source_tol": INNER_TOL,
        "spatial": "raw spatial -> dimension-wise standardization -> Euclidean -> max_cost",
        "expression": "NBVAE30 Euclidean / GLOBAL source-target maximum",
        "support": str(support_path),
        "hard_support": True,
        "forbidden_edges_enter_sinkhorn": False,
        "stage1_tps_used_in_stage2_objective": False,
        "stage1_pareto_scores_used_in_stage2_objective": False,
        "annotation_used_in_optimization": False,
        "curated_used_in_optimization": False,
        "cavity_excluded_from_stage2_evaluation": False,
    }
    with open(stage2_dir / f"{prefix}_config.json", "w") as f:
        json.dump(config, f, indent=2)


# ============================================================
# RUN ONE PAIR
# ============================================================

def run_stage2_pair(pair_name, source_resolution=1.0, target_resolution=0.6):
    if jax.default_backend() != "gpu":
        raise RuntimeError(
            f"JAX backend={jax.default_backend()}, expected GPU. "
            "If JAX was already imported before CUDA_VISIBLE_DEVICES was set, restart the kernel."
        )

    cfg = PAIR_CONFIG[pair_name]
    run_dir = stage1_dir(pair_name, source_resolution, target_resolution)
    support_path = resolve_support_path(run_dir)
    stage2_dir = run_dir / "stage2"
    tag = f"src{res_tag(source_resolution)}_tgt{res_tag(target_resolution)}"
    prefix = f"{pair_name}_{tag}_stage2"

    print("\n" + "="*110)
    print(f"{pair_name} | EXACT-HARD STAGE2 FGW")
    print("stage1:", run_dir)
    print("support:", support_path)
    print("output :", stage2_dir)
    print("="*110)

    source, target = ad.read_h5ad(cfg["source"]), ad.read_h5ad(cfg["target"])
    zs = np.asarray(source.obsm[FEATURE], dtype=np.float32)
    zt = np.asarray(target.obsm[FEATURE], dtype=np.float32)
    ss = np.asarray(source.obsm[SPATIAL], dtype=np.float32)
    st = np.asarray(target.obsm[SPATIAL], dtype=np.float32)
    ns, nt = source.n_obs, target.n_obs

    # --------------------------------------------------------
    # 1. HARD SUPPORT
    # --------------------------------------------------------
    S = load_hard_support(support_path, ns, nt)
    E = build_stage2_edges(S, ns, nt)

    print("\n[1] EXACT HARD SUPPORT")
    print(f"source={ns:,} | target={nt:,} | source domains={S['n_domains']}")
    print(f"allowed edges={E['n_edges']:,} / {ns*nt:,}")
    print(f"support density={E['density']:.3%}")
    print(f"median targets/source={np.median(E['row_degree']):,.0f}")
    print(f"min/max targets/source={E['row_degree'].min():,}/{E['row_degree'].max():,}")

    # --------------------------------------------------------
    # 2. EXPRESSION LINEAR COST
    # --------------------------------------------------------
    print("\n[2] ORIGINAL EXPRESSION COST")
    expr_edge_np, c0_info = build_expression_edges(pair_name, zs, zt, E)
    print(c0_info)
    print("allowed-edge expression range:", float(expr_edge_np.min()), "->", float(expr_edge_np.max()))

    # --------------------------------------------------------
    # 3. RAW WITHIN-SLICE SPATIAL GW GEOMETRIES
    # --------------------------------------------------------
    print("\n[3] RAW SPATIAL GW GEOMETRIES")
    ss_std, st_std = standardize_spatial(ss), standardize_spatial(st)
    euclidean = costs.Euclidean()

    geom_xx = pointcloud.PointCloud(
        jnp.asarray(ss_std), cost_fn=euclidean,
        batch_size=BATCH_SIZE, scale_cost="max_cost"
    )
    geom_yy = pointcloud.PointCloud(
        jnp.asarray(st_std), cost_fn=euclidean,
        batch_size=BATCH_SIZE, scale_cost="max_cost"
    )

    a = jnp.ones(ns, dtype=jnp.float32) / ns
    b = jnp.ones(nt, dtype=jnp.float32) / nt

    qprob = quadratic_problem.QuadraticProblem(
        geom_xx=geom_xx, geom_yy=geom_yy, geom_xy=None,
        a=a, b=b, loss="sqeucl",
        tau_a=TAU_A, tau_b=TAU_B,
        gw_unbalanced_correction=True, ranks=-1
    )

    fused_penalty = float(alpha_to_fused_penalty(ALPHA2))
    print(f"alpha2={ALPHA2} | fused penalty={fused_penalty}")
    print(f"epsilon={EPSILON} | tau=({TAU_A},{TAU_B}) | outer={OUTER_MAX}")

    # --------------------------------------------------------
    # 4. SPARSE SOLVER + INITIAL COUPLING
    # --------------------------------------------------------
    print("\n[4] INITIAL HARD-FEASIBLE COUPLING")
    src_edge, tgt_edge, solve_sparse_ot = make_sparse_solver(
        E["src_edge_np"], E["tgt_edge_np"], ns, nt
    )
    expr_edge = jnp.asarray(expr_edge_np, dtype=jnp.float32)

    edge_mass = jnp.asarray(
        ((1.0 / ns) / E["row_degree"][E["src_edge_np"]]).astype(np.float32)
    )

    P = (
        jnp.zeros((ns, nt), dtype=jnp.float32)
        .at[src_edge, tgt_edge]
        .set(edge_mass)
    )

    print("initial transport mass =", float(jax.device_get(P.sum())))

    # --------------------------------------------------------
    # 5. ORIGINAL OUTER GW LINEARIZATION
    # --------------------------------------------------------
    print("\n[5] EXACT-HARD FGW")
    history = []
    old_transport_mass = float(jax.device_get(qprob.init_transport_mass()))
    t_all = time.time()

    for outer in range(1, OUTER_MAX + 1):
        t0 = time.time()
        current_mass = float(jax.device_get(P.sum()))
        rescale_factor = math.sqrt(old_transport_mass / current_mass)

        # Pure GW + unbalanced correction linearization.
        # Expression is NOT inside qprob.
        lin_prob = qprob.update_linearization(
            DenseTransport(P),
            epsilon=EPSILON,
            old_transport_mass=old_transport_mass
        )

        # Only allowed edges are retained.
        gw_edge_cost = lin_prob.geom.cost_matrix[src_edge, tgt_edge]

        # Exact original FGW local linearized cost.
        edge_cost = gw_edge_cost + fused_penalty * rescale_factor * expr_edge
        eps_eff = float(jax.device_get(lin_prob.geom.epsilon))

        edge_mass, inner_iter, inner_err = solve_sparse_ot(edge_cost, eps_eff)

        # Forbidden entries remain exactly zero.
        P_new = (
            jnp.zeros((ns, nt), dtype=jnp.float32)
            .at[src_edge, tgt_edge]
            .set(edge_mass)
        )

        new_mass = float(jax.device_get(P_new.sum()))
        history.append({
            "outer": outer,
            "transport_mass": new_mass,
            "inner_iterations": inner_iter,
            "inner_source_rel_error": inner_err,
            "epsilon_effective": eps_eff,
            "rescale_factor": rescale_factor,
            "seconds": time.time() - t0,
        })

        print(
            f"outer {outer:02d}/{OUTER_MAX} | mass={new_mass:.6g} | "
            f"inner={inner_iter:3d} | src_err={inner_err:.3e} | "
            f"eps={eps_eff:.3e} | rescale={rescale_factor:.6f} | "
            f"{history[-1]['seconds']:.1f}s",
            flush=True
        )

        # Same original OTT mass bookkeeping.
        old_transport_mass = current_mass
        P = P_new

        del lin_prob, gw_edge_cost, edge_cost, P_new
        gc.collect()

    runtime = time.time() - t_all
    history_df = pd.DataFrame(history)

    # --------------------------------------------------------
    # 6. HARD-SUPPORT SANITY
    # --------------------------------------------------------
    print("\n[6] HARD-SUPPORT SANITY")

    first_forbidden = {}
    for d in range(S["n_domains"]):
        allowed = np.zeros(nt, dtype=bool)
        allowed[S["indices"][S["indptr"][d]:S["indptr"][d+1]]] = True
        bad = np.flatnonzero(~allowed)
        first_forbidden[d] = int(bad[0]) if len(bad) else None

    fi, fj = [], []
    for i in range(ns):
        j = first_forbidden[int(S["codes"][i])]
        if j is not None:
            fi.append(i); fj.append(j)

    if fi:
        sampled_forbidden = np.asarray(jax.device_get(
            P[jnp.asarray(fi, dtype=jnp.int32), jnp.asarray(fj, dtype=jnp.int32)]
        ))
    else:
        sampled_forbidden = np.asarray([], dtype=np.float32)

    row_mass = np.asarray(jax.device_get(P.sum(axis=1)))
    print("max sampled forbidden mass =", float(sampled_forbidden.max(initial=0)))
    print("nonzero sampled forbidden =", int(np.count_nonzero(sampled_forbidden)))
    print("zero source rows =", int(np.sum(row_mass <= 0)))

    assert np.count_nonzero(sampled_forbidden) == 0
    assert np.all(row_mass > 0)

    # --------------------------------------------------------
    # 7. CPU FINAL SPARSE MASS + POST-HOC EVALUATION
    # --------------------------------------------------------
    edge_mass_np = np.asarray(jax.device_get(edge_mass), dtype=np.float32)
    if not np.isfinite(edge_mass_np).all() or np.any(edge_mass_np < 0):
        raise RuntimeError("invalid final edge mass")

    print("\n[7] POST-HOC CURATED EVALUATION — Cavity INCLUDED")
    allowed = load_stage2_allowed()
    summary, per_source, details, top1 = evaluate_stage2(
        pair_name, source, target, E, edge_mass_np, allowed
    )
    summary["runtime_seconds"] = runtime

    display(pd.DataFrame([summary]).style.format({
        "top1_weighted":"{:.2%}", "top1_macro":"{:.2%}",
        "mass_weighted":"{:.2%}", "mass_macro":"{:.2%}",
        "support_density":"{:.2%}", "transport_mass":"{:.6f}",
        "forbidden_mass":"{:.1e}", "runtime_seconds":"{:.1f}",
    }))

    # --------------------------------------------------------
    # 8. SAVE
    # --------------------------------------------------------
    save_stage2(
        stage2_dir=stage2_dir, prefix=prefix, support_path=support_path,
        E=E, edge_mass=edge_mass_np, history=history_df,
        summary=summary, per_source=per_source,
        details=details, top1=top1,
        fused_penalty=fused_penalty
    )

    print(f"\n[DONE] {stage2_dir}")

    # free large GPU objects before next pair
    del P, edge_mass, expr_edge, geom_xx, geom_yy, qprob
    gc.collect()
    try: jax.clear_caches()
    except Exception: pass

    return {
        "summary": summary,
        "per_source": per_source,
        "history": history_df,
        "stage2_dir": stage2_dir,
    }