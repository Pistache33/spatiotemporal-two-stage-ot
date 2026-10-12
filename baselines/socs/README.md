# SOCS baseline

Official SOCS implementation, called via our wrapper. The SOCS source is
**not** vendored; it is a dependency.

## Files

| File | Role |
|---|---|
| `run_official.py` | official SOCS caller (`run_mosta_socs_official_unsupervised.py`) |
| `resume_pairc.py` | the real Pair C resume entry (`resume_socs_pairc.py`) |

## Actual run

```bash
python baselines/socs/run_official.py --pairs PairA ...   # Pair A
python baselines/socs/resume_pairc.py                     # Pair C resume
```

## Parameter mapping (MOSCOT-compatible)

- SOCS `alpha` = **expression** weight (opposite of MOSCOT's GW weight).
- The caller maps `alpha_socs = 1 − alpha_moscot`
  (`--alpha-mode match_moscot`).
- τ conversion: `rho = ε·τ/(1−τ)`, with τ=1 ⇒ ρ=∞.
- Grid: α_moscot ∈ {0.2, 0.4, 0.6} × τ ∈ {(1,.01),(1,.05),(.05,1),(.01,1)},
  ε = 1e-4, max_iters = 30, method `fugw_cpu`.
- Embedding: `X_nbvae_30`; Leiden `n_neighbors=15`, `seed=0`, 2 iterations.

## Best-run selection rule

Maximum `weighted_accuracy`.

## Third-party source & license

- Repository: https://github.com/algo-bio-lab/SOCS-main
- License: MIT, (c) 2025 algo-bio-lab
- Record the exact commit used before publishing; the original runs used a
  checkout at `/mnt/disk2/lirui/baseline_scripts/SOCS-main` with a dedicated
  conda env (`/mnt/disk2/lirui/conda_envs/socs-official`).
