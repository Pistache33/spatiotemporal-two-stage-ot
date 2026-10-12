from pathlib import Path
import importlib.util
import sys

SCRIPT = Path(
    "/mnt/disk2/lirui/baseline_scripts/"
    "run_mosta_socs_official_unsupervised.py"
)

spec = importlib.util.spec_from_file_location(
    "socs_grid_runner", SCRIPT
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

m.PAIR_CONFIG["PairC"] = {
    "name": "E10.5_E2S1 -> E11.5_E1S1",
    "source": Path(
        "/mnt/disk2/lirui/mosta_e105_e115_nbvae30/data/"
        "E10.5_E2S1.nbvae30.h5ad"
    ),
    "target": Path(
        "/mnt/disk2/lirui/mosta_e105_e115_nbvae30/data/"
        "E11.5_E1S1.nbvae30.h5ad"
    ),
    "source_resolution": 1.0,
    "target_resolution": 1.4,
}

sys.argv = [
    str(SCRIPT),

    "--pairs", "PairC",

    "--output-root",
    "/mnt/disk2/lirui/mosta_e105_e115_baselines/socs",

    "--alpha-values",
    "0.2", "0.4", "0.6",

    "--alpha-mode",
    "match_moscot",

    "--epsilon",
    "1e-4",

    "--max-iters",
    "30",

    "--method",
    "fugw_cpu",

    "--resume",
    "--continue-on-error",
]

m.main()
