"""Run the complete manuscript and supplementary analysis in order."""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from config import LOG_DIR
from utils import ensure_dirs

SCRIPTS = [
    "00_validate_inputs.py",
    "01_full_demographic_table.py",
    "01_behavioral_replication.py",
    "02_predefined_roi_voxelvolume_lmm.py",
    "03_predefined_roi_voxelvolume_behavior.py",
    "04_predefined_roi_pca.py",
    "05_predefined_roi_pca_lmm.py",
    "06_predefined_roi_pca_behavior.py",
    "07_whole_brain_voxelvolume_pca.py",
    "08_whole_brain_pca_lmm.py",
    "09_whole_brain_pca_behavior.py",
    "10_whole_brain_pca_bootstrap.py",
    "11_make_plots.py",
    "12_spatial_loading_projection.py",
    "14_export_manuscript_tables.py",
    "13_build_results_index.py",
]


def main() -> None:
    ensure_dirs([LOG_DIR])
    code_dir = Path(__file__).resolve().parent
    log_path = LOG_DIR / "run_all.log"
    env = os.environ.copy()
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"Run started: {datetime.now().isoformat()}\n")
        for script in SCRIPTS:
            message = f"=== Running {script} ==="
            print(message, flush=True)
            log.write(message + "\n")
            completed = subprocess.run(
                [sys.executable, str(code_dir / script)],
                cwd=code_dir,
                env=env,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            print(completed.stdout, end="", flush=True)
            log.write(completed.stdout)
            log.write(f"return_code={completed.returncode}\n")
            log.flush()
            if completed.returncode != 0:
                raise SystemExit(f"Pipeline stopped because {script} failed. See {log_path}.")
        finished = f"Run completed: {datetime.now().isoformat()}"
        print(finished, flush=True)
        log.write(finished + "\n")
    print("Complete pipeline finished successfully.", flush=True)


if __name__ == "__main__":
    main()
