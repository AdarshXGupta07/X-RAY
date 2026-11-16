"""Settings, read once from environment variables (all optional).

TBSHIFT_RUN        P2 run whose model the demo serves        (default p2_densenet)
TBSHIFT_FOLD       which trained model of that run           (default train_shenzhen__none)
TBSHIFT_ROOT       project folder holding artifacts/ and results/ (default: current folder)
TBSHIFT_DEVICE     cpu or cuda                                 (default cpu)
TBSHIFT_FRONTEND   built frontend folder to serve at /         (default frontend/dist)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    run: str = "p2_densenet"
    fold: str = "train_shenzhen__none"
    root: Path = Path(".")
    device: str = "cpu"
    frontend_dir: Path = Path("frontend/dist")

    @property
    def source_site(self) -> str:
        return self.fold.split("__")[0].removeprefix("train_")

    @property
    def corner_mask(self) -> tuple[float, float] | None:
        # Matches configs/p2_densenet.yaml.
        return (0.25, 0.12) if self.fold.endswith("__corners") else None

    @property
    def checkpoint(self) -> Path:
        return self.root / "artifacts" / "checkpoints" / self.run / f"{self.fold}.pt"

    @property
    def source_scores(self) -> Path:
        return self.root / "artifacts" / "scores" / self.run / self.fold / f"{self.source_site}.npz"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def model_name(self) -> str:
        return f"{self.run}/{self.fold}"


def load_settings() -> Settings:
    env = os.environ.get
    return Settings(
        run=env("TBSHIFT_RUN", Settings.run),
        fold=env("TBSHIFT_FOLD", Settings.fold),
        root=Path(env("TBSHIFT_ROOT", ".")),
        device=env("TBSHIFT_DEVICE", Settings.device),
        frontend_dir=Path(env("TBSHIFT_FRONTEND", str(Settings.frontend_dir))),
    )
