from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectConfig:
    """Runtime configuration for data, artifacts, and training defaults."""

    project_root: Path = Path(__file__).resolve().parents[2]
    raw_data_path: Path = project_root / "credit_score.csv"
    artifacts_dir: Path = project_root / "artifacts"
    model_path: Path = artifacts_dir / "credit_score_model.joblib"
    metrics_path: Path = artifacts_dir / "metrics.json"
    random_state: int = 42
    test_size: float = 0.2
    target_column: str = "Credit_Score"

    def ensure_artifact_dir(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
