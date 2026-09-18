"""Environment-driven settings. All timestamps in this project are tz-naive UTC."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class MissingPasswordError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    qdb_host: str = "127.0.0.1"
    qdb_port: int = 19000
    qdb_user: str = "admin"
    qdb_password: str = ""
    qdb_timeout_s: float = 120.0
    data_dir: Path = Path("data")
    hf_home: Path = Path("models")
    device: str | None = None

    @property
    def qdb_url(self) -> str:
        return f"http://{self.qdb_host}:{self.qdb_port}"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw" / "brti"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def backtests_dir(self) -> Path:
        return self.data_dir / "backtests"

    @property
    def forecasts_dir(self) -> Path:
        return self.data_dir / "forecasts"

    @property
    def finetune_dir(self) -> Path:
        return self.data_dir / "finetune"


def load_settings(env_file: Path | None = Path(".env"), data_dir: Path | None = None) -> Settings:
    """Load `.env` (if present) then the process environment. Explicit env vars win over .env."""
    if env_file is not None and env_file.exists():
        load_dotenv(env_file, override=False)

    hf_home = Path(os.environ.get("HF_HOME", "models"))
    # Make the HuggingFace cache project-local unless the user already pointed it elsewhere.
    os.environ.setdefault("HF_HOME", str(hf_home.resolve()))

    return Settings(
        qdb_host=os.environ.get("QDB_HOST", "127.0.0.1"),
        qdb_port=int(os.environ.get("QDB_PORT", "19000")),
        qdb_user=os.environ.get("QDB_USER", "admin"),
        qdb_password=os.environ.get("QDB_PASSWORD", ""),
        qdb_timeout_s=float(os.environ.get("QDB_TIMEOUT_S", "120")),
        data_dir=data_dir or Path(os.environ.get("CHUDP_DATA_DIR", "data")),
        hf_home=hf_home,
        device=os.environ.get("CHUDP_DEVICE") or None,
    )


def require_password(settings: Settings) -> str:
    if not settings.qdb_password:
        raise MissingPasswordError(
            "QDB_PASSWORD is not set. Run `make creds` (fetches it from the cluster into .env) "
            "or paste it into .env by hand; see .env.example."
        )
    return settings.qdb_password
