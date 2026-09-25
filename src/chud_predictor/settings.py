"""Environment-driven settings. All timestamps in this project are tz-naive UTC.

Data comes from the chud-money API (`CHUDP_API_BASE`), whose data stream needs a Verys access token
exchanged for the API's own audience; see `auth.py`. Every value has a production default, so an
empty `.env` is fine; the variables exist for a local dev stack. They are all `CHUDP_`-prefixed on
purpose: bare names such as `VERYS_CLIENT_ID` are commonly exported in a shell for the Verys dev
stack itself and must not leak in here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# The chud-money SPA's public PKCE client. It is also the API's own client id, so the token-exchange
# audience is the same value (verified against https://chud-money.mcmlln.dev/config.js).
SPA_CLIENT_ID = "99e6d288-fab8-4e18-a4df-ada501e18fce"


@dataclass(frozen=True)
class Settings:
    # chud-money API
    api_base: str = "https://api.chud-money.mcmlln.dev"
    market_tag: str = "btc-15m"
    api_timeout_s: float = 120.0          # per-read timeout on the SSE stream (keep-alives every 15 s), not a total cap
    # Verys (auth server in front of the API)
    verys_url: str = "https://api.verys.mcmlln.dev"
    verys_client_id: str = SPA_CLIENT_ID
    chud_money_client_id: str = SPA_CLIENT_ID          # token-exchange audience
    verys_redirect_uri: str = "https://chud-money.mcmlln.dev/auth/callback"   # must be registered verbatim
    auth_file: Path = Path(".auth/session.json")       # written by `chudp auth login`
    # local paths / compute
    data_dir: Path = Path("data")
    hf_home: Path = Path("models")
    device: str | None = None

    @property
    def raw_dir(self) -> Path:
        """Raw BRTI ticks, one Parquet per UTC day."""
        return self.data_dir / "raw" / "brti"

    @property
    def contracts_raw_dir(self) -> Path:
        """Raw Kalshi KXBTC15M 1-minute candles, one Parquet per UTC day."""
        return self.data_dir / "raw" / "contracts"

    def raw_dir_for(self, source: str) -> Path:
        return {"brti": self.raw_dir, "contracts": self.contracts_raw_dir}[source]

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def backtests_dir(self) -> Path:
        return self.data_dir / "backtests"

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

    env = os.environ.get
    return Settings(
        api_base=env("CHUDP_API_BASE", Settings.api_base).rstrip("/"),
        market_tag=env("CHUDP_MARKET", Settings.market_tag),
        api_timeout_s=float(env("CHUDP_API_TIMEOUT_S", str(Settings.api_timeout_s))),
        verys_url=env("CHUDP_VERYS_URL", Settings.verys_url).rstrip("/"),
        verys_client_id=env("CHUDP_VERYS_CLIENT_ID", Settings.verys_client_id),
        chud_money_client_id=env("CHUDP_API_AUDIENCE", Settings.chud_money_client_id),
        verys_redirect_uri=env("CHUDP_VERYS_REDIRECT_URI", Settings.verys_redirect_uri),
        auth_file=Path(env("CHUDP_AUTH_FILE", str(Settings.auth_file))),
        data_dir=data_dir or Path(env("CHUDP_DATA_DIR", "data")),
        hf_home=hf_home,
        device=env("CHUDP_DEVICE") or None,
    )
