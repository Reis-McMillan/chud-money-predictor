"""`chudp auth ...` end to end through Typer, against the in-process fake Verys."""

from __future__ import annotations

import json
import os

import pytest
from fake_verys import BASE_URL, CODE, EMAIL, REDIRECT_URI, FakeVerys
from typer.testing import CliRunner

from chud_predictor import auth
from chud_predictor.cli import app

runner = CliRunner()


@pytest.fixture
def verys(monkeypatch, tmp_path):
    fake = FakeVerys()
    monkeypatch.setattr(auth, "_make_transport", lambda: fake.transport)
    monkeypatch.setenv("CHUDP_VERYS_URL", BASE_URL)
    monkeypatch.setenv("CHUDP_VERYS_REDIRECT_URI", REDIRECT_URI)
    monkeypatch.setenv("CHUDP_AUTH_FILE", str(tmp_path / "auth" / "session.json"))
    fake.auth_file = tmp_path / "auth" / "session.json"
    return fake


def _run(*args: str, input: str | None = None):
    return runner.invoke(app, ["--env-file", "/nonexistent", *args], input=input, catch_exceptions=False)


def test_login_status_token_logout(verys):
    r = _run("auth", "login", input=f"{EMAIL}\n{CODE}\n")
    assert r.exit_code == 0, r.output
    assert "roles chud-money" in r.output
    assert oct(verys.auth_file.stat().st_mode & 0o777) == "0o600"
    assert verys.calls["/verification"] == 2 and verys.grants["authorization_code"] == 1

    r = _run("auth", "status")
    assert r.exit_code == 0, r.output
    assert EMAIL in r.output and "exchange  ok" in r.output and "chud-money" in r.output

    r = _run("auth", "token")
    tok = r.output.strip()
    assert r.exit_code == 0 and tok.count(".") == 2
    assert auth.claims(tok)["aud"] == verys.audience
    r = _run("auth", "token", "--decode")
    assert json.loads(r.output)["roles"] == ["chud-money"]

    # a second login reuses the saved cookie: no new code is requested
    before = verys.calls["/verification"]
    r = _run("auth", "login")
    assert r.exit_code == 0 and "no code needed" in r.output and verys.calls["/verification"] == before

    r = _run("auth", "logout")
    assert r.exit_code == 0 and verys.calls["/token/revoke"] == 1 and not verys.auth_file.exists()
    r = _run("auth", "status")
    assert r.exit_code == 2 and "chudp auth login" in r.output


def test_refresh_reports_rotation(verys):
    assert _run("auth", "login", "--email", EMAIL, "--code", CODE).exit_code == 0
    first = json.loads(verys.auth_file.read_text())["refresh_token"]
    r = _run("auth", "refresh")
    assert r.exit_code == 0 and "rotated" in r.output, r.output
    assert json.loads(verys.auth_file.read_text())["refresh_token"] != first
    assert verys.refresh_tokens[first]["revoked"]


def test_login_missing_role_saves_nothing(verys):
    verys.roles[:] = ["admin"]
    r = _run("auth", "login", "--email", EMAIL, "--code", CODE)
    assert r.exit_code == 2 and "roles/chud-money/identities" in r.output
    assert not verys.auth_file.exists()


def test_status_without_session(verys):
    r = _run("auth", "status")
    assert r.exit_code == 2 and "chudp auth login" in r.output


def test_env_prefix_is_isolated(monkeypatch, tmp_path):
    """A shell that exports VERYS_CLIENT_ID for the Verys dev stack must not leak into chudp."""
    monkeypatch.setenv("VERYS_CLIENT_ID", "verys-client")
    from chud_predictor.settings import SPA_CLIENT_ID, load_settings

    assert load_settings(env_file=None).verys_client_id == SPA_CLIENT_ID
    assert os.environ["VERYS_CLIENT_ID"] == "verys-client"
