"""Offline tests for `chud_predictor.auth` against `tests/fake_verys.FakeVerys`.

Everything goes through `httpx.MockTransport`; no sockets, no sleeps, no real clock (the fake and the
provider share an injected `Clock`).
"""

from __future__ import annotations

import json
import os
import stat
import threading
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from fake_verys import EMAIL, REDIRECT_URI, FakeVerys

from chud_predictor import auth
from chud_predictor.auth import (
    AuthError,
    MissingRoleError,
    OAuthError,
    Session,
    SessionExpiredError,
    StaticTokenProvider,
    VerysClient,
    VerysTokenProvider,
    claims,
    expires_within,
    is_login_page,
    parse_consent,
    token_exp,
)

EXCHANGE = "urn:ietf:params:oauth:grant-type:token-exchange"
LOGGED_IN = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)


class Clock:
    """A settable clock shared by the fake (token `exp`) and the provider (renewal decisions)."""

    def __init__(self, t: float = 1_700_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def fake(clock: Clock) -> FakeVerys:
    return FakeVerys(clock=clock)


@pytest.fixture
def auth_file(tmp_path: Path) -> Path:
    return tmp_path / ".auth" / "session.json"


def client_for(fake: FakeVerys, **kw) -> VerysClient:
    return VerysClient(fake.issuer, fake.client_id, transport=fake.transport, **kw)


def login(fake: FakeVerys, *, email: str = EMAIL, redirect_uri: str = REDIRECT_URI) -> Session:
    """The `chudp auth login` path: email code -> cookies -> authorize -> code grant -> exchange."""
    with client_for(fake) as client:
        client.send_code(email)
        client.verify_code(email, fake.code)
        return auth.mint_session(
            client,
            audience=fake.audience,
            redirect_uri=redirect_uri,
            verys_url=fake.issuer,
            client_id=fake.client_id,
            email=email,
            cookies=client.cookies,
            logged_in_at=LOGGED_IN,
        )


def provider_for(fake: FakeVerys, auth_file: Path, *, now=None, early_s: float = 30.0) -> VerysTokenProvider:
    return VerysTokenProvider(
        verys_url=fake.issuer,
        client_id=fake.client_id,
        audience=fake.audience,
        auth_file=auth_file,
        redirect_uri=REDIRECT_URI,
        early_s=early_s,
        transport=fake.transport,
        now=now or fake.clock,
    )


def logged_in_provider(fake: FakeVerys, auth_file: Path, **kw) -> VerysTokenProvider:
    login(fake, redirect_uri=REDIRECT_URI).save(auth_file)
    return provider_for(fake, auth_file, **kw)


# ----------------------------------------------------------------------------------- login path


def test_login_writes_a_0600_session_with_the_expected_keys(fake: FakeVerys, auth_file: Path):
    session = login(fake)
    session.save(auth_file)

    assert stat.S_IMODE(auth_file.stat().st_mode) == 0o600
    assert stat.S_IMODE(auth_file.parent.stat().st_mode) == 0o700
    raw = json.loads(auth_file.read_text())
    assert set(raw) == {
        "version", "verys_url", "client_id", "audience", "redirect_uri", "email", "sub",
        "refresh_token", "cookies", "logged_in_at", "obtained_at", "roles",
    }
    assert raw["version"] == 1
    assert raw["email"] == EMAIL and raw["sub"] == fake.sub
    assert raw["roles"] == ["chud-money"]
    assert raw["refresh_token"] == "rt-1"
    assert raw["cookies"] == {"token": fake.cookie_value, "token_iv": fake.cookie_iv}
    assert raw["logged_in_at"].startswith("2026-09-24T12:00:00")
    assert "id_token" not in auth_file.read_text()
    assert fake.grants["authorization_code"] == 1 and fake.grants[EXCHANGE] == 1
    assert Session.load(auth_file) == session


def test_login_walks_the_consent_page(clock: Clock):
    fake = FakeVerys(clock=clock, require_consent=True)
    session = login(fake)

    assert fake.calls["/authorize/consent"] == 1
    assert fake.consents == {EMAIL}
    assert session.roles == ["chud-money"]
    # A second login now goes straight through, with no consent page.
    login(fake)
    assert fake.calls["/authorize/consent"] == 1


def test_bad_verification_code_saves_nothing(fake: FakeVerys, auth_file: Path):
    with client_for(fake) as client:
        client.send_code(EMAIL)
        with pytest.raises(AuthError, match="Invalid or expired code"):
            client.verify_code(EMAIL, "000000")
    assert not auth_file.exists()


def test_unknown_email_is_a_clear_error(fake: FakeVerys):
    with client_for(fake) as client, pytest.raises(AuthError, match="no Verys account for nobody@example.com"):
        client.send_code("nobody@example.com")
    assert fake.pending_codes == {}


def test_unregistered_redirect_uri_is_refused(fake: FakeVerys, auth_file: Path):
    with pytest.raises(OAuthError) as excinfo:
        login(fake, redirect_uri="https://chud-money.test/")
    assert excinfo.value.error == "Invalid redirect_uri"
    assert excinfo.value.status == 400
    assert excinfo.value.where == "authorize"
    assert not auth_file.exists()


def test_missing_role_names_the_grant_command(clock: Clock, auth_file: Path):
    fake = FakeVerys(clock=clock, roles=["admin"])
    with pytest.raises(MissingRoleError) as excinfo:
        login(fake)
    msg = str(excinfo.value)
    assert f"POST {fake.issuer}/roles/chud-money/identities/{EMAIL}" in msg
    assert "['admin']" in msg
    assert not auth_file.exists()


def test_denied_consent_surfaces_access_denied(clock: Clock):
    fake = FakeVerys(clock=clock, require_consent=True)
    with client_for(fake) as client:
        client.send_code(EMAIL)
        client.verify_code(EMAIL, fake.code)
        # Drive the deny button by hand; `authorize_code` only ever approves.
        resp = client._http.get(
            "/authorize",
            params={
                "response_type": "code", "client_id": fake.client_id, "redirect_uri": REDIRECT_URI,
                "scope": "openid", "state": "st", "code_challenge": "x", "code_challenge_method": "S256",
            },
        )
        form = parse_consent(resp.text)
        assert form is not None
        denied = client._http.post(
            "/authorize/consent",
            data={"oauth2_session_id": form.session_id, "csrf_token": form.csrf_token, "consent_action": "deny"},
        )
    assert "error=access_denied" in denied.headers["location"]


# ------------------------------------------------------------------------------ HTML scraping


CONSENT_ODD_ORDER = """
<html><body><form method="POST" action="https://verys.test/authorize/consent">
  <input value='sess-9' name='oauth2_session_id' type='hidden'>
  <input type="hidden" value="csrf-9" name="csrf_token">
  <button value="approve" name="consent_action" type="submit">Approve</button>
</form></body></html>
"""


def test_consent_parser_is_attribute_order_independent():
    form = parse_consent(CONSENT_ODD_ORDER)
    assert form is not None
    assert (form.session_id, form.csrf_token) == ("sess-9", "csrf-9")


@pytest.mark.parametrize(
    "html",
    [
        "<html><body><p>nothing to see</p></body></html>",
        "",
        # the two fields but no consent control: not a consent page
        '<form><input type="hidden" name="oauth2_session_id" value="s"><input type="hidden" name="csrf_token" value="c"></form>',
    ],
)
def test_consent_parser_returns_none_on_unrelated_html(html: str):
    assert parse_consent(html) is None


def test_login_page_is_recognised():
    from fake_verys import LOGIN_HTML

    assert is_login_page(LOGIN_HTML)
    assert parse_consent(LOGIN_HTML) is None
    assert not is_login_page(CONSENT_ODD_ORDER)


def test_unknown_html_from_authorize_is_an_oauth_error(fake: FakeVerys):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, html="<html><body>maintenance</body></html>"))
    with VerysClient(fake.issuer, fake.client_id, transport=transport) as client:
        with pytest.raises(OAuthError) as excinfo:
            client.authorize_code(REDIRECT_URI)
    assert excinfo.value.error == "unexpected_response"
    assert "browser" in excinfo.value.description


def test_dead_cookies_on_authorize_ask_for_a_new_login(fake: FakeVerys):
    fake.cookies_valid = False
    with client_for(fake, cookies={"token": "stale", "token_iv": "stale"}) as client:
        with pytest.raises(SessionExpiredError, match="chudp auth login"):
            client.authorize_code(REDIRECT_URI)


# --------------------------------------------------------------------------------- provider


def test_token_is_cached_until_the_early_window(fake: FakeVerys, auth_file: Path, clock: Clock):
    provider = logged_in_provider(fake, auth_file)
    exchanges_after_login = fake.grants[EXCHANGE]

    first = provider.token()
    assert provider.token() == first
    assert fake.grants[EXCHANGE] == exchanges_after_login + 1
    assert claims(first)["aud"] == fake.audience

    clock.t += 269  # 31 s of life left: still cached
    assert provider.token() == first
    clock.t += 2  # 29 s left: inside the 30 s early window
    assert provider.token() != first
    assert fake.grants[EXCHANGE] == exchanges_after_login + 2


def test_rotated_refresh_token_is_persisted_and_the_old_one_dies(fake: FakeVerys, auth_file: Path):
    provider = logged_in_provider(fake, auth_file)
    assert json.loads(auth_file.read_text())["refresh_token"] == "rt-1"

    provider.token()

    assert json.loads(auth_file.read_text())["refresh_token"] == "rt-2"
    assert Session.load(auth_file).refresh_age.total_seconds() >= 0
    with client_for(fake) as client, pytest.raises(OAuthError) as excinfo:
        client.refresh("rt-1")
    assert excinfo.value.error == "invalid_grant"


def test_revoked_refresh_token_recovers_through_the_stored_cookies(fake: FakeVerys, auth_file: Path):
    provider = logged_in_provider(fake, auth_file)
    fake.refresh_tokens["rt-1"]["revoked"] = True  # as an SPA sign-out would
    authorizes = fake.calls["/authorize"]

    token = provider.token()

    assert claims(token)["aud"] == fake.audience
    assert fake.calls["/authorize"] == authorizes + 1  # a fresh code, no email round trip
    assert json.loads(auth_file.read_text())["refresh_token"] == "rt-2"
    assert fake.calls["/verification"] == 2  # only the two calls the login made


def test_dead_refresh_token_and_dead_cookies_need_a_human(fake: FakeVerys, auth_file: Path):
    provider = logged_in_provider(fake, auth_file)
    fake.refresh_tokens["rt-1"]["revoked"] = True
    fake.cookies_valid = False

    with pytest.raises(SessionExpiredError) as excinfo:
        provider.token()
    assert str(excinfo.value).endswith("run `chudp auth login`")


def test_missing_session_file_asks_for_login(fake: FakeVerys, tmp_path: Path):
    provider = provider_for(fake, tmp_path / "nope.json")
    with pytest.raises(SessionExpiredError, match="no saved Verys session"):
        provider.token()


def test_exchange_invalid_grant_is_retried_once(fake: FakeVerys, auth_file: Path):
    provider = logged_in_provider(fake, auth_file)
    provider.token()
    provider.invalidate()
    fake.fail_next_exchange = True
    exchanges, refreshes = fake.grants[EXCHANGE], fake.grants["refresh_token"]

    token = provider.token()

    assert claims(token)["aud"] == fake.audience
    assert fake.grants[EXCHANGE] == exchanges + 2  # refused, then retried with a fresh subject token
    assert fake.grants["refresh_token"] == refreshes + 2


def test_exchange_invalid_grant_is_not_retried_twice(fake: FakeVerys, auth_file: Path, monkeypatch):
    provider = logged_in_provider(fake, auth_file)
    monkeypatch.setattr(
        fake, "_exchange_grant", lambda form: httpx.Response(400, json={"error": "invalid_grant"})
    )
    exchanges = fake.grants[EXCHANGE]
    with pytest.raises(OAuthError) as excinfo:
        provider.token()
    assert excinfo.value.error == "invalid_grant"
    assert fake.grants[EXCHANGE] == exchanges + 2  # the first attempt and the single retry, no more


def test_missing_role_on_renewal_is_reported(clock: Clock, auth_file: Path):
    fake = FakeVerys(clock=clock)
    provider = logged_in_provider(fake, auth_file)
    fake.roles = ["admin"]
    with pytest.raises(MissingRoleError):
        provider.token()


def test_exchange_without_consent_hints_at_the_browser(fake: FakeVerys, auth_file: Path):
    provider = logged_in_provider(fake, auth_file)
    fake.consents.clear()  # consent row deleted server-side; /authorize would re-ask interactively
    with pytest.raises(AuthError, match="no consent record"):
        provider.token()


def test_concurrent_token_calls_refresh_exactly_once(fake: FakeVerys, auth_file: Path):
    provider = logged_in_provider(fake, auth_file)
    results: list[str] = []
    errors: list[BaseException] = []
    start = threading.Barrier(32)

    def worker() -> None:
        try:
            start.wait(timeout=10)
            results.append(provider.token())
        except BaseException as e:  # noqa: BLE001 - re-raised in the assertion below
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors
    assert len(set(results)) == 1 and len(results) == 32
    assert fake.grants["refresh_token"] == 1
    assert fake.grants[EXCHANGE] == 2  # one from login, one for the whole storm


def test_provider_exposes_email_and_claims(fake: FakeVerys, auth_file: Path):
    provider = logged_in_provider(fake, auth_file)
    assert provider.email == EMAIL
    payload = provider.claims()
    assert payload["aud"] == fake.audience
    assert payload["roles"] == ["chud-money"]
    assert payload["exp"] > fake.clock()
    provider.close()
    assert provider._token is None


def test_token_provider_factory_reads_settings(tmp_path: Path):
    from chud_predictor.settings import Settings

    settings = Settings(auth_file=tmp_path / "s.json")
    provider = auth.token_provider(settings)
    assert (provider.verys_url, provider.client_id) == (settings.verys_url, settings.verys_client_id)
    assert provider.audience == settings.chud_money_client_id
    assert provider.redirect_uri == settings.verys_redirect_uri
    assert provider.auth_file == tmp_path / "s.json"
    assert provider.lock_file == tmp_path / "s.lock"


def test_transport_seam_is_monkeypatchable(fake: FakeVerys, auth_file: Path, monkeypatch):
    login(fake).save(auth_file)
    monkeypatch.setattr(auth, "_make_transport", lambda: fake.transport)
    provider = VerysTokenProvider(
        verys_url=fake.issuer,
        client_id=fake.client_id,
        audience=fake.audience,
        auth_file=auth_file,
        redirect_uri=REDIRECT_URI,
        now=fake.clock,
    )
    assert claims(provider.token())["aud"] == fake.audience


def test_mint_session_accepts_a_naive_datetime_or_the_stored_iso_string(fake: FakeVerys):
    """`chudp auth login` re-mints from the cookie jar and keeps the original login time (a string)."""
    first = login(fake)
    with client_for(fake, cookies=first.cookies) as client:
        again = auth.mint_session(
            client,
            audience=fake.audience,
            redirect_uri=REDIRECT_URI,
            verys_url=fake.issuer,
            client_id=fake.client_id,
            email=first.email,
            cookies=first.cookies,
            logged_in_at=first.logged_in_at,
        )
    assert again.logged_in_at == first.logged_in_at
    assert again.refresh_token != first.refresh_token  # a private refresh token per login
    with client_for(fake, cookies=first.cookies) as client:
        naive = auth.mint_session(
            client,
            audience=fake.audience,
            redirect_uri=REDIRECT_URI,
            verys_url=fake.issuer,
            client_id=fake.client_id,
            email=first.email,
            cookies=first.cookies,
            logged_in_at=datetime(2026, 9, 24, 12, 0),  # tz-naive UTC, like the rest of the project
        )
    assert naive.logged_in_at == "2026-09-24T12:00:00+00:00"


def test_verys_client_uses_the_transport_seam(fake: FakeVerys, monkeypatch):
    """cli.py builds clients without a transport, so the seam has to reach them too."""
    monkeypatch.setattr(auth, "_make_transport", lambda: fake.transport)
    with VerysClient(fake.issuer, fake.client_id) as client:
        client.send_code(EMAIL)
    assert fake.calls["/verification"] == 1


def test_revoke_is_best_effort(fake: FakeVerys):
    with client_for(fake) as client:
        client.revoke("rt-does-not-exist")
    assert fake.calls["/token/revoke"] == 1

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    with VerysClient(fake.issuer, fake.client_id, transport=httpx.MockTransport(boom)) as client:
        client.revoke("rt-1")  # must not raise


# ------------------------------------------------------------------------------ session file


def test_save_is_atomic_when_replace_fails(fake: FakeVerys, auth_file: Path, monkeypatch):
    session = login(fake)
    session.save(auth_file)
    before = auth_file.read_text()

    monkeypatch.setattr(auth.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    session.refresh_token = "rt-later"
    with pytest.raises(OSError, match="disk full"):
        session.save(auth_file)

    assert auth_file.read_text() == before
    assert list(auth_file.parent.glob("*.tmp")) == []


def test_load_warns_when_the_file_is_world_readable(fake: FakeVerys, auth_file: Path, caplog):
    login(fake).save(auth_file)
    os.chmod(auth_file, 0o644)
    with caplog.at_level("WARNING"):
        Session.load(auth_file)
    assert "credential" in caplog.text


def test_load_rejects_a_truncated_file(auth_file: Path):
    auth_file.parent.mkdir(parents=True)
    auth_file.write_text('{"email": "a@b.c"}')
    with pytest.raises(SessionExpiredError, match="incomplete"):
        Session.load(auth_file)
    auth_file.write_text("not json")
    with pytest.raises(SessionExpiredError, match="cannot read"):
        Session.load(auth_file)


def test_load_ignores_unknown_keys(fake: FakeVerys, auth_file: Path):
    login(fake).save(auth_file)
    raw = json.loads(auth_file.read_text()) | {"id_token": "leaked", "future_field": 1}
    auth_file.write_text(json.dumps(raw))
    assert Session.load(auth_file).email == EMAIL


# -------------------------------------------------------------------------------- jwt helpers


def test_claims_helpers(fake: FakeVerys, clock: Clock):
    token = fake.access_token(aud="someone", ttl=120)
    assert claims(token)["aud"] == "someone"
    assert claims(token)["sub"] == fake.sub
    assert token_exp(token) == clock.t + 120
    assert not expires_within(token, 30, now=clock)
    assert expires_within(token, 121, now=clock)
    assert expires_within(token, 0, now=lambda: clock.t + 200)
    assert auth.token_roles(token) == ["chud-money"]


@pytest.mark.parametrize("bad", ["", "not-a-jwt", "a.b", "a.!!!.c", "a.e30=.c.d"])
def test_claims_rejects_malformed_tokens(bad: str):
    with pytest.raises(AuthError):
        claims(bad)


def test_token_exp_requires_the_claim():
    import base64 as b64

    payload = b64.urlsafe_b64encode(json.dumps({"sub": "x"}).encode()).rstrip(b"=").decode()
    with pytest.raises(AuthError, match="exp"):
        token_exp(f"h.{payload}.s")


def test_static_provider():
    provider = StaticTokenProvider("tok")
    assert provider.token() == "tok"
    provider.invalidate()
    assert provider.token() == "tok"
