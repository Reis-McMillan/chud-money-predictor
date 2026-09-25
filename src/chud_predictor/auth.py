"""Headless Verys auth for the chud-money API: email-code login, then unattended refresh + exchange.

Verys has no password or client-credentials grant, so `chudp auth login` drives the same public
PKCE flow the SPA uses (one emailed code), stores the resulting refresh token *and* the browser
session cookies in `.auth/session.json`, and from then on mints API tokens without a human:

    refresh (Verys access token, aud = issuer) -> RFC 8693 exchange (aud = chud-money) -> Bearer

Invariants worth keeping in mind:
* Exchanged tokens live 5 minutes and carry no refresh token, so they are cached in memory only and
  renewed 30 s early (chud-money allows 30 s of skew).
* The cookies are stored on purpose: they outlive refresh tokens (an SPA sign-out revokes every
  refresh token of identity+client but not our cookie copy), so a lost refresh token recovers
  silently through `/authorize` instead of a new email code.
* `id_token` is never written to disk, and no token is ever logged.
* The session file is a ~60-day credential: 0600 in a 0700 directory, written atomically, and
  updated under `flock` because two `chudp` processes may share it.
* JWT payloads are read unverified (base64url only); chud-money verifies signatures.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import logging
import os
import secrets
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qs, urlparse

import httpx

from .settings import Settings

log = logging.getLogger(__name__)

#: Verys role an identity needs before chud-money serves it (`middleware/authenticated.rs`).
REQUIRED_ROLE = "chud-money"
#: Scopes the SPA's public client is allowed; requesting fewer would narrow the stored consent.
SCOPE = "openid email profile"
LOGIN_HINT = "run `chudp auth login`"
EXCHANGE_GRANT = "urn:ietf:params:oauth:grant-type:token-exchange"
ACCESS_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"
COOKIE_NAMES = ("token", "token_iv")
_REDIRECT_CODES = (301, 302, 303, 307, 308)


# --------------------------------------------------------------------------------------- errors


class AuthError(RuntimeError):
    """Anything that stops us from producing a chud-money bearer token."""


class OAuthError(AuthError):
    """An OAuth2 error response (`{"error", "error_description"}`) from Verys."""

    def __init__(self, error: str, description: str = "", status: int = 0, where: str = "") -> None:
        self.error = error
        self.description = description
        self.status = status
        self.where = where
        detail = f": {description}" if description else ""
        super().__init__(f"Verys {where or 'request'} failed [{status}] {error}{detail}")


class SessionExpiredError(AuthError):
    """The stored credentials cannot be recovered without a human; message ends with the login hint."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"{reason}; {LOGIN_HINT}")


class MissingRoleError(AuthError):
    """Authentication worked but the identity lacks `chud-money`; the message says how to grant it."""

    def __init__(self, email: str, verys_url: str, roles: list[str]) -> None:
        self.email = email
        self.roles = roles
        super().__init__(
            f"{email} has roles {roles or '[]'} but not '{REQUIRED_ROLE}'; an admin must "
            f"POST {verys_url}/roles/{REQUIRED_ROLE}/identities/{email}"
        )


def _oauth_error(resp: httpx.Response, where: str) -> OAuthError:
    """Build an `OAuthError` from a Verys body, which is `{"error", "error_description"}` on the
    token endpoint and `{"error": "<human message>"}` on the browser-facing ones."""
    body: Any = None
    with suppress(ValueError):
        body = resp.json()
    if isinstance(body, dict):
        error = str(body.get("error") or f"http_{resp.status_code}")
        description = str(body.get("error_description") or body.get("message") or "")
    else:
        error, description = f"http_{resp.status_code}", resp.text[:200]
    return OAuthError(error, description, resp.status_code, where)


# ----------------------------------------------------------------------------------- jwt helpers


def claims(token: str) -> dict[str, Any]:
    """Unverified JWT payload (chud-money verifies the signature; we only read `exp`/`roles`/`sub`)."""
    parts = token.split(".")
    if len(parts) != 3:
        raise AuthError("not a JWT: expected three dot-separated segments")
    payload = parts[1]
    try:
        raw = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
        data = json.loads(raw)
    except Exception as e:  # noqa: BLE001 - any malformed token is one error to the caller
        raise AuthError(f"cannot decode JWT payload: {e}") from e
    if not isinstance(data, dict):
        raise AuthError("JWT payload is not an object")
    return data


def token_exp(token: str) -> float:
    """Expiry of `token` as a POSIX timestamp."""
    exp = claims(token).get("exp")
    if not isinstance(exp, (int, float)):
        raise AuthError("JWT has no numeric `exp` claim")
    return float(exp)


def expires_within(token: str, seconds: float, *, now: Callable[[], float] = time.time) -> bool:
    """True when `token` is already expired or expires within `seconds` (renew early, not late)."""
    return token_exp(token) - now() <= seconds


def token_roles(token: str) -> list[str]:
    """Roles carried by `token`, `[]` when the claim is absent or malformed."""
    roles = claims(token).get("roles")
    return [str(r) for r in roles] if isinstance(roles, list) else []


# ---------------------------------------------------------------------------------- consent HTML


@dataclass(frozen=True)
class ConsentForm:
    """The two hidden fields `/authorize/consent` needs back."""

    session_id: str
    csrf_token: str


class _FormParser(HTMLParser):
    """Collects hidden inputs, submit-control names and form ids, independent of attribute order."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hidden: dict[str, str] = {}
        self.controls: set[str] = set()
        self.form_ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "form" and a.get("id"):
            self.form_ids.add(a["id"])
        elif tag == "input":
            if a.get("type", "").lower() == "hidden" and a.get("name"):
                self.hidden[a["name"]] = a.get("value", "")
            elif a.get("name"):
                self.controls.add(a["name"])
        elif tag == "button" and a.get("name"):
            self.controls.add(a["name"])


def parse_consent(html: str) -> ConsentForm | None:
    """Scrape Verys' consent page; `None` for any other HTML (login page, error page, anything)."""
    p = _FormParser()
    p.feed(html)
    session_id, csrf = p.hidden.get("oauth2_session_id"), p.hidden.get("csrf_token")
    if session_id and csrf and "consent_action" in p.controls:
        return ConsentForm(session_id, csrf)
    return None


def is_login_page(html: str) -> bool:
    """True for Verys' `login.html`, which is what `/authorize` serves when our cookies are dead."""
    p = _FormParser()
    p.feed(html)
    return "email-form" in p.form_ids


# ----------------------------------------------------------------------------------- verys client


@dataclass(frozen=True)
class Tokens:
    """One `/token` response. `refresh_token`/`id_token` are absent from the exchange grant."""

    access_token: str
    refresh_token: str | None = None
    id_token: str | None = None
    expires_in: int | None = None


class VerysClient:
    """The handful of Verys calls this project makes, as a public PKCE client (no client secret).

    Redirects are never followed: the authorization code is read out of the 302 `Location`, which is
    why the CLI needs no local callback server.
    """

    def __init__(
        self,
        base_url: str,
        client_id: str,
        *,
        cookies: Mapping[str, str] | None = None,
        timeout: float = 20.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id
        self._http = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            follow_redirects=False,
            transport=transport if transport is not None else _make_transport(),
        )
        for name in COOKIE_NAMES:
            value = (cookies or {}).get(name)
            if value:
                self._http.cookies.set(name, value)  # domain-less: sent to the Verys host we talk to

    # ---- plumbing

    @property
    def cookies(self) -> dict[str, str]:
        """Just the Verys browser-session cookies, ready to store in the session file."""
        out = {}
        for name in COOKIE_NAMES:
            value = self._http.cookies.get(name)
            if value:
                out[name] = value
        return out

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> VerysClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- email code

    def send_code(self, email: str) -> None:
        """`POST /verification`: mail a 6-digit code. A 302 to `/register/` means no such identity."""
        resp = self._http.post("/verification", params={"email": email})
        if resp.status_code in _REDIRECT_CODES and "/register/" in resp.headers.get("location", ""):
            raise AuthError(f"no Verys account for {email}; register at {self.base_url}/register/ first")
        if resp.status_code not in (200, 201):
            raise _oauth_error(resp, "send_code")
        log.info("verification code mailed to %s", email)

    def verify_code(self, email: str, code: str) -> None:
        """`GET /verification`: swap the code for the `token`/`token_iv` cookies (404 = bad/expired)."""
        resp = self._http.get("/verification", params={"email": email, "code": code})
        if resp.status_code == 404:
            raise AuthError(f"verification failed: {_oauth_error(resp, 'verify_code').error}")
        if resp.status_code != 200:
            raise _oauth_error(resp, "verify_code")
        if set(self.cookies) != set(COOKIE_NAMES):
            raise AuthError("Verys accepted the code but set no session cookies")

    # ---- authorization code (PKCE S256)

    def authorize_code(self, redirect_uri: str) -> tuple[str, str]:
        """`GET /authorize` with our cookies, returning `(code, code_verifier)`.

        Four outcomes are possible and all of them are handled here: a 302 carrying `code`, a 302
        carrying `error`, the consent page (approved once, in-band), or the login page - which means
        the cookies are dead and only a human can fix it.
        """
        verifier = secrets.token_urlsafe(64)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode()
        state = secrets.token_urlsafe(32)
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "scope": SCOPE,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        resp = self._http.get("/authorize", params=params)
        if resp.status_code in _REDIRECT_CODES:
            return self._code_from(resp, redirect_uri, state, "authorize"), verifier
        if resp.status_code != 200:
            raise _oauth_error(resp, "authorize")

        consent = parse_consent(resp.text)
        if consent is None:
            if is_login_page(resp.text):
                raise SessionExpiredError("the stored Verys session cookies are no longer accepted")
            raise OAuthError(
                "unexpected_response",
                "/authorize returned HTML that is neither the consent nor the login form; "
                f"approve this client once at {self.base_url} in a browser",
                resp.status_code,
                "authorize",
            )
        log.info("approving the consent screen for client %s", self.client_id)
        posted = self._http.post(
            "/authorize/consent",
            data={
                "oauth2_session_id": consent.session_id,
                "csrf_token": consent.csrf_token,
                "consent_action": "approve",
            },
        )
        if posted.status_code not in _REDIRECT_CODES:
            raise _oauth_error(posted, "authorize/consent")
        return self._code_from(posted, redirect_uri, state, "authorize/consent"), verifier

    def _code_from(self, resp: httpx.Response, redirect_uri: str, state: str, where: str) -> str:
        """Read `code` out of a 302 `Location`, checking the redirect target and `state`."""
        location = resp.headers.get("location", "")
        if not location.startswith(redirect_uri):
            raise OAuthError("invalid_redirect", f"unexpected redirect to {location}", resp.status_code, where)
        q = parse_qs(urlparse(location).query)
        if error := q.get("error", [""])[0]:
            raise OAuthError(error, q.get("error_description", [""])[0], resp.status_code, where)
        code = q.get("code", [""])[0]
        if not code:
            raise OAuthError("invalid_response", f"no code in redirect to {location}", resp.status_code, where)
        if q.get("state", [""])[0] != state:
            raise OAuthError("invalid_state", "state in the redirect does not match the request", resp.status_code, where)
        return code

    # ---- grants (form-encoded POST /token; a public client sends client_id and no secret)

    def token_by_code(self, code: str, code_verifier: str, redirect_uri: str) -> Tokens:
        return self._token(
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "code_verifier": code_verifier,
            },
            "token_by_code",
        )

    def refresh(self, refresh_token: str) -> Tokens:
        """Refresh grant. Verys rotates: the presented token is revoked, a new one is returned."""
        return self._token({"grant_type": "refresh_token", "refresh_token": refresh_token}, "refresh")

    def exchange(self, subject_token: str, audience: str) -> Tokens:
        """RFC 8693 exchange of a Verys access token for one scoped to `audience`."""
        return self._token(
            {
                "grant_type": EXCHANGE_GRANT,
                "subject_token": subject_token,
                "subject_token_type": ACCESS_TOKEN_TYPE,
                "audience": audience,
            },
            "exchange",
        )

    def revoke(self, refresh_token: str) -> None:
        """`POST /token/revoke`, best effort. Never `/end-session`: that kills the browser SPA too."""
        try:
            resp = self._http.post("/token/revoke", data={"token": refresh_token, "client_id": self.client_id})
            if resp.status_code != 200:
                log.warning("token revocation returned %s", resp.status_code)
        except httpx.HTTPError as e:
            log.warning("token revocation failed: %s", e)

    def _token(self, data: dict[str, str], where: str) -> Tokens:
        resp = self._http.post("/token", data={"client_id": self.client_id, **data})
        if resp.status_code != 200:
            raise _oauth_error(resp, where)
        try:
            body = resp.json()
            access = body["access_token"]
        except (ValueError, KeyError, TypeError) as e:
            raise AuthError(f"malformed /token response for {where}: {e}") from e
        return Tokens(access, body.get("refresh_token"), body.get("id_token"), body.get("expires_in"))


# ---------------------------------------------------------------------------------- session file


def _parse_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _iso(dt: datetime | str) -> str:
    """Canonical UTC ISO-8601. Naive datetimes are UTC here, and an ISO string passes through."""
    if isinstance(dt, str):
        dt = _parse_iso(dt)
    return (dt if dt.tzinfo else dt.replace(tzinfo=UTC)).astimezone(UTC).isoformat(timespec="seconds")


@dataclass
class Session:
    """`.auth/session.json`: a long-lived credential. Cookies are kept, `id_token` is not."""

    verys_url: str
    client_id: str
    audience: str
    redirect_uri: str
    email: str
    sub: str
    refresh_token: str
    cookies: dict[str, str] = field(default_factory=dict)
    logged_in_at: str = ""
    obtained_at: str = ""
    roles: list[str] = field(default_factory=list)
    version: int = 1

    @classmethod
    def load(cls, path: Path) -> Session:
        """Read the session file; anything unusable is a `SessionExpiredError`, not a crash."""
        try:
            raw = json.loads(path.read_text())
        except FileNotFoundError:
            raise SessionExpiredError(f"no saved Verys session at {path}") from None
        except (OSError, ValueError) as e:
            raise SessionExpiredError(f"cannot read the Verys session at {path}: {e}") from e
        mode = path.stat().st_mode & 0o777
        if mode & 0o077:
            log.warning("%s is group/other-readable (mode %o); it is a credential, chmod 600 it", path, mode)
        known = {f.name for f in fields(cls)}
        try:
            return cls(**{k: v for k, v in raw.items() if k in known})
        except TypeError as e:
            raise SessionExpiredError(f"the Verys session at {path} is incomplete ({e})") from e

    def save(self, path: Path) -> None:
        """Atomically replace `path` with a 0600 file in a 0700 directory (tmp + fsync + replace)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with suppress(OSError):
            os.chmod(path.parent, 0o700)
        payload = {
            "version": self.version,
            "verys_url": self.verys_url,
            "client_id": self.client_id,
            "audience": self.audience,
            "redirect_uri": self.redirect_uri,
            "email": self.email,
            "sub": self.sub,
            "refresh_token": self.refresh_token,
            "cookies": self.cookies,
            "logged_in_at": self.logged_in_at,
            "obtained_at": self.obtained_at,
            "roles": self.roles,
        }
        tmp = path.with_name(path.name + ".tmp")
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as fh:
                json.dump(payload, fh, indent=1)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            with suppress(OSError):
                tmp.unlink()
            raise

    @property
    def refresh_age(self) -> timedelta:
        """How long ago the stored refresh token was issued (it rotates on every refresh)."""
        if not self.obtained_at:
            return timedelta(0)
        return datetime.now(UTC) - _parse_iso(self.obtained_at)


@contextmanager
def _flock(path: Path) -> Iterator[None]:
    """Serialise session-file updates between `chudp` processes (advisory lock, own file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        with suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


# --------------------------------------------------------------------------------- token provider


class TokenProvider(Protocol):
    """What `ChudApi` needs: a valid bearer token, and a way to say "that one was refused"."""

    def token(self) -> str: ...

    def invalidate(self) -> None: ...


class StaticTokenProvider:
    """A fixed token, for tests and for `CHUDP_TOKEN`-style manual use."""

    def __init__(self, token: str) -> None:
        self._token = token

    def token(self) -> str:
        return self._token

    def invalidate(self) -> None:
        return None


def _make_transport() -> httpx.BaseTransport | None:
    """Seam for tests: `None` means httpx' default (real network)."""
    return None


def _exchange_for_role(
    client: VerysClient, subject_token: str, audience: str, *, verys_url: str, email: str
) -> str:
    """Exchange, then assert the exchanged token actually carries `chud-money`."""
    try:
        exchanged = client.exchange(subject_token, audience).access_token
    except OAuthError as e:
        if e.error == "access_denied":
            raise AuthError(
                f"Verys has no consent record for client {audience}; approve it once in the browser at "
                f"{verys_url} (or re-run `chudp auth login --force`) and try again"
            ) from e
        raise
    roles = token_roles(exchanged)
    if REQUIRED_ROLE not in roles:
        raise MissingRoleError(email, verys_url, roles)
    return exchanged


def mint_session(
    client: VerysClient,
    *,
    audience: str,
    redirect_uri: str,
    verys_url: str,
    client_id: str,
    email: str,
    cookies: dict[str, str],
    logged_in_at: datetime | str,
) -> Session:
    """Turn live Verys cookies into a `Session`: authorize -> code grant -> one exchange to prove the role.

    Nothing is written; the caller decides where (and whether) to save.
    """
    code, verifier = client.authorize_code(redirect_uri)
    tokens = client.token_by_code(code, verifier, redirect_uri)
    if not tokens.refresh_token:
        raise AuthError("Verys issued no refresh token; unattended refresh would be impossible")
    exchanged = _exchange_for_role(client, tokens.access_token, audience, verys_url=verys_url, email=email)
    payload = claims(exchanged)
    now = datetime.now(UTC)
    return Session(
        verys_url=verys_url,
        client_id=client_id,
        audience=audience,
        redirect_uri=redirect_uri,
        email=email,
        sub=str(payload.get("sub", "")),
        refresh_token=tokens.refresh_token,
        cookies={k: v for k, v in cookies.items() if k in COOKIE_NAMES},
        logged_in_at=_iso(logged_in_at),
        obtained_at=_iso(now),
        roles=token_roles(exchanged),
    )


class VerysTokenProvider:
    """Caches one exchanged chud-money token and renews it from `auth_file` without a human.

    Thread-safe through a single lock (a token storm produces exactly one refresh and one exchange)
    and process-safe through `flock` on `<auth_file>.lock`, inside which the file is re-read because
    a sibling process may have rotated the refresh token.
    """

    def __init__(
        self,
        *,
        verys_url: str,
        client_id: str,
        audience: str,
        auth_file: Path,
        redirect_uri: str,
        early_s: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        now: Callable[[], float] = time.time,
    ) -> None:
        self.verys_url = verys_url.rstrip("/")
        self.client_id = client_id
        self.audience = audience
        self.auth_file = Path(auth_file)
        self.lock_file = self.auth_file.with_suffix(".lock")
        self.redirect_uri = redirect_uri
        self.early_s = early_s
        self._transport = transport if transport is not None else _make_transport()
        self._now = now
        self._lock = threading.Lock()
        self._token: str | None = None
        self._access: str | None = None  # Verys access token (aud = issuer), memory only
        self._session: Session | None = None

    # ---- TokenProvider

    def token(self) -> str:
        """A chud-money bearer token, renewed `early_s` before it expires."""
        with self._lock:
            if self._token and not expires_within(self._token, self.early_s, now=self._now):
                return self._token
            self._token = self._mint()
            return self._token

    def invalidate(self) -> None:
        """Drop the cached tokens; the next `token()` mints fresh ones (used after a 401)."""
        with self._lock:
            self._token = None
            self._access = None

    # ---- extras the CLI uses

    def claims(self) -> dict[str, Any]:
        """Claims of a live exchanged token (`aud`, `roles`, `exp`) - `chudp auth status`."""
        return claims(self.token())

    @property
    def email(self) -> str:
        """Email of the saved session."""
        if self._session is None:
            self._session = Session.load(self.auth_file)
        return self._session.email

    def close(self) -> None:
        """Forget the cached tokens (HTTP clients are per-call and already closed)."""
        self.invalidate()

    # ---- renewal

    def _mint(self) -> str:
        """Refresh the Verys access token if needed, then exchange it. Caller holds `self._lock`."""
        with _flock(self.lock_file):
            self._session = Session.load(self.auth_file)
            with VerysClient(
                self.verys_url, self.client_id, cookies=self._session.cookies, transport=self._transport
            ) as client:
                access = self._access
                if access is None or expires_within(access, self.early_s, now=self._now):
                    access = self._refresh_access(client)
                try:
                    exchanged = self._exchange(client, access)
                except OAuthError as e:
                    if e.error != "invalid_grant":
                        raise
                    # The subject token was refused (revoked, or rotated by a sibling): one retry.
                    log.info("token exchange rejected the subject token; refreshing it once")
                    access = self._refresh_access(client)
                    exchanged = self._exchange(client, access)
                self._access = access
                return exchanged

    def _exchange(self, client: VerysClient, access: str) -> str:
        session = self._require_session()
        return _exchange_for_role(client, access, self.audience, verys_url=self.verys_url, email=session.email)

    def _refresh_access(self, client: VerysClient) -> str:
        """Refresh grant, falling back to `/authorize` with the stored cookies."""
        session = self._require_session()
        try:
            tokens = client.refresh(session.refresh_token)
        except OAuthError as e:
            if e.error != "invalid_grant":
                raise
            log.info("stored refresh token is no longer valid; re-authorizing from the saved cookies")
            return self._reauthorize(client)
        self._store_refresh(tokens.refresh_token)
        return tokens.access_token

    def _reauthorize(self, client: VerysClient) -> str:
        """Mint a brand new refresh token from the stored browser cookies (no email round trip)."""
        session = self._require_session()
        if not session.cookies.get("token"):
            raise SessionExpiredError("the refresh token was rejected and no Verys session cookie is stored")
        code, verifier = client.authorize_code(self.redirect_uri)
        tokens = client.token_by_code(code, verifier, self.redirect_uri)
        if not tokens.refresh_token:
            raise SessionExpiredError("Verys re-authorized us but issued no refresh token")
        self._store_refresh(tokens.refresh_token)
        return tokens.access_token

    def _store_refresh(self, refresh_token: str | None) -> None:
        """Persist a rotated refresh token *before* it is used, so a crash cannot lose it."""
        session = self._require_session()
        if not refresh_token or refresh_token == session.refresh_token:
            return
        session.refresh_token = refresh_token
        session.obtained_at = _iso(datetime.now(UTC))
        session.save(self.auth_file)

    def _require_session(self) -> Session:
        if self._session is None:  # pragma: no cover - _mint always loads first
            self._session = Session.load(self.auth_file)
        return self._session


def token_provider(settings: Settings) -> VerysTokenProvider:
    """The single auth seam `cli.py` imports."""
    return VerysTokenProvider(
        verys_url=settings.verys_url,
        client_id=settings.verys_client_id,
        audience=settings.chud_money_client_id,
        auth_file=settings.auth_file,
        redirect_uri=settings.verys_redirect_uri,
    )
