"""In-process Verys stand-in over `httpx.MockTransport`, mirroring the real routes closely enough
that `auth.py` cannot pass by accident.

What is modelled faithfully (see `verys/src/verys/routes/{oauth2,verification,session}.py`):
* `/verification` POST mails a code (302 to `/register/` for an unknown email), GET swaps it for the
  `token`/`token_iv` cookies - set without `Domain` or `Secure`, as httpx sees them in tests.
* `/authorize` validates the client, the registered `redirect_uri` (400 `Invalid redirect_uri`), the
  `openid` scope and PKCE `S256`, serves `login.html` when the cookies are not accepted, and the real
  consent markup on the first visit when `require_consent`.
* `/token` runs all three grants with real PKCE verification, single-use codes, refresh rotation
  (`invalid_grant` on reuse) and an exchange that refuses a subject token whose `aud` is not the
  issuer, or an identity with no consent record (`access_denied`).
* `/token/revoke` always answers 200 (RFC 7009).

Tokens are JWT-shaped `b64url(header).b64url(payload).sig` strings: unsigned on purpose, since the
client only base64-decodes them.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from urllib.parse import parse_qs, urlencode

import httpx

CLIENT_ID = "99e6d288-fab8-4e18-a4df-ada501e18fce"
AUDIENCE = CLIENT_ID
BASE_URL = "https://verys.test"
REDIRECT_URI = "https://chud-money.test/auth/callback"
EMAIL = "reis@luminal.com"
SUB = "0193e0a1-2b3c-4d5e-8f90-1234567890ab"
CODE = "123456"

LOGIN_HTML = """<!DOCTYPE html><html><body><div class="card"><h1>Sign In</h1>
<div id="step-email" class="step active"><form id="email-form">
<input type="email" id="email" name="email" required autofocus>
<button type="submit">Send verification code</button></form></div>
</body></html>"""

# Fields filled in as Jinja would; attribute order deliberately differs from the template so the
# scraper cannot depend on it.
CONSENT_HTML = """<!DOCTYPE html><html><head><title>Authorize Application</title></head><body>
<div class="card"><h1>Authorize Application</h1>
<p><span class="client-name">{client_name}</span> is requesting access to your account with the following permissions:</p>
<ul class="scopes">{scopes}</ul>
<form action="{issuer}/authorize/consent" method="POST">
<input value="{session_id}" type="hidden" name="oauth2_session_id">
<input name="csrf_token" value="{csrf}" type="hidden">
<div class="actions">
<button class="btn-deny" type="submit" name="consent_action" value="deny">Deny</button>
<button value="approve" type="submit" name="consent_action" class="btn-approve">Approve</button>
</div></form></div></body></html>"""


def _b64(obj: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()


def _form(request: httpx.Request) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(request.content.decode(), keep_blank_values=True).items()}


def _query(request: httpx.Request) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(request.url.query.decode(), keep_blank_values=True).items()}


def _s256(verifier: str) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode()


def _err(error: str, description: str = "", status: int = 400) -> httpx.Response:
    body = {"error": error}
    if description:
        body["error_description"] = description
    return httpx.Response(status, json=body)


def _redirect(redirect_uri: str, **params: str) -> httpx.Response:
    return httpx.Response(302, headers={"location": f"{redirect_uri}?{urlencode(params)}"})


class FakeVerys:
    """Stateful fake; `.transport` goes straight into `VerysClient(transport=...)`."""

    def __init__(
        self,
        *,
        base_url: str = BASE_URL,
        client_id: str = CLIENT_ID,
        audience: str = AUDIENCE,
        email: str = EMAIL,
        sub: str = SUB,
        roles: Iterable[str] = ("chud-money",),
        require_consent: bool = False,
        cookies_valid: bool = True,
        redirect_uris: Iterable[str] = (REDIRECT_URI,),
        clock: Callable[[], float] | None = None,
        ttl: int = 300,
    ) -> None:
        self.issuer = base_url.rstrip("/")
        self.client_id = client_id
        self.audience = audience
        self.email = email
        self.sub = sub
        self.roles = list(roles)
        self.require_consent = require_consent
        self.cookies_valid = cookies_valid
        self.redirect_uris = set(redirect_uris)
        self.clock = clock or time.time
        self.ttl = ttl
        self.code = CODE

        self.known_emails = {email}
        self.consents: set[str] = set() if require_consent else {email}
        self.pending_codes: dict[str, str] = {}          # email -> emailed code
        self.auth_codes: dict[str, dict] = {}            # code -> {challenge, redirect_uri, used}
        self.refresh_tokens: dict[str, dict] = {}        # token -> {revoked}
        self.sessions: dict[str, dict] = {}              # consent session id -> authorize params
        self.cookie_value = "enc-cookie"
        self.cookie_iv = "enc-iv"
        self.fail_next_exchange = False
        self.calls: Counter[str] = Counter()
        self.grants: Counter[str] = Counter()
        self.transport = httpx.MockTransport(self.handle)

    # ------------------------------------------------------------------ token minting

    def access_token(self, *, aud: str | None = None, roles: Iterable[str] | None = None, ttl: int | None = None) -> str:
        now = int(self.clock())
        payload = {
            "iss": self.issuer,
            "sub": self.sub,
            "aud": aud or self.issuer,
            "roles": list(self.roles if roles is None else roles),
            "iat": now,
            "exp": now + (self.ttl if ttl is None else ttl),
            "scopes": ["openid", "email", "profile"],
        }
        return f"{_b64({'alg': 'EdDSA', 'kid': 'fake'})}.{_b64(payload)}.sig"

    def _id_token(self) -> str:
        return self.access_token(aud=self.client_id)

    def _new_refresh(self) -> str:
        token = f"rt-{len(self.refresh_tokens) + 1}"
        self.refresh_tokens[token] = {"revoked": False}
        return token

    def _token_response(self, *, refresh: str, with_id: bool = True) -> httpx.Response:
        body = {
            "access_token": self.access_token(),
            "token_type": "Bearer",
            "expires_in": self.ttl,
            "refresh_token": refresh,
            "scope": "openid email profile",
        }
        if with_id:
            body["id_token"] = self._id_token()
        return httpx.Response(200, json=body, headers={"cache-control": "no-store"})

    # ------------------------------------------------------------------ routing

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls[path] += 1
        if path == "/verification":
            return self._send_code(request) if request.method == "POST" else self._verify_code(request)
        if path == "/authorize":
            return self._authorize(request)
        if path == "/authorize/consent":
            return self._consent(request)
        if path == "/token":
            return self._token(request)
        if path == "/token/revoke":
            form = _form(request)
            self.refresh_tokens.get(form.get("token", ""), {}).update(revoked=True)
            return httpx.Response(200, json={"message": "Token revoked."})
        return httpx.Response(404, json={"error": f"no route for {path}"})

    # ------------------------------------------------------------------ /verification

    def _send_code(self, request: httpx.Request) -> httpx.Response:
        email = _query(request).get("email", "")
        if email not in self.known_emails:
            return httpx.Response(302, headers={"location": f"/register/?{urlencode({'email': email})}"})
        self.pending_codes[email] = self.code
        return httpx.Response(201, json={"message": "Verification code sent."})

    def _verify_code(self, request: httpx.Request) -> httpx.Response:
        q = _query(request)
        email, code = q.get("email", ""), q.get("code", "")
        if self.pending_codes.get(email) != code:
            return _err("Invalid or expired code", status=404)
        del self.pending_codes[email]
        self.cookies_valid = True
        opts = "Path=/; HttpOnly; SameSite=lax"
        return httpx.Response(
            200,
            json={"message": "Email verified."},
            headers=[
                ("set-cookie", f"token={self.cookie_value}; {opts}"),
                ("set-cookie", f"token_iv={self.cookie_iv}; {opts}"),
            ],
        )

    # ------------------------------------------------------------------ /authorize

    def _cookies_ok(self, request: httpx.Request) -> bool:
        jar = {
            c.split("=", 1)[0].strip(): c.split("=", 1)[1]
            for c in request.headers.get("cookie", "").split(";")
            if "=" in c
        }
        return self.cookies_valid and jar.get("token") == self.cookie_value and jar.get("token_iv") == self.cookie_iv

    def _authorize(self, request: httpx.Request) -> httpx.Response:
        q = _query(request)
        for name in ("response_type", "client_id", "redirect_uri", "scope"):
            if not q.get(name):
                return _err(f"Missing required query parameter: {name}")
        if q["client_id"] != self.client_id:
            return _err("Invalid client_id")
        redirect_uri = q["redirect_uri"]
        if redirect_uri not in self.redirect_uris:
            return _err("Invalid redirect_uri")
        state = q.get("state", "")
        if q["response_type"] != "code":
            return _redirect(redirect_uri, error="unsupported_response_type", error_description="code only", state=state)
        if "openid" not in q["scope"].split():
            return _redirect(redirect_uri, error="invalid_scope", error_description="The 'openid' scope is required", state=state)
        if not q.get("code_challenge"):
            return _redirect(redirect_uri, error="invalid_request", error_description="Public clients must use PKCE", state=state)
        if q.get("code_challenge_method") != "S256":
            return _redirect(
                redirect_uri, error="invalid_request", error_description="Only S256 code_challenge_method is supported", state=state
            )
        if not self._cookies_ok(request):
            return httpx.Response(200, html=LOGIN_HTML)
        if self.email not in self.consents:
            session_id = f"sess-{len(self.sessions) + 1}"
            self.sessions[session_id] = {
                "csrf_token": f"csrf-{session_id}",
                "redirect_uri": redirect_uri,
                "state": state,
                "code_challenge": q["code_challenge"],
            }
            html = CONSENT_HTML.format(
                client_name="chud-money",
                scopes="".join(f"<li>{s}</li>" for s in q["scope"].split()),
                issuer=self.issuer,
                session_id=session_id,
                csrf=f"csrf-{session_id}",
            )
            return httpx.Response(200, html=html)
        return self._issue_code(redirect_uri, state, q["code_challenge"])

    def _issue_code(self, redirect_uri: str, state: str, challenge: str) -> httpx.Response:
        code = f"ac-{len(self.auth_codes) + 1}"
        self.auth_codes[code] = {"challenge": challenge, "redirect_uri": redirect_uri, "used": False}
        params = {"code": code}
        if state:
            params["state"] = state
        return _redirect(redirect_uri, **params)

    def _consent(self, request: httpx.Request) -> httpx.Response:
        form = _form(request)
        session = self.sessions.get(form.get("oauth2_session_id", ""))
        if session is None:
            return _err("Invalid or expired session")
        if form.get("csrf_token") != session["csrf_token"]:
            return _err("Invalid CSRF token", status=403)
        del self.sessions[form["oauth2_session_id"]]
        if form.get("consent_action") != "approve":
            return _redirect(
                session["redirect_uri"], error="access_denied", error_description="The user denied the authorization request",
                state=session["state"],
            )
        self.consents.add(self.email)
        return self._issue_code(session["redirect_uri"], session["state"], session["code_challenge"])

    # ------------------------------------------------------------------ /token

    def _token(self, request: httpx.Request) -> httpx.Response:
        form = _form(request)
        if form.get("client_id") != self.client_id:
            return httpx.Response(401, json={"error": "invalid_client", "error_description": "Client authentication failed"})
        grant = form.get("grant_type", "")
        self.grants[grant] += 1
        if grant == "authorization_code":
            return self._code_grant(form)
        if grant == "refresh_token":
            return self._refresh_grant(form)
        if grant == "urn:ietf:params:oauth:grant-type:token-exchange":
            return self._exchange_grant(form)
        return _err("unsupported_grant_type", f"unknown grant {grant!r}")

    def _code_grant(self, form: Mapping[str, str]) -> httpx.Response:
        entry = self.auth_codes.get(form.get("code", ""))
        if entry is None:
            return _err("invalid_grant", "Authorization code not found")
        if entry["used"]:
            # Replay: the real server revokes every refresh token of identity+client.
            for rt in self.refresh_tokens.values():
                rt["revoked"] = True
            return _err("invalid_grant", "Authorization code has already been used")
        if entry["redirect_uri"] != form.get("redirect_uri"):
            return _err("invalid_grant", "Redirect URI mismatch")
        verifier = form.get("code_verifier", "")
        if not verifier:
            return _err("invalid_request", "Code verifier is required")
        if _s256(verifier) != entry["challenge"]:
            return _err("invalid_grant", "Invalid code verifier")
        entry["used"] = True
        return self._token_response(refresh=self._new_refresh())

    def _refresh_grant(self, form: Mapping[str, str]) -> httpx.Response:
        token = form.get("refresh_token", "")
        entry = self.refresh_tokens.get(token)
        if entry is None:
            return _err("invalid_grant", "Refresh token not found")
        if entry["revoked"]:
            return _err("invalid_grant", "Refresh token is revoked or expired")
        entry["revoked"] = True  # rotation revokes only the presented token
        new = self._new_refresh()
        entry["replaced_by"] = new
        return self._token_response(refresh=new)

    def _exchange_grant(self, form: Mapping[str, str]) -> httpx.Response:
        subject = form.get("subject_token", "")
        if not subject:
            return _err("invalid_request", "subject_token is required")
        if form.get("subject_token_type") != "urn:ietf:params:oauth:token-type:access_token":
            return _err("invalid_request", "Only access_token subject_token_type is supported")
        audience = form.get("audience", "")
        if not audience:
            return _err("invalid_request", "audience is required")
        if audience != self.audience:
            return _err("invalid_target", "Audience is not a registered OAuth client")
        try:
            payload = json.loads(base64.urlsafe_b64decode(subject.split(".")[1] + "=="))
        except Exception:  # noqa: BLE001 - any unparseable token is simply invalid here
            return _err("invalid_grant", "Invalid or expired subject token")
        if payload.get("aud") != self.issuer or payload.get("exp", 0) < self.clock():
            return _err("invalid_grant", "Invalid or expired subject token")
        if self.fail_next_exchange:
            self.fail_next_exchange = False
            return _err("invalid_grant", "Invalid or expired subject token")
        if self.email not in self.consents:
            return _err("access_denied", "User has not consented to target client.", status=403)
        return httpx.Response(
            200,
            json={
                "access_token": self.access_token(aud=audience),
                "issued_token_type": "urn:ietf:params:oauth:token-type:access_token",
                "token_type": "Bearer",
                "expires_in": self.ttl,
            },
            headers={"cache-control": "no-store"},
        )
