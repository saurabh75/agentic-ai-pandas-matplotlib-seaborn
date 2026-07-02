"""Resolve the current user_id from a reverse-proxy header (Option D).

Deploy behind nginx / Traefik / OAuth2-Proxy / Cloudflare Access that injects
an authenticated header (e.g. X-Forwarded-User, X-Auth-Request-Email,
Cf-Access-Authenticated-User-Email). Streamlit exposes request headers via
`st.context.headers` (Streamlit >= 1.37).

Fallback:
- If AUTH_ENABLED=false → single-user mode returning "default".
- If header missing and AUTH_ALLOW_ANON=true → uses a per-browser cookie UUID.
- Otherwise raises PermissionError (Streamlit shows a friendly login-required page).
"""
from __future__ import annotations
import os, re, uuid, hashlib
from dataclasses import dataclass

USER_ID_RE = re.compile(r"^[A-Za-z0-9_.\-@]{3,64}$")
SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_-]")

DEFAULT_HEADERS = [
    "X-Forwarded-User",
    "X-Forwarded-Email",
    "X-Auth-Request-User",
    "X-Auth-Request-Email",
    "Cf-Access-Authenticated-User-Email",
    "Remote-User",
]


@dataclass(frozen=True)
class UserContext:
    user_id: str           # raw identity (e.g. email)
    safe_id: str           # filesystem/collection-safe slug
    source: str            # "header:X-Forwarded-User" | "anon" | "default"

    @property
    def is_anonymous(self) -> bool:
        return self.source == "anon"


def _slugify(uid: str) -> str:
    s = SAFE_ID_RE.sub("_", uid)[:48]
    h = hashlib.sha1(uid.encode()).hexdigest()[:8]
    return f"{s}_{h}"


def _read_header(name: str) -> str | None:
    try:
        import streamlit as st
        headers = getattr(st.context, "headers", None)
        if headers is None:
            return None
        # Streamlit headers dict is case-insensitive
        return headers.get(name)
    except Exception:
        return None


def _anon_cookie_id() -> str:
    import streamlit as st
    key = "_rag_anon_uid"
    if key not in st.session_state:
        # persist across reruns; not across browser restarts (Streamlit limitation)
        st.session_state[key] = f"anon-{uuid.uuid4().hex[:12]}"
    return st.session_state[key]


def get_current_user() -> UserContext:
    if os.getenv("AUTH_ENABLED", "false").lower() != "true":
        return UserContext(user_id="default", safe_id="default", source="default")

    header_names = os.getenv("AUTH_USER_HEADER", "").strip()
    candidates = [h.strip() for h in header_names.split(",") if h.strip()] or DEFAULT_HEADERS

    for h in candidates:
        val = _read_header(h)
        if val and USER_ID_RE.match(val.strip()):
            uid = val.strip().lower()
            return UserContext(user_id=uid, safe_id=_slugify(uid), source=f"header:{h}")

    if os.getenv("AUTH_ALLOW_ANON", "false").lower() == "true":
        uid = _anon_cookie_id()
        return UserContext(user_id=uid, safe_id=_slugify(uid), source="anon")

    raise PermissionError(
        "No authenticated user header found. "
        f"Expected one of: {', '.join(candidates)}. "
        "Configure your reverse proxy (nginx / oauth2-proxy / Cloudflare Access) "
        "to inject the header, or set AUTH_ALLOW_ANON=true for testing."
    )
