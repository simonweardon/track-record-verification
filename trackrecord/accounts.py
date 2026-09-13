"""Accounts, sessions and private per-user storage.  Stdlib only.

Users live in USERDATA/users.json (PBKDF2-SHA256 password hashes, 200k rounds);
sessions are HMAC-signed tokens in an HttpOnly cookie; every user's uploads and
results sit under USERDATA/<uid>/ and are served only to that user.

USERDATA defaults to <repo>/userdata (gitignored); on Railway point USERDATA_DIR
at a mounted volume so accounts survive deploys.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USERDATA = Path(os.environ.get("USERDATA_DIR", ROOT / "userdata"))
_LOCK = threading.Lock()
SESSION_DAYS = 30
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _users_path() -> Path:
    USERDATA.mkdir(parents=True, exist_ok=True)
    return USERDATA / "users.json"


def _load() -> dict:
    p = _users_path()
    return json.loads(p.read_text()) if p.exists() else {"users": {}}


def _save(d: dict) -> None:
    tmp = _users_path().with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1)); tmp.replace(_users_path())


def _secret() -> bytes:
    env = os.environ.get("SESSION_SECRET")
    if env:
        return env.encode()
    p = USERDATA / "session.secret"
    USERDATA.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text(secrets.token_hex(32)); os.chmod(p, 0o600)
    return p.read_text().strip().encode()


def _hash(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
    return base64.b64encode(salt).decode() + "$" + base64.b64encode(dk).decode()


def _verify(password: str, stored: str) -> bool:
    try:
        salt_b64, _ = stored.split("$", 1)
        return hmac.compare_digest(_hash(password, base64.b64decode(salt_b64)), stored)
    except Exception:
        return False


def create_user(email: str, password: str) -> tuple[str | None, str | None]:
    """Returns (uid, error)."""
    email = email.strip().lower()
    if not EMAIL_RE.match(email):
        return None, "That doesn't look like an email address."
    if len(password) < 10:
        return None, "Use a password of at least 10 characters."
    with _LOCK:
        d = _load()
        if any(u["email"] == email for u in d["users"].values()):
            return None, "An account with that email already exists — sign in instead."
        uid = secrets.token_hex(8)
        d["users"][uid] = dict(email=email, pw=_hash(password), created=time.time())
        _save(d)
    (USERDATA / uid).mkdir(parents=True, exist_ok=True)
    return uid, None


GUEST_TTL = 24 * 3600


def create_guest() -> str:
    """A temporary identity: no email, no password; purged after GUEST_TTL."""
    with _LOCK:
        d = _load()
        uid = "g" + secrets.token_hex(8)
        d["users"][uid] = dict(email="", guest=True, created=time.time())
        _save(d)
    (USERDATA / uid).mkdir(parents=True, exist_ok=True)
    return uid


def is_guest(uid: str) -> bool:
    return bool(_load()["users"].get(uid, {}).get("guest"))


def purge_guests(ttl: int = GUEST_TTL) -> int:
    """Remove guest identities (and their files) older than ttl.  Called opportunistically."""
    import shutil
    n = 0
    with _LOCK:
        d = _load(); now = time.time()
        for uid, u in list(d["users"].items()):
            if u.get("guest") and now - u.get("created", now) > ttl:
                shutil.rmtree(USERDATA / uid, ignore_errors=True); del d["users"][uid]; n += 1
        if n:
            _save(d)
    return n


def adopt_guest(guest_uid: str, uid: str) -> int:
    """Move a guest's records into a real account (on sign-up from a guest session)."""
    import shutil
    src, dst = USERDATA / guest_uid, user_dir(uid); n = 0
    if src.exists() and guest_uid != uid:
        for d in src.iterdir():
            if d.is_dir() and (d / "meta.json").exists():
                target = dst / d.name; k = 2
                while target.exists():
                    target = dst / f"{d.name}-{k}"; k += 1
                shutil.move(str(d), str(target)); n += 1
        shutil.rmtree(src, ignore_errors=True)
        with _LOCK:
            d = _load(); d["users"].pop(guest_uid, None); _save(d)
    return n


def authenticate(email: str, password: str) -> str | None:
    email = email.strip().lower()
    d = _load()
    for uid, u in d["users"].items():
        if u["email"] == email:
            return uid if _verify(password, u["pw"]) else None
    time.sleep(0.3)          # same cost whether or not the email exists
    return None


def user_email(uid: str) -> str:
    return _load()["users"].get(uid, {}).get("email", "")


# ---- sessions -------------------------------------------------------------------

def issue_token(uid: str) -> str:
    exp = int(time.time()) + SESSION_DAYS * 86400
    msg = f"{uid}.{exp}".encode()
    sig = hmac.new(_secret(), msg, hashlib.sha256).hexdigest()
    return f"{uid}.{exp}.{sig}"


def verify_token(token: str | None) -> str | None:
    if not token:
        return None
    try:
        uid, exp, sig = token.split(".")
        if int(exp) < time.time():
            return None
        good = hmac.new(_secret(), f"{uid}.{exp}".encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(good, sig):
            return None
        return uid if uid in _load()["users"] else None
    except Exception:
        return None


def cookie_header(token: str, secure: bool, session_only: bool = False) -> str:
    return (f"trsession={token}; Path=/; HttpOnly; SameSite=Lax" + ("" if session_only else f"; Max-Age={SESSION_DAYS * 86400}")
            + ("; Secure" if secure else ""))


def clear_cookie_header(secure: bool) -> str:
    return "trsession=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0" + ("; Secure" if secure else "")


def user_dir(uid: str) -> Path:
    p = USERDATA / uid
    p.mkdir(parents=True, exist_ok=True)
    return p


SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9\-]{0,40}$")


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")[:40]
    return s or "record"


def list_records(uid: str) -> list[dict]:
    out = []
    base = user_dir(uid)
    for d in sorted(base.iterdir()):
        m = d / "meta.json"
        if d.is_dir() and m.exists():
            try:
                meta = json.loads(m.read_text())
                meta["built"] = (d / "out" / "dashboard.html").exists()
                out.append(meta)
            except Exception:
                pass
    return out
