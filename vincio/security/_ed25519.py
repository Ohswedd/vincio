"""Dependency-free Ed25519 signatures (RFC 8032), the self-certifying default.

The identity substrate needs an *asymmetric* signature so an agent's public key
can be published, a third party can verify a signature offline with the public
key alone, and a stable identifier can be *derived from* the key (a DID). HMAC is
symmetric and cannot do any of that. The audit chain already ships an
:class:`~vincio.security.audit.Ed25519Signer`, but it requires the optional
``cryptography`` package — so the dependency-free, offline-first default path
(the platform's contract) had no asymmetric signer.

This module closes that gap with a compact, **RFC 8032**-compliant Ed25519 in
pure Python: key generation from a 32-byte seed, deterministic signing, and
verification. It is interoperable byte-for-byte with the ``cryptography`` backend
(both implement the same standard), and :func:`signing_backend` automatically
prefers the native, constant-time C implementation when ``vincio[crypto]`` is
installed — so the pure-Python path is the *default*, never the *only*, and a
deployment that wants hardened, faster crypto installs one extra and gets it
without changing a line of calling code.

The pure-Python kernel is intended for **deterministic, offline** signing of
content-bound artifacts (identities, delegations, credentials, audit entries),
not high-throughput hot paths. It is *not* hardened against side-channel
(timing) attacks the way the native backend is; that is the explicit trade-off
the ``crypto`` extra exists to make, and the docstrings say so.

Everything here is private (module name leads with an underscore); the public
surface is :mod:`vincio.security.identity`.
"""

from __future__ import annotations

import hashlib
import secrets
from typing import Protocol

# ---------------------------------------------------------------------------
# Field / curve constants (Curve25519 / edwards25519, RFC 8032 §5.1)
# ---------------------------------------------------------------------------

_P = 2**255 - 19  # field prime
_L = 2**252 + 27742317777372353535851937790883648493  # group order
_D = (-121665 * pow(121666, _P - 2, _P)) % _P  # curve constant d
_I = pow(2, (_P - 1) // 4, _P)  # sqrt(-1) mod p


def _sha512(data: bytes) -> bytes:
    return hashlib.sha512(data).digest()


def _inv(x: int) -> int:
    """Multiplicative inverse mod p via Fermat's little theorem."""
    return pow(x, _P - 2, _P)


def _x_recover(y: int) -> int:
    """Recover the x-coordinate for a y on edwards25519 (RFC 8032 §5.1.3)."""
    xx = (y * y - 1) * _inv(_D * y * y + 1)
    x = pow(xx, (_P + 3) // 8, _P)
    if (x * x - xx) % _P != 0:
        x = (x * _I) % _P
    if x % 2 != 0:
        x = _P - x
    return x


# Base point B.
_BY = (4 * _inv(5)) % _P
_BX = _x_recover(_BY)
# Extended homogeneous coordinates (X, Y, Z, T) with Z=1, T=XY.
_B = (_BX % _P, _BY % _P, 1, (_BX * _BY) % _P)


def _edwards_add(
    p: tuple[int, int, int, int], q: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    """Twisted-Edwards point addition in extended coordinates (RFC 8032 §5.1.4)."""
    x1, y1, z1, t1 = p
    x2, y2, z2, t2 = q
    a = ((y1 - x1) * (y2 - x2)) % _P
    b = ((y1 + x1) * (y2 + x2)) % _P
    c = (t1 * 2 * _D * t2) % _P
    dd = (z1 * 2 * z2) % _P
    e = b - a
    f = dd - c
    g = dd + c
    h = b + a
    return ((e * f) % _P, (g * h) % _P, (f * g) % _P, (e * h) % _P)


def _scalar_mult(point: tuple[int, int, int, int], scalar: int) -> tuple[int, int, int, int]:
    """Double-and-add scalar multiplication; returns the identity for scalar 0."""
    result = (0, 1, 1, 0)  # neutral element
    addend = point
    while scalar > 0:
        if scalar & 1:
            result = _edwards_add(result, addend)
        addend = _edwards_add(addend, addend)
        scalar >>= 1
    return result


def _encode_point(point: tuple[int, int, int, int]) -> bytes:
    """Compress a point to 32 bytes (y with x's low bit in the high bit)."""
    x, y, z, _t = point
    zinv = _inv(z)
    x = (x * zinv) % _P
    y = (y * zinv) % _P
    encoded = bytearray((y % _P).to_bytes(32, "little"))
    encoded[31] |= (x & 1) << 7
    return bytes(encoded)


def _decode_point(data: bytes) -> tuple[int, int, int, int] | None:
    """Decompress 32 bytes to a curve point, or ``None`` if not on the curve."""
    if len(data) != 32:
        return None
    y = int.from_bytes(data, "little")
    sign = (y >> 255) & 1
    y &= (1 << 255) - 1
    if y >= _P:
        return None
    x = _x_recover(y)
    if x & 1 != sign:
        x = _P - x
    point = (x, y, 1, (x * y) % _P)
    if not _on_curve(point):
        return None
    return point


def _on_curve(point: tuple[int, int, int, int]) -> bool:
    """Check -x²+y² = 1+d·x²·y² (affine form via Z=1 points)."""
    x, y, z, _t = point
    zinv = _inv(z)
    x = (x * zinv) % _P
    y = (y * zinv) % _P
    return (-x * x + y * y - 1 - _D * x * x * y * y) % _P == 0


def _secret_scalar(h: bytes) -> int:
    """Clamp the lower 32 bytes of the seed hash into the secret scalar."""
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8  # clear low 3 bits
    a |= 1 << 254  # set the high bit
    return a


def public_key_from_seed(seed: bytes) -> bytes:
    """Derive the 32-byte Ed25519 public key from a 32-byte seed (private key)."""
    if len(seed) != 32:
        raise ValueError("Ed25519 seed must be exactly 32 bytes")
    h = _sha512(seed)
    a = _secret_scalar(h)
    return _encode_point(_scalar_mult(_B, a))


def sign(seed: bytes, message: bytes) -> bytes:
    """Deterministically sign ``message`` with the 32-byte ``seed``; 64-byte signature."""
    if len(seed) != 32:
        raise ValueError("Ed25519 seed must be exactly 32 bytes")
    h = _sha512(seed)
    a = _secret_scalar(h)
    prefix = h[32:]
    public = _encode_point(_scalar_mult(_B, a))
    r = int.from_bytes(_sha512(prefix + message), "little") % _L
    big_r = _encode_point(_scalar_mult(_B, r))
    k = int.from_bytes(_sha512(big_r + public + message), "little") % _L
    s = (r + k * a) % _L
    return big_r + s.to_bytes(32, "little")


def verify(public: bytes, message: bytes, signature: bytes) -> bool:
    """Verify a 64-byte Ed25519 signature against ``public`` and ``message``."""
    if len(signature) != 64 or len(public) != 32:
        return False
    big_r = signature[:32]
    s = int.from_bytes(signature[32:], "little")
    if s >= _L:
        return False
    point_a = _decode_point(public)
    point_r = _decode_point(big_r)
    if point_a is None or point_r is None:
        return False
    k = int.from_bytes(_sha512(big_r + public + message), "little") % _L
    # Check [s]B == R + [k]A
    left = _scalar_mult(_B, s)
    right = _edwards_add(point_r, _scalar_mult(point_a, k))
    return _encode_point(left) == _encode_point(right)


def generate_seed() -> bytes:
    """Return a fresh cryptographically-random 32-byte seed (private key)."""
    return secrets.token_bytes(32)


# ---------------------------------------------------------------------------
# Backend selection — pure-Python default, native ``cryptography`` when present
# ---------------------------------------------------------------------------


class _Ed25519Backend(Protocol):
    """The minimal seed-based Ed25519 surface both backends implement."""

    name: str

    def public_key_from_seed(self, seed: bytes) -> bytes: ...

    def sign(self, seed: bytes, message: bytes) -> bytes: ...

    def verify(self, public: bytes, message: bytes, signature: bytes) -> bool: ...


class _PureBackend:
    """The dependency-free, RFC 8032 reference backend (this module)."""

    name = "pure-python"

    def public_key_from_seed(self, seed: bytes) -> bytes:
        return public_key_from_seed(seed)

    def sign(self, seed: bytes, message: bytes) -> bytes:
        return sign(seed, message)

    def verify(self, public: bytes, message: bytes, signature: bytes) -> bool:
        return verify(public, message, signature)


class _NativeBackend:
    """The native ``cryptography`` (libsodium/OpenSSL) backend — constant-time."""

    name = "cryptography"

    def __init__(self) -> None:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # noqa: F401
            Ed25519PrivateKey,
        )

    def public_key_from_seed(self, seed: bytes) -> bytes:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        sk = Ed25519PrivateKey.from_private_bytes(seed)
        return sk.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )

    def sign(self, seed: bytes, message: bytes) -> bytes:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        return Ed25519PrivateKey.from_private_bytes(seed).sign(message)

    def verify(self, public: bytes, message: bytes, signature: bytes) -> bool:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        try:
            Ed25519PublicKey.from_public_bytes(public).verify(signature, message)
            return True
        except (InvalidSignature, ValueError):
            return False


_BACKEND: _Ed25519Backend | None = None


def signing_backend() -> _Ed25519Backend:
    """Return the active Ed25519 backend (native if available, else pure-Python).

    Cached after the first call. The two backends produce byte-identical RFC 8032
    signatures, so an artifact signed under one verifies under the other — the
    ``crypto`` extra is a drop-in acceleration, never a format change.
    """
    global _BACKEND
    if _BACKEND is None:
        try:
            _BACKEND = _NativeBackend()
        except Exception:  # pragma: no cover - exercised only without the extra
            _BACKEND = _PureBackend()
    return _BACKEND


def backend_name() -> str:
    """Name of the active backend (``"cryptography"`` or ``"pure-python"``)."""
    return signing_backend().name
