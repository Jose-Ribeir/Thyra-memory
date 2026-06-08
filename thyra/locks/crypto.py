"""Locked memory encryption using Fernet with PBKDF2 key derivation."""

from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken


class MemoryLocker:
    def _derive_key(self, token: str, memory_id: str, agent_id: str) -> bytes:
        salt = (agent_id + ":" + memory_id).encode()
        raw_key = hashlib.pbkdf2_hmac(
            "sha256",
            token.encode(),
            salt,
            iterations=100_000,
        )
        return base64.urlsafe_b64encode(raw_key)

    def encrypt(self, content: str, token: str, memory_id: str, agent_id: str) -> str:
        key = self._derive_key(token, memory_id, agent_id)
        return "ENCRYPTED:" + Fernet(key).encrypt(content.encode()).decode()

    def decrypt(self, encrypted: str, token: str, memory_id: str, agent_id: str) -> str:
        if not encrypted.startswith("ENCRYPTED:"):
            raise ValueError("Content is not in encrypted format")
        key = self._derive_key(token, memory_id, agent_id)
        try:
            return Fernet(key).decrypt(encrypted[10:].encode()).decode()
        except InvalidToken as exc:
            raise ValueError("Invalid token — cannot decrypt this memory") from exc
