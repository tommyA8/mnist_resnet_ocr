import os
import base64
import hashlib
from functools import lru_cache
from typing import Optional
from dotenv import load_dotenv
load_dotenv(override=True)

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Allow overriding paths via environment for flexibility
PRIVATE_KEY_PATH = os.environ.get("PRIVATE_KEY_PATH", "secrets/private_key.pem")  # RSA private key (server)
PUBLIC_KEY_PATH = os.environ.get("PUBLIC_KEY_PATH", "secrets/public_key.pem")     # RSA public key (to share with clients)


def generate_rsa_keypair(private_path: str = PRIVATE_KEY_PATH, public_path: str = PUBLIC_KEY_PATH, bits: int = 2048) -> None:
    """Generate an RSA keypair if either key is missing (idempotent)."""
    if os.path.exists(private_path) and os.path.exists(public_path):
        return

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=bits)

    # write private key (PEM) - keep file permission restricted in production
    # ensure directories exist
    priv_dir = os.path.dirname(private_path)
    pub_dir = os.path.dirname(public_path)
    if priv_dir:
        os.makedirs(priv_dir, exist_ok=True)
    if pub_dir:
        os.makedirs(pub_dir, exist_ok=True)

    pem_priv = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,  # or PKCS8
        encryption_algorithm=serialization.NoEncryption(),
    )
    with open(private_path, "wb") as f:
        f.write(pem_priv)

    # write public key
    public_key = private_key.public_key()
    pem_pub = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    with open(public_path, "wb") as f:
        f.write(pem_pub)

    print(f"Generated RSA keypair: {private_path}, {public_path}")


@lru_cache(maxsize=1)
def get_private_key():
    """Load and cache the RSA private key. Call generate_rsa_keypair() beforehand if needed."""
    if not os.path.exists(PRIVATE_KEY_PATH):
        raise FileNotFoundError(f"Private key not found at {PRIVATE_KEY_PATH}. Ensure keys are generated before use.")
    with open(PRIVATE_KEY_PATH, "rb") as f:
        pem = f.read()
    return serialization.load_pem_private_key(pem, password=None)


def get_public_key_pem() -> str:
    """Return the public key PEM as a UTF-8 string."""
    if not os.path.exists(PUBLIC_KEY_PATH):
        raise FileNotFoundError(f"Public key not found at {PUBLIC_KEY_PATH}. Ensure keys are generated before use.")
    with open(PUBLIC_KEY_PATH, "rb") as f:
        return f.read().decode("utf-8")


def rsa_decrypt_base64(cipher_b64: str) -> bytes:
    """
    Decrypt base64-encoded RSA-OAEP ciphertext using server private key.
    Returns plaintext bytes.
    """
    try:
        ciphertext = base64.b64decode(cipher_b64)
        plaintext = get_private_key().decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        return plaintext
    except Exception as e:
        raise ValueError(f"RSA decryption failed: {str(e)}")


def aesgcm_decrypt_with_rsa_key(
    enc_key_b64: str,
    nonce_b64: str,
    ciphertext_b64: str,
    aad_b64: Optional[str] = None,
) -> bytes:
    """
    Hybrid decrypt: RSA-OAEP to unwrap AES key, then AES-GCM to decrypt ciphertext.
    All inputs are base64 strings. Returns plaintext bytes.
    """
    try:
        # 1) Unwrap AES key via RSA
        enc_key = base64.b64decode(enc_key_b64)
        key = get_private_key().decrypt(
            enc_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        if len(key) not in (16, 24, 32):
            raise ValueError("Invalid AES key size; expected 16/24/32 bytes")

        # 2) Decrypt with AES-GCM
        nonce = base64.b64decode(nonce_b64)
        ciphertext = base64.b64decode(ciphertext_b64)
        aad = base64.b64decode(aad_b64) if aad_b64 else None
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, aad)
        return plaintext
    except Exception as e:
        # For GCM, failures include auth errors
        raise ValueError(f"AES-GCM decryption failed: {str(e)}")


# def hash_national_id(nid_plaintext: str, salt: Optional[str] = None) -> str:
#     """
#     Hash the (decrypted) national id with a salt using SHA-256.
#     Returns hex digest. Do not store plaintext.
#     """
#     if salt is None:
#         # In production, prefer a per-record random salt stored alongside the hash (or use an HSM/KMS)
#         salt = os.environ.get("NID_SALT", "static-default-salt-change-me")
#     h = hashlib.sha256()
#     h.update(salt.encode("utf-8"))
#     h.update(nid_plaintext.encode("utf-8"))
#     return h.hexdigest()

