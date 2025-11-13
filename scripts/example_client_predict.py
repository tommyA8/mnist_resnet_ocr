"""AES-GCM client for the /predict endpoint.

Encrypts the base64-encoded first MNIST image using AES-GCM with a random key,
wraps the AES key using the server's RSA public key (RSA-OAEP), and sends the
hybrid-encrypted payload to the server.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from io import BytesIO
from typing import Dict, Optional, Tuple

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from PIL import Image
import torchvision


def fetch_public_key(api_url: str) -> str:
    """Fetch the server's RSA public key (PEM)."""
    resp = requests.get(f"{api_url}/public_key", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data["public_key_pem"]


def rsa_encrypt_key(public_key_pem: str, key_bytes: bytes) -> str:
    """RSA-OAEP encrypt an AES key and return base64 string."""
    public_key = serialization.load_pem_public_key(public_key_pem.encode("utf-8"))
    ciphertext = public_key.encrypt(
        key_bytes,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return base64.b64encode(ciphertext).decode("utf-8")


def aesgcm_encrypt(key: bytes, plaintext: bytes, aad: Optional[bytes] = None) -> Tuple[str, str]:
    """AES-GCM encrypt returning (nonce_b64, ciphertext_b64)."""
    if len(key) not in (16, 24, 32):
        raise ValueError("AES key must be 16/24/32 bytes")
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ciphertext = aesgcm.encrypt(nonce, plaintext, aad)
    return base64.b64encode(nonce).decode("utf-8"), base64.b64encode(ciphertext).decode("utf-8")

def get_demo_image_b64() -> str:
    image = Image.open("assets/demo_mnist_first.png").convert("L")
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def call_predict_aesgcm(
    api_url: str,
    enc_key_b64: str,
    nonce_b64: str,
    ciphertext_b64: str,
    aad_b64: Optional[str] = None,
) -> Dict:
    """POST the AES-GCM hybrid payload to /predict and return JSON response."""
    payload = {
        "enc_key_b64": enc_key_b64,
        "nonce_b64": nonce_b64,
        "ciphertext_b64": ciphertext_b64,
    }
    print(payload)
    if aad_b64:
        payload["aad_b64"] = aad_b64
    resp = requests.post(f"{api_url}/predict", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="AES-GCM client for MNIST /predict endpoint")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="Base URL of the API server, e.g. http://127.0.0.1:8000")
    parser.add_argument("--data-root", default="data", help="Root directory containing MNIST data (expects data/MNIST/raw)")
    parser.add_argument("--key-bytes", type=int, default=32, choices=[16, 24, 32], help="AES key size in bytes (16, 24, 32)")
    parser.add_argument("--aad", default=None, help="Optional AAD string to bind into AES-GCM (will be UTF-8 encoded)")
    args = parser.parse_args()

    # 1) Fetch server public key
    pub_pem = fetch_public_key(args.api_url)

    # 2) Prepare plaintext: first MNIST image as base64 (utf-8 bytes)
    img_b64_bytes = get_demo_image_b64().encode("utf-8")

    # 3) AES-GCM encrypt
    key = os.urandom(args.key_bytes)
    aad_bytes = args.aad.encode("utf-8") if args.aad else None
    nonce_b64, ct_b64 = aesgcm_encrypt(key, img_b64_bytes, aad=aad_bytes)

    # 4) Wrap AES key with RSA
    enc_key_b64 = rsa_encrypt_key(pub_pem, key)

    # 5) Call predict
    aad_b64 = base64.b64encode(aad_bytes).decode("utf-8") if aad_bytes else None
    result = call_predict_aesgcm(args.api_url, enc_key_b64, nonce_b64, ct_b64, aad_b64=aad_b64)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
