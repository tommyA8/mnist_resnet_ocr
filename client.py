import argparse
import base64
import json
from io import BytesIO
from typing import Dict

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from PIL import Image
import torchvision


def fetch_public_key(api_url: str) -> str:
    resp = requests.get(f"{api_url}/public_key", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data["public_key_pem"]


def encrypt_nid(national_id: str, public_key_pem: str) -> str:
    public_key = serialization.load_pem_public_key(public_key_pem.encode("utf-8"))
    ciphertext = public_key.encrypt(
        national_id.encode("utf-8"),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return base64.b64encode(ciphertext).decode("utf-8")


def get_mnist_first_image_b64(data_root: str = "data") -> str:
    # torchvision will use raw files in data/MNIST/raw if present and build processed dataset
    ds = torchvision.datasets.MNIST(root=data_root, train=True, download=False)
    img, label = ds[0]
    if not isinstance(img, Image.Image):
        # just in case a transform was applied; convert to PIL
        img = torchvision.transforms.ToPILImage()(img)
    # Save as PNG to bytes
    buf = BytesIO()
    img.save(buf, format="PNG")
    b = buf.getvalue()
    return base64.b64encode(b).decode("utf-8")


def call_predict(api_url: str, encrypted_nid_b64: str, image_b64: str) -> Dict:
    payload = {
        "encrypted_nid": encrypted_nid_b64,
        "image_b64": image_b64,
        "metadata": {"note": "mnist first image"},
    }
    resp = requests.post(f"{api_url}/predict", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Client for MNIST /predict endpoint")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="Base URL of the API server")
    parser.add_argument("--nid", default="1234567890", help="National ID to encrypt and send")
    parser.add_argument("--data-root", default="data", help="Root directory containing MNIST data")
    args = parser.parse_args()

    print(f"Using API: {args.api_url}")
    # 1) Fetch server public key
    pub_pem = fetch_public_key(args.api_url)
    print("Fetched public key")

    # 2) Encrypt national id
    enc_nid = encrypt_nid(args.nid, pub_pem)
    print("Encrypted national ID")

    # 3) Load first MNIST image and base64 encode
    img_b64 = get_mnist_first_image_b64(args.data_root)
    print("Loaded MNIST first image")

    # 4) Call predict
    result = call_predict(args.api_url, enc_nid, img_b64)

    print("-- Prediction Result --")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
