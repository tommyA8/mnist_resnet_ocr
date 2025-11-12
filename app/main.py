import os
import base64
from io import BytesIO
from typing import List

import torch
import torchvision
import torchvision.transforms as transforms
from fastapi import FastAPI, HTTPException
from PIL import Image

from app.models.predict import PredictRequest, PredictResponse
from app.utils.encryption import (
    generate_rsa_keypair,
    get_public_key_pem,
    hash_national_id,
    rsa_decrypt_base64,
)

MODEL_PATH = os.environ.get("MODEL_PATH", "checkpoints/mnist_resnet18_epoch_3.pth")  # path to saved model weights
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path: str = MODEL_PATH) -> torch.nn.Module:
    # create architecture
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=10)

    # load weights (map to device)
    if os.path.exists(path):
        state = torch.load(path, map_location=DEVICE)
        # allow non-strict to be tolerant if minor mismatches occur
        model.load_state_dict(state, strict=False)
    else:
        print(f"Warning: model weights not found at {path}. Using random weights.")
    
    model.to(DEVICE)
    model.eval()
    return model

app = FastAPI(title="MNIST-ResNet Prediction API (with field encryption)")


@app.on_event("startup")
def on_startup() -> None:
    """Generate keys if needed and load model and transforms once."""
    # Ensure RSA keys exist
    generate_rsa_keypair()

    # Load trained model (ResNet18 for MNIST)
    app.state.model = load_model()

    # Preprocessing (same as training)
    app.state.preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet input size
        transforms.Grayscale(num_output_channels=3),  # convert 1 → 3 channels
        transforms.ToTensor(),
        # 3-channel mean/std to match 3 output channels from Grayscale
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

# Simple health check
@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}

# Endpoint to retrieve public key (clients use this to encrypt national_id)
@app.get("/public_key")
def get_public_key():
    # return PEM as string
    return {"public_key_pem": get_public_key_pem()}

# Prediction endpoint
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    # decrypt national id
    try:
        nid_plain = rsa_decrypt_base64(req.encrypted_nid).decode("utf-8")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid encrypted_nid: {str(e)}")

    # hash the national id for storage/audit (do NOT log plaintext)
    hashed_nid = hash_national_id(nid_plain)

    # decode image
    try:
        image_b64 = req.image_b64
        # Support data URLs (e.g., "data:image/png;base64,....")
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

    # preprocess -> tensor
    input_tensor = app.state.preprocess(image).unsqueeze(0).to(DEVICE)  # shape [1, C, H, W]

    # inference
    with torch.no_grad():
        outputs = app.state.model(input_tensor)  # logits [1, 10]
        probs: List[float] = (
            torch.nn.functional.softmax(outputs, dim=1).cpu().numpy().tolist()[0]
        )
        pred_class = int(torch.argmax(outputs, dim=1).cpu().item())

    # 6) respond (no plaintext nid returned)
    resp = PredictResponse(
        predicted_class=pred_class,
        probabilities=probs,
        hashed_nid=hashed_nid,
        note="national id received (decrypted server-side) and hashed for audit/storage"
    )
    return resp