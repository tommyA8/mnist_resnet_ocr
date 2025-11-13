from __future__ import annotations
import os
import base64
from io import BytesIO
import torch
import torchvision
import torchvision.transforms as transforms
from fastapi import FastAPI, HTTPException
from PIL import Image

from app.models.predict import PredictRequest, PredictResponse
from app.utils.encryption import (
    generate_rsa_keypair,
    get_public_key_pem,
    rsa_decrypt_base64,
    aesgcm_decrypt_with_rsa_key,
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

app = FastAPI(title="MNIST-ResNet Prediction API")


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
        transforms.Normalize((0.5,), (0.5,))
    ])




@app.get("/public_key")
def get_public_key():
    return {"public_key_pem": get_public_key_pem()}



@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    # Prefer AES-GCM hybrid if provided; fallback to legacy RSA transport
    if req.enc_key_b64 and req.nonce_b64 and req.ciphertext_b64:
        try:
            plaintext = aesgcm_decrypt_with_rsa_key(
                enc_key_b64=req.enc_key_b64,
                nonce_b64=req.nonce_b64,
                ciphertext_b64=req.ciphertext_b64,
                aad_b64=req.aad_b64,
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"AES-GCM decrypt failed: {str(e)}")
    elif req.encrypted_image_b64:
        try:
            plaintext = rsa_decrypt_base64(req.encrypted_image_b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"RSA decrypt failed: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Missing encryption fields: provide AES-GCM fields (enc_key_b64, nonce_b64, ciphertext_b64) or legacy encrypted_image_b64")

    # plaintext is expected to be base64-encoded image bytes
    try:
        image_bytes = base64.b64decode(plaintext)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

    # preprocess -> tensor
    input_tensor = app.state.preprocess(image).unsqueeze(0).to(DEVICE)  # shape [1, C, H, W]

    # inference
    with torch.no_grad():
        outputs = app.state.model(input_tensor)  # logits [1, 10]
        probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy().tolist()[0]
        pred_class = int(torch.argmax(outputs, dim=1).cpu().item())

    return PredictResponse(
        predicted_class=pred_class,
        probabilities=probs
    )

