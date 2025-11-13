from pydantic import BaseModel
from typing import Optional, List


class PredictRequest(BaseModel):
    enc_key_b64: Optional[str] = None  # RSA-OAEP encrypted AES key (base64)
    nonce_b64: Optional[str] = None    # AES-GCM nonce (base64)
    ciphertext_b64: Optional[str] = None  # AES-GCM ciphertext (base64). Tag is expected to be appended per AESGCM API.
    aad_b64: Optional[str] = None      # Optional AAD for AES-GCM (base64)

class PredictResponse(BaseModel):
    predicted_class: int
    probabilities: List[float]