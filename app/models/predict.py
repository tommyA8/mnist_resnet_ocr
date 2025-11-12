from pydantic import BaseModel, Field
from typing import Optional


class PredictRequest(BaseModel):
    # encrypted_nid must be base64 string of RSA-OAEP ciphertext encrypted with server public key
    encrypted_nid: str
    # image is expected to be base64-encoded image (png/jpg) in the request
    image_b64: str
    # optional metadata
    metadata: Optional[dict] = None

class PredictResponse(BaseModel):
    predicted_class: int
    probabilities: list
    hashed_nid: str
    note: Optional[str] = None