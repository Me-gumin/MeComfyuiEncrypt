"""
ComfyUI nodes: ChaCha20 / AES-CTR Image Encrypt-Decrypt (RGB encrypt + Alpha metadata)

加密逻辑：
- RGB 通道存放 ciphertext（和原图同样大小）。
- Alpha 通道嵌入 metadata（nonce、算法、shape）。
- 解密时先从 Alpha 取 metadata，再解密 RGB。

"""

import torch
import numpy as np
import json, zlib
from Crypto.Cipher import ChaCha20, AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Random import get_random_bytes

# --------- Metadata helpers ---------
def embed_metadata_in_alpha(rgb_arr: np.ndarray, meta: dict) -> np.ndarray:
    """把 metadata 压缩后写到 Alpha 通道"""
    h, w, _ = rgb_arr.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb_arr

    meta_json = json.dumps(meta).encode("utf-8")
    meta_compressed = zlib.compress(meta_json)
    meta_bytes = np.frombuffer(meta_compressed, dtype=np.uint8)

    capacity = h * w
    if len(meta_bytes) > capacity:
        raise ValueError(
            f"Metadata ({len(meta_bytes)} bytes) too large for alpha capacity {capacity}"
        )
    alpha = np.zeros(capacity, dtype=np.uint8)
    alpha[: len(meta_bytes)] = meta_bytes
    rgba[:, :, 3] = alpha.reshape(h, w)
    return rgba

def extract_metadata_from_alpha(rgba: np.ndarray) -> dict:
    """从 Alpha 通道解出 metadata"""
    h, w, _ = rgba.shape
    alpha = rgba[:, :, 3].flatten()
    # 去掉末尾多余的 0
    raw = bytes(alpha).rstrip(b"\x00")
    meta_json = zlib.decompress(raw)
    return json.loads(meta_json.decode("utf-8"))

# --------- ChaCha20 ---------
def encrypt_chacha20(arr: np.ndarray, passphrase: str, nonce_bytes: int = 12):
    h, w, c = arr.shape
    key = PBKDF2(passphrase, b"comfyui-salt", dkLen=32)
    nonce = get_random_bytes(nonce_bytes)
    cipher = ChaCha20.new(key=key, nonce=nonce)
    data = arr.tobytes()
    enc = cipher.encrypt(data)
    rgb = np.frombuffer(enc, dtype=np.uint8).reshape(h, w, c)
    meta = {"algo": "ChaCha20", "nonce": nonce.hex(), "shape": [h, w, c]}
    return rgb, meta

def decrypt_chacha20(cipher_bytes: bytes, passphrase: str, meta: dict, shape):
    key = PBKDF2(passphrase, b"comfyui-salt", dkLen=32)
    nonce = bytes.fromhex(meta["nonce"])
    cipher = ChaCha20.new(key=key, nonce=nonce)
    plain = cipher.decrypt(cipher_bytes)
    return np.frombuffer(plain, dtype=np.uint8).reshape(shape)

# --------- AES-CTR ---------
def encrypt_aes_ctr(arr: np.ndarray, passphrase: str, nonce_bytes: int = 8):
    h, w, c = arr.shape
    key = PBKDF2(passphrase, b"comfyui-salt", dkLen=32)
    nonce = get_random_bytes(nonce_bytes)
    cipher = AES.new(key, AES.MODE_CTR, nonce=nonce)
    data = arr.tobytes()
    enc = cipher.encrypt(data)
    rgb = np.frombuffer(enc, dtype=np.uint8).reshape(h, w, c)
    meta = {"algo": "AES-CTR", "nonce": nonce.hex(), "shape": [h, w, c]}
    return rgb, meta

def decrypt_aes_ctr(cipher_bytes: bytes, passphrase: str, meta: dict, shape):
    key = PBKDF2(passphrase, b"comfyui-salt", dkLen=32)
    nonce = bytes.fromhex(meta["nonce"])
    cipher = AES.new(key, AES.MODE_CTR, nonce=nonce)
    plain = cipher.decrypt(cipher_bytes)
    return np.frombuffer(plain, dtype=np.uint8).reshape(shape)

# --------- ComfyUI Nodes ---------
class ChaCha20EncryptEmbed:
    CATEGORY = "Crypto/Embed"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "encrypt"
    DISPLAY_NAME = "ChaCha20 Encrypt (Embed)"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "passphrase": ("STRING", {"default": "changeme", "multiline": False}),
                "nonce_bytes": ("INT", {"default": 12, "min": 8, "max": 16}),
            }
        }

    def encrypt(self, image, passphrase, nonce_bytes=12):
        arr = (image[0].cpu().numpy() * 255).astype(np.uint8)
        rgb, meta = encrypt_chacha20(arr, passphrase, nonce_bytes)
        rgba = embed_metadata_in_alpha(rgb, meta)
        out = torch.from_numpy(rgba.astype(np.float32) / 255.0).unsqueeze(0)
        return (out,)

class ChaCha20DecryptEmbed:
    CATEGORY = "Crypto/Embed"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decrypt"
    DISPLAY_NAME = "ChaCha20 Decrypt (Embed)"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "passphrase": ("STRING", {"default": "changeme"})}}

    def decrypt(self, image, passphrase):
        arr = (image[0].cpu().numpy() * 255).astype(np.uint8)
        meta = extract_metadata_from_alpha(arr)
        h, w, c = meta["shape"]
        cipher_bytes = arr[:, :, :3].tobytes()
        if meta["algo"] == "ChaCha20":
            plain = decrypt_chacha20(cipher_bytes, passphrase, meta, (h, w, c))
        else:
            raise ValueError(f"Unsupported algo {meta['algo']}")
        out = torch.from_numpy(plain.astype(np.float32) / 255.0).unsqueeze(0)
        return (out,)

class AESCTREncryptEmbed:
    CATEGORY = "Crypto/Embed"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "encrypt"
    DISPLAY_NAME = "AES-CTR Encrypt (Embed)"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "passphrase": ("STRING", {"default": "changeme"}),
                "nonce_bytes": ("INT", {"default": 8, "min": 4, "max": 16}),
            }
        }

    def encrypt(self, image, passphrase, nonce_bytes=8):
        arr = (image[0].cpu().numpy() * 255).astype(np.uint8)
        rgb, meta = encrypt_aes_ctr(arr, passphrase, nonce_bytes)
        rgba = embed_metadata_in_alpha(rgb, meta)
        out = torch.from_numpy(rgba.astype(np.float32) / 255.0).unsqueeze(0)
        return (out,)

class AESCTRDecryptEmbed:
    CATEGORY = "Crypto/Embed"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decrypt"
    DISPLAY_NAME = "AES-CTR Decrypt (Embed)"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "passphrase": ("STRING", {"default": "changeme"})}}

    def decrypt(self, image, passphrase):
        arr = (image[0].cpu().numpy() * 255).astype(np.uint8)
        meta = extract_metadata_from_alpha(arr)
        h, w, c = meta["shape"]
        cipher_bytes = arr[:, :, :3].tobytes()
        if meta["algo"] == "AES-CTR":
            plain = decrypt_aes_ctr(cipher_bytes, passphrase, meta, (h, w, c))
        else:
            raise ValueError(f"Unsupported algo {meta['algo']}")
        out = torch.from_numpy(plain.astype(np.float32) / 255.0).unsqueeze(0)
        return (out,)
