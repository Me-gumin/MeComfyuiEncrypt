"""
ComfyUI nodes: Chaos-based Image Encrypt / Decrypt (IMAGE in / IMAGE out)

Usage:
- ChaosEncrypt: input a ComfyUI IMAGE (tensor), set `key` and select `mode`.
- ChaosDecrypt: input encrypted IMAGE and same `key`/`mode`, get decrypted IMAGE.

WARNING: This is a pedagogical / experimental chaotic encryption demo.
It is NOT cryptographically secure.
"""

import torch

# ------------------- Chaotic Keystream Generator -------------------
def logistic_map_keystream(seed: float, size: int, r: float = 3.99, skip: int = 100) -> torch.ByteTensor:
    if isinstance(seed, int):
        seed = (seed % 2**31) / float(2**31 - 1)
    seed = float(seed)
    if seed <= 0 or seed >= 1:
        seed = (seed % 1.0)
        if seed == 0:
            seed = 0.3141592653
    x = seed
    out = torch.empty(size, dtype=torch.uint8)
    for _ in range(skip):
        x = r * x * (1.0 - x)
    i = 0
    while i < size:
        x = r * x * (1.0 - x)
        val = int((x * 2**32) % 256)
        out[i] = val
        i += 1
    return out

# ------------------- Encryption / Decryption -------------------
def chaotic_encrypt_tensor(img: torch.Tensor, seed: float, mode: str = 'xor') -> torch.Tensor:
    b, h, w, c = img.shape
    flat = (img * 255.0).round().to(torch.uint8).view(-1)
    ks = logistic_map_keystream(seed, flat.numel())
    if mode == 'xor':
        enc = torch.bitwise_xor(flat, ks)
    else:  # addmod
        enc = ((flat.to(torch.int16) + ks.to(torch.int16)) % 256).to(torch.uint8)
    out = enc.view(b, h, w, c).to(torch.float32) / 255.0
    return out

def chaotic_decrypt_tensor(img: torch.Tensor, seed: float, mode: str = 'xor') -> torch.Tensor:
    b, h, w, c = img.shape
    flat = (img * 255.0).round().to(torch.uint8).view(-1)
    ks = logistic_map_keystream(seed, flat.numel())
    if mode == 'xor':
        dec = torch.bitwise_xor(flat, ks)
    else:  # addmod
        dec = ((flat.to(torch.int16) - ks.to(torch.int16)) % 256).to(torch.uint8)
    out = dec.view(b, h, w, c).to(torch.float32) / 255.0
    return out

# ------------------- ComfyUI Node Classes -------------------
class ChaosEncrypt:
    CATEGORY = "Crypto/Chaos"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "encrypt"
    DISPLAY_NAME = "Chaos Encrypt"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "key": ("INT", {"default": 123456, "min": 0}),
                "mode": (["xor", "addmod"], {"default": "xor"}),  # 改为下拉选项
            }
        }

    def encrypt(self, image, key=123456, mode="xor"):
        seed = ((int(key) & 0x7fffffff) / float(2**31 - 1)) * 0.999999
        out = chaotic_encrypt_tensor(image, seed, mode=mode)
        return (out,)

class ChaosDecrypt:
    CATEGORY = "Crypto/Chaos"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decrypt"
    DISPLAY_NAME = "Chaos Decrypt"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "key": ("INT", {"default": 123456, "min": 0}),
                "mode": (["xor", "addmod"], {"default": "xor"}),  # 改为下拉选项
            }
        }

    def decrypt(self, image, key=123456, mode="xor"):
        seed = ((int(key) & 0x7fffffff) / float(2**31 - 1)) * 0.999999
        out = chaotic_decrypt_tensor(image, seed, mode=mode)
        return (out,)
