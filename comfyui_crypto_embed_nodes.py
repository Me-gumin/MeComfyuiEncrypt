import os
import json
import zlib
import numpy as np
import torch
from PIL import Image, PngImagePlugin
from Crypto.Cipher import ChaCha20, AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Random import get_random_bytes
import folder_paths

# ========= PNG metadata helpers =========
def save_png_with_metadata(rgb_arr: np.ndarray, meta: dict, path: str):
    img = Image.fromarray(rgb_arr, mode="RGB")
    meta_json = json.dumps(meta).encode("utf-8")
    meta_compressed = zlib.compress(meta_json)

    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("crypto_meta", meta_compressed.hex())
    img.save(path, "PNG", pnginfo=pnginfo)


def load_png_with_metadata(path: str):
    img = Image.open(path).convert("RGB")
    rgb_arr = np.array(img)
    if "crypto_meta" not in img.info:
        raise ValueError("No crypto_meta in PNG metadata")
    meta_hex = img.info["crypto_meta"]
    meta_compressed = bytes.fromhex(meta_hex)
    meta_json = zlib.decompress(meta_compressed)
    meta = json.loads(meta_json.decode("utf-8"))
    return rgb_arr, meta


# ========= ChaCha20 =========
def encrypt_chacha20(arr: np.ndarray, passphrase: str, nonce_bytes: int = 12):
    h, w, c = arr.shape
    key = PBKDF2(passphrase, b"comfyui-salt", dkLen=32)
    nonce = get_random_bytes(nonce_bytes)
    cipher = ChaCha20.new(key=key, nonce=nonce)
    enc = cipher.encrypt(arr.tobytes())
    rgb = np.frombuffer(enc, dtype=np.uint8).reshape(h, w, c)
    meta = {"algo": "ChaCha20", "nonce": nonce.hex(), "shape": [h, w, c]}
    return rgb, meta


def decrypt_chacha20(cipher_bytes: bytes, passphrase: str, meta: dict, shape):
    key = PBKDF2(passphrase, b"comfyui-salt", dkLen=32)
    nonce = bytes.fromhex(meta["nonce"])
    cipher = ChaCha20.new(key=key, nonce=nonce)
    plain = cipher.decrypt(cipher_bytes)
    return np.frombuffer(plain, dtype=np.uint8).reshape(shape)


# ========= AES-CTR =========
def encrypt_aes_ctr(arr: np.ndarray, passphrase: str, nonce_bytes: int = 8):
    h, w, c = arr.shape
    key = PBKDF2(passphrase, b"comfyui-salt", dkLen=32)
    nonce = get_random_bytes(nonce_bytes)
    cipher = AES.new(key, AES.MODE_CTR, nonce=nonce)
    enc = cipher.encrypt(arr.tobytes())
    rgb = np.frombuffer(enc, dtype=np.uint8).reshape(h, w, c)
    meta = {"algo": "AES-CTR", "nonce": nonce.hex(), "shape": [h, w, c]}
    return rgb, meta


def decrypt_aes_ctr(cipher_bytes: bytes, passphrase: str, meta: dict, shape):
    key = PBKDF2(passphrase, b"comfyui-salt", dkLen=32)
    nonce = bytes.fromhex(meta["nonce"])
    cipher = AES.new(key, AES.MODE_CTR, nonce=nonce)
    plain = cipher.decrypt(cipher_bytes)
    return np.frombuffer(plain, dtype=np.uint8).reshape(shape)


# ========= ComfyUI Nodes =========
import os
import numpy as np
import torch
from PIL import Image
import folder_paths

class PNGEncryptNode:
    CATEGORY = "Crypto/PNG"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "encrypt"
    DISPLAY_NAME = "Encrypt Image"

    counter = 1  # 静态计数器

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "EncImage_"}),
                "passphrase": ("STRING", {"default": "changeme"}),
                "algo": (["ChaCha20", "AES-CTR"],),
            }
        }

    def encrypt(self, image, filename_prefix, passphrase, algo):
        workdir = folder_paths.get_output_directory()
        #os.makedirs(workdir, exist_ok=True)

        filename = f"{filename_prefix}_{PNGEncryptNode.counter:03}.png"
        PNGEncryptNode.counter += 1
        out_path = os.path.join(workdir, filename)

        arr = (image[0].cpu().numpy() * 255).astype(np.uint8)

        if algo == "ChaCha20":
            rgb, meta = encrypt_chacha20(arr, passphrase)
        else:  # AES-CTR
            rgb, meta = encrypt_aes_ctr(arr, passphrase)

        save_png_with_metadata(rgb, meta, out_path)

        saved_img = Image.open(out_path).convert("RGB")
        saved_np = np.array(saved_img).astype(np.float32) / 255.0
        out = torch.from_numpy(saved_np).unsqueeze(0)
        
        return (out,)

class PNGDecryptNode:
    CATEGORY = "Crypto/PNG"
    FUNCTION = "decrypt"
    RETURN_TYPES = ("IMAGE",)
    DISPLAY_NAME = "Decrypt PNG"

    @classmethod
    def INPUT_TYPES(cls):
        output_dir = folder_paths.get_output_directory()
        files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {
            "required": {
                "passphrase": ("STRING", {"default": "changeme"}),
                "algo": (["ChaCha20", "AES-CTR"],),
                "image": (sorted(files), {"image_upload": True})
            }
        }
    
    def decrypt(self, image, passphrase, algo):
        path = folder_paths.get_annotated_filepath(image)
        rgb_enc, meta = load_png_with_metadata(path)
        cipher_bytes = rgb_enc.tobytes()
        shape = tuple(meta["shape"])

        if meta["algo"] == "ChaCha20":
            if algo == "ChaCha20":
                plain = decrypt_chacha20(cipher_bytes, passphrase, meta, shape)
            else:
                raise ValueError("Wrong algorithm selection")
        elif meta["algo"] == "AES-CTR":
            if algo == "AES-CTR":
                plain = decrypt_aes_ctr(cipher_bytes, passphrase, meta, shape)
            else:
                raise ValueError("Wrong algorithm selection")
        else:
            raise ValueError(f"Unsupported algorithm: {meta['algo']}")

        out = torch.from_numpy(plain.astype(np.float32) / 255.0).unsqueeze(0)
        return (out,)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "PNGEncryptNode": PNGEncryptNode,
    "PNGDecryptNode": PNGDecryptNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PNGEncryptNode": "Encrypt Image",
    "PNGDecryptNode": "Decrypt PNG",
}
