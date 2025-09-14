from .comfyui_chaos_encrypt_node import ChaosEncrypt, ChaosDecrypt
from .comfyui_crypto_embed_nodes import (
	PNGEncryptNode,PNGDecryptNode
)

NODE_CLASS_MAPPINGS = {
    "ChaosEncrypt": ChaosEncrypt,
    "ChaosDecrypt": ChaosDecrypt,
    "PNGEncryptNode": PNGEncryptNode,
    "PNGDecryptNode": PNGDecryptNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChaosEncrypt": "Chaos Encrypt",
    "ChaosDecrypt": "Chaos Decrypt",
    "PNGEncryptNode": "Encrypt Image",
    "PNGDecryptNode": "Decrypt PNG",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]