from .comfyui_chaos_encrypt_node import ChaosEncrypt, ChaosDecrypt
from .comfyui_crypto_embed_nodes import (
    ChaCha20EncryptEmbed,
    ChaCha20DecryptEmbed,
    AESCTREncryptEmbed,
    AESCTRDecryptEmbed,
)

NODE_CLASS_MAPPINGS = {
    "ChaosEncrypt": ChaosEncrypt,
    "ChaosDecrypt": ChaosDecrypt,
    "ChaCha20EncryptEmbed": ChaCha20EncryptEmbed,
    "ChaCha20DecryptEmbed": ChaCha20DecryptEmbed,
    "AESCTREncryptEmbed": AESCTREncryptEmbed,
    "AESCTRDecryptEmbed": AESCTRDecryptEmbed,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChaosEncrypt": "Chaos Encrypt",
    "ChaosDecrypt": "Chaos Decrypt",
    "ChaCha20EncryptEmbed": "ChaCha20 Encrypt",
    "ChaCha20DecryptEmbed": "ChaCha20 Decrypt",
    "AESCTREncryptEmbed": "AES-CTR Encrypt",
    "AESCTRDecryptEmbed": "AES-CTR Decrypt",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]