import torch.nn as nn
from transformers import AutoModel

REVISION = {
    "autoencoder1d-AT-v1": "57b6cde1969208d10fdd3e813708c1abe49f25c1",
    "dmae1d-ATC64-v1": "07885065867977af43b460bb9c1422bdc90c29a0",
    "dmae1d-ATC64-v2": "3ffeea68d4c069777055fce9ac77bbb67eec1d68",
    "diff-text-M-v1": "679c4550e0bc49074778238eb92588ccd5a60ff2",
}


class ArchiSound:
    @staticmethod
    def from_pretrained(name: str = "", **kwargs) -> nn.Module:
        default_kwargs = dict(revision=REVISION[name])
        return AutoModel.from_pretrained(
            f"archinetai/{name}", trust_remote_code=True, **{**default_kwargs, **kwargs}
        )
