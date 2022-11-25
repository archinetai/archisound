
# ArchiSound

A collection of pre-trained audio models in PyTorch from [`audio-encoders-pytorch`](https://github.com/archinetai/audio-encoders-pytorch) and [`audio-diffusion-pytorch`](https://github.com/archinetai/audio-diffusion-pytorch).

## Install
```bash
pip install archisound
```

[![PyPI - Python Version](https://img.shields.io/pypi/v/archisound?style=flat&colorA=black&colorB=black)](https://pypi.org/project/archisound/)


## Autoencoders

### [`autoencoder1d-AT-v1`](https://huggingface.co/archinetai/autoencoder1d-AT-v1/tree/main)

```py
from archisound import ArchiSound

autoencoder = ArchiSound.from_pretrained('autoencoder1d-AT-v1')

x = torch.randn(1, 2, 2**18)    # [1, 2, 262144]
z = autoencoder.encode(x)       # [1, 32, 8192]
y = autoencoder.decode(z)       # [1, 2, 262144]
```

| Info  | |
| ------------- | ------------- |
| Input type | Audio (stereo @ 48kHz) |
| Number of parameters  | 20.7M  |
| Compression Factor | 2x |
| Downsampling Factor | 32x |
| Bottleneck Type | Tanh |
| Known Limitations | Slight blurriness in high frequency spectrogram reconstruction |


### [`dmae1d-ATC64-v1`](https://huggingface.co/archinetai/dmae1d-ATC64-v1/tree/main)
A diffusion based autoencoder with high compression ratio. Requires `audio_diffusion_pytorch==0.0.92`.

```py
from archisound import ArchiSound

autoencoder = ArchiSound.from_pretrained("dmae1d-ATC64-v1")

x = torch.randn(1, 2, 2**18)
z = autoencoder.encode(x) # [1, 32, 256]
y = autoencoder.decode(z, num_steps=20) # [1, 2, 262144]
```

| Info  | |
| ------------- | ------------- |
| Input type | Audio (stereo @ 48kHz) |
| Number of parameters  | 234.2M  |
| Compression Factor | 64x |
| Downsampling Factor | 1024x |
| Bottleneck Type | Tanh |
