# haar_pytorch: Pytorch implementation of forward and inverse Haar Wavelets 2D

A simple library that implements differentiable forward and inverse Haar Wavelets.

<p align="center">
  <img src="resources/haar.png"/>
</p>

## Install package

```bash
pip install haar_pytorch
```

## Install the latest version

```bash
pip install --upgrade git+https://github.com/bes-dev/haar_pytorch.git
```

## Example
```python
import torch
from haar_pytorch import HaarForward, HaarInverse

haar = HaarForward()
ihaar = HaarInverse()

img = torch.randn(5, 4, 64, 64)
wavelets = haar(img)
img_reconstructed = ihaar(wavelets)
```
