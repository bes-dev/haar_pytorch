"""
Copyright 2021 Sergei Belousov

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import pytest
import torch
from haar_pytorch import HaarForward, HaarInverse


HAVE_GPU = torch.cuda.is_available()
if HAVE_GPU:
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


def test_almost_equal():
    haar = HaarForward()
    ihaar = HaarInverse()
    img = torch.randn(5, 4, 64, 64).to(DEVICE)
    freq = haar(img)
    img_rec = ihaar(freq)
    diff = (img_rec - img).pow(2).mean().item()
    assert diff == pytest.approx(0.0)
