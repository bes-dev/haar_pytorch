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
import torch
import torch.nn as nn


class HaarForward(nn.Module):
    """
    Performs a 2d DWT Forward decomposition of an image using Haar Wavelets
    """
    alpha = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a 2d DWT Forward decomposition of an image using Haar Wavelets

        Arguments:
            x (torch.Tensor): input tensor of shape [b, c, h, w]

        Returns:
            out (torch.Tensor): output tensor of shape [b, c * 4, h / 2, w / 2]
        """

        ll = self.alpha * (x[:,:,0::2,0::2] + x[:,:,0::2,1::2] + x[:,:,1::2,0::2] + x[:,:,1::2,1::2])
        lh = self.alpha * (x[:,:,0::2,0::2] + x[:,:,0::2,1::2] - x[:,:,1::2,0::2] - x[:,:,1::2,1::2])
        hl = self.alpha * (x[:,:,0::2,0::2] - x[:,:,0::2,1::2] + x[:,:,1::2,0::2] - x[:,:,1::2,1::2])
        hh = self.alpha * (x[:,:,0::2,0::2] - x[:,:,0::2,1::2] - x[:,:,1::2,0::2] + x[:,:,1::2,1::2])
        return torch.cat([ll,lh,hl,hh], axis=1)


class HaarInverse(nn.Module):
    """
    Performs a 2d DWT Inverse reconstruction of an image using Haar Wavelets
    """
    alpha = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a 2d DWT Inverse reconstruction of an image using Haar Wavelets

        Arguments:
            x (torch.Tensor): input tensor of shape [b, c, h, w]

        Returns:
            out (torch.Tensor): output tensor of shape [b, c / 4, h * 2, w * 2]
        """
        assert x.size(1) % 4 == 0, "The number of channels must be divisible by 4."
        size = [x.shape[0], x.shape[1] // 4, x.shape[2] * 2, x.shape[3] * 2]
        f = lambda i: x[:, size[1] * i : size[1] * (i + 1)]
        out = torch.zeros(size, dtype=x.dtype, device=x.device)
        out[:,:,0::2,0::2] = self.alpha * (f(0) + f(1) + f(2) + f(3))
        out[:,:,0::2,1::2] = self.alpha * (f(0) + f(1) - f(2) - f(3))
        out[:,:,1::2,0::2] = self.alpha * (f(0) - f(1) + f(2) - f(3))
        out[:,:,1::2,1::2] = self.alpha * (f(0) - f(1) - f(2) + f(3))
        return out
