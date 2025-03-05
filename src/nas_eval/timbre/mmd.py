"""
Timbre Transfer MMD Evaluation
"""

from typing import List

import torch
import torchaudio


class MMD:
    """
    Unbiased squared MMD
    """

    def __init__(self, biased=False):
        super(MMD, self).__init__()
        self.biased = biased

    def guassian_kernel(self, source, target, comparing_self=False, bandwidth=None):
        l2 = torch.cdist(
            source, target, p=2, compute_mode="donot_use_mm_for_euclid_dist"
        )
        # Remove diagonal elements if comparing self and this is an unbiased MMD
        if comparing_self and not self.biased:
            n = l2.shape[0]
            l2 = l2[~torch.eye(n, dtype=bool)].reshape(n, n - 1)

        if bandwidth is None:
            bandwidth = torch.median(l2)

        kernel_val = torch.exp(-l2 / bandwidth)
        return kernel_val

    def __call__(self, source, target):
        total = torch.cat([source, target], dim=0)
        l2 = torch.cdist(total, total, p=2, compute_mode="donot_use_mm_for_euclid_dist")
        bandwidth = torch.median(l2)

        XX = self.guassian_kernel(
            source, source, comparing_self=True, bandwidth=bandwidth
        ).mean()
        YY = self.guassian_kernel(
            target, target, comparing_self=True, bandwidth=bandwidth
        ).mean()
        XY = self.guassian_kernel(
            source, target, comparing_self=False, bandwidth=bandwidth
        ).mean()

        distance = XX + YY - 2 * XY
        return distance


def compute_mfcc_features(
    audio: List[torch.Tensor], sr: int, texture_window: int = 40
) -> torch.Tensor:
    melkwargs = {
        "n_fft": 2048,
        "win_length": 2048,
        "hop_length": 512,
        "n_mels": 128,
    }
    mfcc = torchaudio.transforms.MFCC(sample_rate=sr, melkwargs=melkwargs)
    mfcc_features = []
    for waveform in audio:
        # MFCCs
        feature = mfcc(waveform)[..., 1:13, :]

        # Texture window
        if texture_window > 1:
            feature = torch.nn.functional.avg_pool2d(
                feature, (1, texture_window), stride=(1, texture_window // 2)
            )

        mfcc_features.append(feature[0].T)

    return torch.vstack(mfcc_features)
