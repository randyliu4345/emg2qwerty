# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import torch
import torchaudio


TTransformIn = TypeVar("TTransformIn")
TTransformOut = TypeVar("TTransformOut")
Transform = Callable[[TTransformIn], TTransformOut]


@dataclass
class ToTensor:
    """Extracts the specified ``fields`` from a numpy structured array
    and stacks them into a ``torch.Tensor``.

    Following TNC convention as a default, the returned tensor is of shape
    (time, field/batch, electrode_channel).

    Args:
        fields (list): List of field names to be extracted from the passed in
            structured numpy ndarray.
        stack_dim (int): The new dimension to insert while stacking
            ``fields``. (default: 1)
    """

    fields: Sequence[str] = ("emg_left", "emg_right")
    stack_dim: int = 1

    def __call__(self, data: np.ndarray) -> torch.Tensor:
        return torch.stack(
            [torch.as_tensor(data[f]) for f in self.fields], dim=self.stack_dim
        )


@dataclass
class Lambda:
    """Applies a custom lambda function as a transform.

    Args:
        lambd (lambda): Lambda to wrap within.
    """

    lambd: Transform[Any, Any]

    def __call__(self, data: Any) -> Any:
        return self.lambd(data)


@dataclass
class ForEach:
    """Applies the provided ``transform`` over each item of a batch
    independently. By default, assumes the input is of shape (T, N, ...).

    Args:
        transform (Callable): The transform to apply to each batch item of
            the input tensor.
        batch_dim (int): The bach dimension, i.e., the dim along which to
            unstack/unbind the input tensor prior to mapping over
            ``transform`` and restacking. (default: 1)
    """

    transform: Transform[torch.Tensor, torch.Tensor]
    batch_dim: int = 1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [self.transform(t) for t in tensor.unbind(self.batch_dim)],
            dim=self.batch_dim,
        )


@dataclass
class Compose:
    """Compose a chain of transforms.

    Args:
        transforms (list): List of transforms to compose.
    """

    transforms: Sequence[Transform[Any, Any]]

    def __call__(self, data: Any) -> Any:
        for transform in self.transforms:
            data = transform(data)
        return data


@dataclass
class RandomBandRotation:
    """Applies band rotation augmentation by shifting the electrode channels
    by an offset value randomly chosen from ``offsets``. By default, assumes
    the input is of shape (..., C).

    NOTE: If the input is 3D with batch dim (TNC), then this transform
    applies band rotation for all items in the batch with the same offset.
    To apply different rotations each batch item, use the ``ForEach`` wrapper.

    Args:
        offsets (list): List of integers denoting the offsets by which the
            electrodes are allowed to be shift. A random offset from this
            list is chosen for each application of the transform.
        channel_dim (int): The electrode channel dimension. (default: -1)
    """

    offsets: Sequence[int] = (-1, 0, 1)
    channel_dim: int = -1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        offset = np.random.choice(self.offsets) if len(self.offsets) > 0 else 0
        return tensor.roll(offset, dims=self.channel_dim)


@dataclass
class TemporalAlignmentJitter:
    """Applies a temporal jittering augmentation that randomly jitters the
    alignment of left and right EMG data by up to ``max_offset`` timesteps.
    The input must be of shape (T, ...).

    Args:
        max_offset (int): The maximum amount of alignment jittering in terms
            of number of timesteps.
        stack_dim (int): The dimension along which the left and right data
            are stacked. See ``ToTensor()``. (default: 1)
    """

    max_offset: int
    stack_dim: int = 1

    def __post_init__(self) -> None:
        assert self.max_offset >= 0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.shape[self.stack_dim] == 2
        left, right = tensor.unbind(self.stack_dim)

        offset = np.random.randint(-self.max_offset, self.max_offset + 1)
        if offset > 0:
            left = left[offset:]
            right = right[:-offset]
        if offset < 0:
            left = left[:offset]
            right = right[-offset:]

        return torch.stack([left, right], dim=self.stack_dim)


@dataclass
class LogSpectrogram:
    """Creates log10-scaled spectrogram from an EMG signal. In the case of
    multi-channeled signal, the channels are treated independently.
    The input must be of shape (T, ...) and the returned spectrogram
    is of shape (T, ..., freq).

    Args:
        n_fft (int): Size of FFT, creates n_fft // 2 + 1 frequency bins.
            (default: 64)
        hop_length (int): Number of samples to stride between consecutive
            STFT windows. (default: 16)
    """

    n_fft: int = 64
    hop_length: int = 16

    def __post_init__(self) -> None:
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            normalized=True,
            # Disable centering of FFT windows to avoid padding inconsistencies
            # between train and test (due to differing window lengths), as well
            # as to be more faithful to real-time/streaming execution.
            center=False,
        )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor.movedim(0, -1)  # (T, ..., C) -> (..., C, T)
        spec = self.spectrogram(x)  # (..., C, freq, T)
        logspec = torch.log10(spec + 1e-6)  # (..., C, freq, T)
        return logspec.movedim(-1, 0)  # (T, ..., C, freq)


@dataclass
class SpecAugment:
    """Applies time and frequency masking as per the paper
    "SpecAugment: A Simple Data Augmentation Method for Automatic Speech
    Recognition, Park et al" (https://arxiv.org/abs/1904.08779).

    Args:
        n_time_masks (int): Maximum number of time masks to apply,
            uniformly sampled from 0. (default: 0)
        time_mask_param (int): Maximum length of each time mask,
            uniformly sampled from 0. (default: 0)
        iid_time_masks (int): Whether to apply different time masks to
            each band/channel (default: True)
        n_freq_masks (int): Maximum number of frequency masks to apply,
            uniformly sampled from 0. (default: 0)
        freq_mask_param (int): Maximum length of each frequency mask,
            uniformly sampled from 0. (default: 0)
        iid_freq_masks (int): Whether to apply different frequency masks to
            each band/channel (default: True)
        mask_value (float): Value to assign to the masked columns (default: 0.)
    """

    n_time_masks: int = 0
    time_mask_param: int = 0
    iid_time_masks: bool = True
    n_freq_masks: int = 0
    freq_mask_param: int = 0
    iid_freq_masks: bool = True
    mask_value: float = 0.0

    def __post_init__(self) -> None:
        self.time_mask = torchaudio.transforms.TimeMasking(
            self.time_mask_param, iid_masks=self.iid_time_masks
        )
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            self.freq_mask_param, iid_masks=self.iid_freq_masks
        )

    def __call__(self, specgram: torch.Tensor) -> torch.Tensor:
        # (T, ..., C, freq) -> (..., C, freq, T)
        x = specgram.movedim(0, -1)

        # Time masks
        n_t_masks = np.random.randint(self.n_time_masks + 1)
        for _ in range(n_t_masks):
            x = self.time_mask(x, mask_value=self.mask_value)

        # Frequency masks
        n_f_masks = np.random.randint(self.n_freq_masks + 1)
        for _ in range(n_f_masks):
            x = self.freq_mask(x, mask_value=self.mask_value)

        # (..., C, freq, T) -> (T, ..., C, freq)
        return x.movedim(-1, 0)


@dataclass
class GaussianNoiseJitter:
    """Adds small Gaussian noise jittering to the input tensor as a data
    augmentation technique. The noise is sampled from a zero-mean Gaussian
    distribution with the specified standard deviation.

    Args:
        std (float): Standard deviation of the Gaussian noise to add.
            (default: 0.01)
    """

    std: float = 0.01

    def __post_init__(self) -> None:
        assert self.std >= 0, "std must be non-negative"

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std
        return tensor + noise


@dataclass
class TimeWarp:
    """Applies time warping augmentation by locally stretching or compressing
    the time axis of the signal using PyTorch-native operations. This mimics
    natural timing variations in keystroke patterns.

    Time warping generates a smooth, random warp curve by sampling positive
    increments between knot points (guaranteeing strict monotonicity), then
    resamples the signal along this warped time axis using vectorized piecewise
    linear interpolation and PyTorch gather. The output has the same length as
    the input, but with locally stretched/compressed regions.

    This implementation is fully GPU-compatible, using only PyTorch operations
    with no Python loops over time or knot segments.

    Args:
        sigma (float): Controls the randomness/aggressiveness of the warp.
            Higher values produce more extreme local stretching/compression.
            Typical values: 0.05 (subtle), 0.1-0.2 (moderate), 0.3+ (aggressive).
            (default: 0.1)
        knots (int): Number of internal knot points (excluding endpoints) for
            the warp curve. More knots allow more complex warps. (default: 4)
    """

    sigma: float = 0.1
    knots: int = 4

    def __post_init__(self) -> None:
        assert self.sigma >= 0, "sigma must be non-negative"
        assert self.knots >= 1, "knots must be at least 1"

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor: Input tensor of shape (T, ...) where T is the time dimension.

        Returns:
            Warped tensor of the same shape as input.
        """
        T = tensor.shape[0]
        if T <= 1:
            return tensor

        device = tensor.device
        dtype = tensor.dtype

        # --- Build monotonic warp path via positive increments ---
        # Sample K+2 positive gaps (including endpoints), then cumsum + rescale.
        # This guarantees strict monotonicity by construction, avoiding the
        # fragile "cumulative max" approach.
        num_knots = self.knots + 2  # includes start and end
        gaps = torch.abs(torch.randn(num_knots, device=device, dtype=dtype)) * self.sigma + (1.0 / num_knots)
        knot_warps = torch.cumsum(gaps, dim=0)
        knot_warps = knot_warps / knot_warps[-1] * (T - 1)  # rescale to [0, T-1]
        knot_warps[0] = 0.0
        knot_warps[-1] = float(T - 1)

        # Evenly-spaced source knot locations (fixed)
        knot_indices = torch.linspace(0, T - 1, num_knots, device=device, dtype=dtype)

        # --- Vectorized piecewise linear interpolation ---
        # For each of the T output time steps, find which knot segment it falls
        # in, then linearly interpolate the corresponding warped index.
        time_grid = torch.arange(T, device=device, dtype=dtype)  # (T,)

        # Segment indices: how many knot boundaries each t has passed
        t_expand = time_grid.unsqueeze(1)                        # (T, 1)
        ki = knot_indices.unsqueeze(0)                           # (1, K)
        seg = (t_expand >= ki).sum(dim=1) - 1                    # (T,)
        seg = seg.clamp(0, num_knots - 2)

        x0 = knot_indices[seg]    # (T,)
        x1 = knot_indices[seg + 1]
        y0 = knot_warps[seg]
        y1 = knot_warps[seg + 1]

        t_frac = (time_grid - x0) / (x1 - x0 + 1e-8)
        warped_indices = (y0 + t_frac * (y1 - y0)).clamp(0.0, float(T - 1))  # (T,)

        # --- Resample via linear interpolation ---
        idx_low = warped_indices.floor().long()                  # (T,)
        idx_high = (idx_low + 1).clamp(0, T - 1)                # (T,)

        # alpha broadcasts over all trailing dims regardless of tensor rank
        alpha = (warped_indices - idx_low.float())
        for _ in range(tensor.ndim - 1):
            alpha = alpha.unsqueeze(-1)                          # (T, 1, 1, ...)

        lower = tensor[idx_low]                                  # (T, ...)
        upper = tensor[idx_high]                                 # (T, ...)
        warped = lower * (1.0 - alpha) + upper * alpha

        return warped.to(dtype=dtype)