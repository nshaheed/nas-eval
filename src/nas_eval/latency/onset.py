"""
Core funcionality for the numpy backend.
"""

from functools import partial
from typing import Callable
from typing import Literal
from typing import Union

import numpy as np
import torch
from numba import jit
from scipy import signal
from scipy.signal import find_peaks

try:
    import librosa
except ImportError:
    librosa = None

# import matplotlib.pyplot as plt


def get_onset_detection_function(
    name: Literal["amplitude", "energy", "spectral_flux", "librosa", "combined"],
    sample_rate: int,
):
    """
    Get an onset detection function
    """
    if name == "amplitude":
        return OnsetDetection(sample_rate)
    elif name == "energy":
        return PeakPickingOnsetDetection(
            partial(energy_based_onset_function, log=True, norm=True)
        )
    elif name == "spectral_flux":
        return PeakPickingOnsetDetection(
            partial(spectral_flux_onset_function, log=True, norm=True)
        )
    elif name == "librosa":
        return partial(librosa_onset_detection, sr=sample_rate)
    elif name == "combined":
        return CombinedOnsetDetection(
            [
                OnsetDetection(sample_rate),
                partial(librosa_onset_detection, sr=sample_rate),
            ]
        )
    else:
        raise ValueError("Invalid onset detection function")


class HighPassFilter:
    """
    Simple implementation of a high-pass filter
    """

    def __init__(
        self, sr: int, cutoff: float, q: float = 0.707, peak_gain: float = 0.0
    ):
        self.sr = sr
        self.cutoff = cutoff
        self.q = q
        self.peak_gain = peak_gain
        self._init_filter()

    def _init_filter(self):
        K = np.tan(np.pi * self.cutoff / self.sr)
        norm = 1 / (1 + K / self.q + K * K)
        self.a0 = 1 * norm
        self.a1 = -2 * self.a0
        self.a2 = self.a0
        self.b1 = 2 * (K * K - 1) * norm
        self.b2 = (1 - K / self.q + K * K) * norm

    def __call__(self, x: np.array):
        assert x.ndim == 2 and x.shape[0] == 1
        y = signal.lfilter(
            [self.a0, self.a1, self.a2], [1, self.b1, self.b2], x, axis=1, zi=None
        )
        return y


@jit(nopython=True)
def envelope_follower(x: np.array, up: float, down: float, initial: float = 0.0):
    y = np.zeros_like(x)
    y0 = initial
    for i in range(y.shape[-1]):
        if x[0, i] > y0:
            y0 = up * (x[0, i] - y0) + y0
        else:
            y0 = down * (x[0, i] - y0) + y0
        y[0, i] = y0
    return y


class EnvelopeFollower:
    def __init__(self, attack_samples: int, release_samples: int):
        self.up = 1.0 / attack_samples
        self.down = 1.0 / release_samples

    def __call__(self, x: np.array, initial: float = 0.0):
        assert x.ndim == 2 and x.shape[0] == 1
        return envelope_follower(x, self.up, self.down, initial=initial)


@jit(nopython=True)
def detect_onset(x: np.array, on_thresh: float, off_thresh: float, wait: int):
    debounce = -1
    onsets = []
    for i in range(1, x.shape[-1]):
        if x[0, i] >= on_thresh and x[0, i - 1] < on_thresh and debounce == -1:
            onsets.append(i)
            debounce = wait

        if debounce > 0:
            debounce -= 1

        if debounce == 0 and x[0, i] < off_thresh:
            debounce = -1

    return onsets


class OnsetDetection:
    def __init__(
        self,
        sr: int,
        on_thresh: float = 16.0,
        off_thresh: float = 4.6666,
        wait: int = 1323,
        min_db: float = -55.0,
        eps: float = 1e-8,
    ):
        self.env_fast = EnvelopeFollower(3.0, 383.0)
        self.env_slow = EnvelopeFollower(2205.0, 2205.0)
        self.high_pass = HighPassFilter(sr, 600.0)
        self.on_thresh = on_thresh
        self.off_thresh = off_thresh
        self.min_db = min_db
        self.wait = wait
        self.eps = eps

    def _onset_signal(self, x: np.array):
        # Filter
        x = self.high_pass(x)

        # Rectify, convert to dB, and set minimum value
        x = np.abs(x)
        x = 20 * np.log10(x + self.eps)
        x[x < self.min_db] = self.min_db

        # Calculate envelope
        env_fast = self.env_fast(x, initial=self.min_db)
        env_slow = self.env_slow(x, initial=self.min_db)
        diff = env_fast - env_slow

        return diff

    def __call__(self, x: Union[np.array, torch.Tensor]) -> int:
        assert x.ndim == 1, "Monophone audio only."
        x = x[None, :]
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()

        # Calculate envelope
        novelty = self._onset_signal(x)
        onsets = detect_onset(novelty, self.on_thresh, self.off_thresh, self.wait)

        return onsets, novelty[0]


class PeakPickingOnsetDetection:
    """
    Peak picking based onset detection
    """

    def __init__(self, novelty_function: Callable, threshold: float = 0.1):
        self.novelty_function = novelty_function
        self.threshold = threshold

    def __call__(self, x: Union[np.array, torch.Tensor]) -> int:
        novelty = self.novelty_function(signal)
        if isinstance(novelty, torch.Tensor):
            novelty = novelty.cpu().numpy()

        onset = find_onset(novelty, threshold=self.threshold)
        return onset, novelty


class CombinedOnsetDetection:
    def __init__(self, onset_functions: list):
        self.onset_functions = onset_functions

    def __call__(self, x: Union[np.array, torch.Tensor]) -> int:
        onsets = []
        envelopes = []
        for onset_function in self.onset_functions:
            onset, env = onset_function(x)
            onsets.append(onset)
            envelopes.append(env)

        # Select the onset with the minimum value
        argmin = -1
        for i, onset in enumerate(onsets):
            if len(onset) > 0:
                if argmin == -1 or onset[0] < onsets[argmin][0]:
                    argmin = i

        if argmin == -1:
            return None, None

        return onsets[argmin], envelopes[argmin]


class OffsetDetection:
    def __init__(self, threshold: float = 0.001):
        self.threshold = threshold

    def __call__(self, signal: Union[np.ndarray, torch.Tensor]) -> int:
        if isinstance(signal, np.ndarray):
            signal = torch.from_numpy(signal)
        envelope = spectral_flux_onset_function(
            signal, novelty=False, log=True, norm=True, offset=True
        )
        envelope = envelope.cpu().numpy()
        envelope = np.flip(envelope)
        offset = find_onset(envelope, threshold=0.01)
        return signal.shape[-1] - offset[0], np.flip(envelope)


def energy_based_onset_function(
    x: np.ndarray, log: bool = False, norm: bool = False
) -> np.ndarray:
    """
    Energy onset detection function
    """
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()

    # Compute the energy of the signal
    x = np.pad(x, (512, 512), mode="constant")
    x = np.convolve(x**2, np.hanning(1024) ** 2, mode="valid")
    x = np.diff(x)
    x[x < 0] = 0

    if log:
        x = np.log(1 + x)

    if norm:
        max_val = np.max(x)
        if max_val > 0:
            x = x / max_val

    return x


def spectral_flux_onset_function(
    x: torch.Tensor,
    log: bool = False,
    norm: bool = False,
    n_fft: int = 1024,
    hop_length: int = 64,
    novelty: bool = True,
    offset: bool = False,
) -> torch.Tensor:
    """
    Spectral Flux novelty function
    """
    # x_pad = torch.nn.functional.pad(x, (n_fft - hop_length, 0), mode="constant")
    X = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.hann_window(n_fft, device=x.device),
        return_complex=True,
        pad_mode="constant",
        normalized=False,
        onesided=True,
        center=True,
    )

    # L2-norm on the rectified difference of the magnitude spectrogram
    flux = torch.diff(torch.abs(X), dim=-1)

    # Rectify the flux
    if offset:
        flux = (flux - torch.abs(flux)) / 2
    else:
        flux = (flux + torch.abs(flux)) / 2

    flux = torch.square(flux)
    flux = torch.sum(flux, dim=0)

    # Novelty function
    if novelty:
        flux = torch.diff(flux, dim=-1)
        flux[flux < 0.0] = 0.0

    if log:
        flux = torch.log(1.0 + flux)

    if norm:
        max_val = torch.max(flux)
        if max_val > 0:
            flux = flux / max_val

    # Interpolate back to original length and remove padding from the end
    flux = torch.nn.functional.interpolate(
        flux[None, None, :], x.shape[-1], mode="linear"
    )
    # flux = flux[0, 0, : -(n_fft - hop_length)]

    return flux[0, 0]


def rms_envelope(
    x: Union[np.ndarray, torch.Tensor], window_size: int = 1024, hop_size: int = 64
) -> torch.Tensor:
    """
    Compute the RMS envelope of the signal
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    assert x.ndim == 1, "Monophone audio only."
    rms = x[None, None, :]

    # Compute the RMS envelope
    rms = torch.nn.functional.pad(rms, (window_size - hop_size, 0), mode="constant")
    rms = torch.nn.functional.unfold(rms, (1, window_size), stride=(1, hop_size))
    rms = torch.sqrt(torch.mean(rms**2, dim=0))

    # Interpolate back to original length and remove padding from the end
    rms = torch.nn.functional.interpolate(
        rms[None, None, :], x.shape[-1], mode="linear"
    )
    rms = rms[0, 0]

    max_val = torch.max(rms)
    if max_val > 0:
        rms = rms / max_val

    return rms


def find_onset(x: np.ndarray, threshold: float = None) -> int:
    """
    Pick the first peak above the threshold
    """
    peaks, _ = find_peaks(x, height=threshold)
    return peaks


def librosa_onset_detection(x: Union[np.ndarray, torch.Tensor], sr: int) -> int:
    """
    Onset detection using librosa
    """
    if librosa is None:
        raise ImportError("Librosa is not installed")

    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()

    onset_envelope = librosa.onset.onset_strength(
        y=x, sr=sr, n_fft=2048, hop_length=512, pad_mode="reflect"
    )
    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_envelope, sr=sr, backtrack=False, units="samples"
    )

    # interpolate envelope back to original length
    onset_envelope = np.interp(
        np.arange(x.shape[-1]), np.arange(onset_envelope.shape[-1]), onset_envelope
    )

    return onsets.astype(np.int32), onset_envelope


@jit(nopython=True)
def find_offset(envelope: np.array, threshold: float, normalize: bool = True):
    """
    Find the offset of the signal
    """
    if normalize:
        envelope = envelope / np.max(np.abs(envelope))

    offset = envelope.shape[-1] - 1
    amplitude = envelope[offset]
    while amplitude < threshold:
        offset -= 1
        amplitude = envelope[offset]

    return offset
