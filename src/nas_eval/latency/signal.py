"""
Test signal generation
"""

from typing import Optional
from typing import Tuple

import numpy as np
import torch


class TestSignalBase:
    """
    Base class for test signal generators.
    Should implement an iterator interface.
    """

    def __init__(
        self,
        sample_rate: int,  # Target sample rate
        num_signals: int,  # Number of signals to generate in the iterator
        random_start: Tuple[int, int],  # Tuple of (start, end) for random start
        generator: Optional[torch.Generator] = None,
        tail_length: int = 0,  # Minimum tail length for the signal
        required_length_multiple: int = 1,  # Required length multiple for the signal
    ):
        self.sample_rate = sample_rate
        self.num_signals = num_signals
        self.generator = generator
        if self.generator is None:
            self.generator = torch.Generator(device="cpu")

        # Get the initial state of the generator
        self.gen_state = self.generator.get_state()

        self.random_start = random_start
        self.tail_length = tail_length
        self.required_length_multiple = required_length_multiple

    def __len__(self):
        return self.num_signals

    def __iter__(self):
        """
        Reset the generator state and return the iterator.
        """
        self.i = 1
        self.generator.set_state(self.gen_state)
        return self

    def __next__(self):
        """
        Updates the iterator state. Subclasses should call this at the beginning
        of the __next__ method.
        """
        if self.i > self.num_signals:
            raise StopIteration
        self.i += 1

    def get_signal_info(self):
        """
        Can be implemented and used to return current signal information.
        """
        return None

    def _random_shift(self, signal: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Create a randomly shift version of the input signal within the
        target signal length.
        """
        # Generate a random start position
        random_start = torch.randint(
            self.random_start[0], self.random_start[1], (1,), generator=self.generator
        )

        # Calculate the required length of the signal
        length = random_start + signal.shape[-1] + self.tail_length
        if length % self.required_length_multiple != 0:
            length += (
                self.required_length_multiple - length % self.required_length_multiple
            )

        x = torch.zeros(length, dtype=signal.dtype, device=signal.device)

        # Insert the signal at the random start position
        x[random_start : random_start + signal.shape[-1]] = signal

        return x, random_start.item(), (random_start + signal.shape[-1]).item()


class WhiteNoise(TestSignalBase):
    """
    White noise test signal generator.
    """

    def __init__(
        self,
        sample_rate: int,
        num_signals: int,
        random_start: Tuple[int, int],
        signal_length: int,
        amplitude: float = 0.5,
        decay: Optional[float] = False,
        generator: Optional[torch.Generator] = None,
        tail_length: int = 0,
        required_length_multiple: int = 1,
        cache: bool = True,
    ):
        super().__init__(
            sample_rate=sample_rate,
            num_signals=num_signals,
            random_start=random_start,
            tail_length=tail_length,
            required_length_multiple=required_length_multiple,
            generator=generator,
        )
        self.signal_length = signal_length
        self.amplitude = amplitude
        self.decay = decay
        self.cache = cache
        if self.cache:
            self.regenerate_noise()

    def __next__(self):
        super().__next__()
        if not self.cache:
            self.regenerate_noise()
        return self._random_shift(self.noise)

    def regenerate_noise(self):
        self.noise = (
            torch.rand(self.signal_length, generator=self.generator) * 2.0 - 1.0
        )
        self.noise = self.noise * self.amplitude
        if self.decay:
            self.noise = apply_decay(self.noise, amount=-60.0)


class StackedSinusoidal(TestSignalBase):
    """
    Signal generator that generates a stack of sinusoidal signals at log frequencies
    """

    def __init__(
        self,
        sample_rate: int,
        num_signals: int,
        random_start: Tuple[int, int],
        signal_length: int,
        amplitude: float = 0.5,
        decay: Optional[float] = False,
        generator: Optional[torch.Generator] = None,
        tail_length: int = 0,
        required_length_multiple: int = 1,
        cache: bool = True,
        min_frequency_range: Tuple[float, float] = (20, 400),
        max_frequency_range: Tuple[float, float] = (5000, 15000),
        resolution: float = 0.5,
    ):
        super().__init__(
            sample_rate=sample_rate,
            num_signals=num_signals,
            random_start=random_start,
            tail_length=tail_length,
            required_length_multiple=required_length_multiple,
            generator=generator,
        )
        self.signal_length = signal_length
        self.amplitude = amplitude
        self.decay = decay
        self.cache = cache
        self.min_frequency_range = min_frequency_range
        self.max_frequency_range = max_frequency_range
        self.resolution = resolution
        if self.cache:
            self.regenerate()

    def __next__(self):
        super().__next__()
        if not self.cache:
            self.regenerate()
        return self._random_shift(self.signal)

    def get_signal_info(self):
        return {
            "bandwidth": self.bandwidth,
        }

    def regenerate(self):
        # Select random frequency ranges
        min_frequency = torch.randint(
            self.min_frequency_range[0],
            self.min_frequency_range[1],
            (1,),
            generator=self.generator,
        ).item()
        max_frequency = torch.randint(
            self.max_frequency_range[0],
            self.max_frequency_range[1],
            (1,),
            generator=self.generator,
        ).item()
        self.bandwidth = max_frequency - min_frequency

        i = 0
        freq = 0
        n = np.arange(self.signal_length) * 2 * np.pi / self.sample_rate
        x = torch.zeros(self.signal_length, dtype=torch.float32)

        # Generate log-spaced sinusoidal signals within the frequency range
        while freq < max_frequency:
            freq = np.power(np.power(2.0, self.resolution / 12.0), i) * min_frequency
            phase = torch.rand(1, generator=self.generator).item() * 2 * np.pi
            signal = np.sin(n * freq + phase)
            x += torch.tensor(signal, dtype=torch.float32)
            i += 1

        self.signal = x / torch.max(torch.abs(x)) * self.amplitude
        if self.decay:
            self.signal = apply_decay(self.signal, amount=-60.0)


class HarmonicSinusoidal(TestSignalBase):
    """
    Signal generator that generates a harmonic signal
    """

    def __init__(
        self,
        sample_rate: int,
        num_signals: int,
        random_start: Tuple[int, int],
        signal_length: int,
        amplitude: float = 0.5,
        decay: Optional[float] = False,
        generator: Optional[torch.Generator] = None,
        tail_length: int = 0,
        required_length_multiple: int = 1,
        cache: bool = True,
        pitch_range: Tuple[float, float] = (24, 96),
        harmonic_range: Tuple[int, int] = (1, 50),
    ):
        super().__init__(
            sample_rate=sample_rate,
            num_signals=num_signals,
            random_start=random_start,
            tail_length=tail_length,
            required_length_multiple=required_length_multiple,
            generator=generator,
        )
        self.signal_length = signal_length
        self.amplitude = amplitude
        self.decay = decay
        self.cache = cache
        self.pitch_range = pitch_range
        self.harmonic_range = harmonic_range
        if self.cache:
            self.regenerate()

    def __next__(self):
        super().__next__()
        if not self.cache:
            self.regenerate()
        return self._random_shift(self.signal)

    def get_signal_info(self):
        return {
            "pitch": self.pitch,
            "harmonics": self.harmonics,
        }

    def regenerate(self):
        # Select random pitch
        pitch = torch.rand(1, generator=self.generator).item()
        self.pitch = (
            pitch * (self.pitch_range[1] - self.pitch_range[0]) + self.pitch_range[0]
        )

        # Calculate the frequency of the pitch
        min_frequency = 440.0 * np.power(2.0, (pitch - 69.0) / 12.0)

        # Number of harmonics
        self.harmonics = torch.randint(
            self.harmonic_range[0],
            self.harmonic_range[1],
            (1,),
            generator=self.generator,
        ).item()

        n = torch.arange(self.signal_length) * 2 * torch.pi / self.sample_rate
        x = torch.zeros(self.signal_length, dtype=torch.float32)

        i = 1
        freq = min_frequency

        # Generate a harmonic signal with random phase and amplitude
        while i <= self.harmonics and freq < self.sample_rate / 2.0:
            freq = torch.tensor(min_frequency, dtype=torch.float32) * i
            phase = torch.rand(1, generator=self.generator).item() * 2 * torch.pi

            amp = torch.rand(1, generator=self.generator).item()
            # Scale amplitude to decibels
            if i > 1:
                amp = (amp - 1.0) * 60.0
                amp = np.power(10, amp / 20.0)
            else:
                amp = 1.0

            x += torch.sin(n * freq + phase) * amp
            i += 1

        self.signal = x / torch.max(torch.abs(x)) * self.amplitude
        if self.decay:
            self.signal = apply_decay(self.signal, amount=-60.0)


def apply_decay(x: torch.Tensor, amount: float = -60.0) -> torch.Tensor:
    """
    Apply exponential decay to the input signal. This causes the signal to decay
    by the specified amount in decibels. Defaults to -60dB.
    """
    n = torch.linspace(0, 1, x.shape[-1])
    decay = torch.exp(-n * np.log(1 / np.power(10, amount / 20)))
    return x * decay
