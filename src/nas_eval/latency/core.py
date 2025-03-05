"""
Neural Latency Evaluation
"""

import json
from dataclasses import dataclass
from collections import OrderedDict
from itertools import product
from pathlib import Path
from typing import List, Literal, Optional

from loguru import logger
import numpy as np
import torch
from tqdm import tqdm

from ..common import ModelWrapper
from .onset import get_onset_detection_function, OffsetDetection
from .signal import HarmonicSinusoidal, StackedSinusoidal, TestSignalBase, WhiteNoise


EVAL_SEED = 42

SOURCE_TYPES = ["white_noise", "sinusoidal", "harmonic"]


@dataclass
class TestConfig:
    """
    A single test configuration for latency evaluation
    """

    source: Literal["white_noise", "sinusoidal", "harmonic"]
    length: int
    decibel: int
    decay: bool


class TestSuiteConfig:
    """
    Class to setup signal generators for latency evaluation.
    Iterates over all combinations of signal generators and parameters.
    """

    def __init__(
        self,
        sources: Optional[
            List[Literal["white_noise", "sinusoidal", "harmonic"]]
        ] = None,
        lengths: Optional[List[int]] = None,
        decibels: Optional[List[int]] = None,
        decays: Optional[List[bool]] = None,
    ):
        # Setup the test suite, using a default set of parameters if none are provided
        if sources is None:
            sources = SOURCE_TYPES
        self.sources = sources

        if lengths is None:
            lengths = [4096, 44100]
        self.lengths = lengths

        if decibels is None:
            decibels = [0, -6]
        self.decibels = decibels

        if decays is None:
            decays = [True]
        self.decays = decays

    def __iter__(self):
        for source, length, db, decay in product(
            self.sources, self.lengths, self.decibels, self.decays
        ):
            yield TestConfig(source, length, db, decay)

    def __len__(self):
        return (
            len(self.sources)
            * len(self.lengths)
            * len(self.decibels)
            * len(self.decays)
        )


class NeuralLatencyEvaluator:
    """
    Core class for evaluation neural audio synthesis model for latency.
    """

    def __init__(
        self,
        model: ModelWrapper,
        iterations_per_test: int = 500,
        test_suite: Optional[TestSuiteConfig] = None,
        onset_detection: Literal[
            "amplitude", "energy", "spectral_flux", "librosa", "combined"
        ] = "combined",
        eval_offset: bool = False,
        seed: int = EVAL_SEED,
    ):
        assert isinstance(model, ModelWrapper), "model must be a ModelWrapper"
        self.model = model
        self.iterations_per_test = iterations_per_test
        self.onset_detector = get_onset_detection_function(
            onset_detection, model.sample_rate()
        )

        # Test suite configuration
        self.test_suite = test_suite
        if test_suite is None:
            self.test_suite = TestSuiteConfig()

        # Setup a reproducible test suite
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.create_test_suite()

        # Create the offset envelope detector
        if eval_offset:
            self.offset_detector = OffsetDetection()

    def create_test_suite(self):
        """
        Register all signal generators for the evaluation.
        Can be overridden by subclasses to add / modify the tests performed.
        """
        # Register the signal generators
        sample_rate = self.model.sample_rate()

        for config in self.test_suite:
            if config.source == "white_noise":
                name = f"white_noise_{config.length}_{config.decibel}dB"
                if config.decay:
                    name += "_decay"

                self._register_signal_generator(
                    WhiteNoise(
                        sample_rate=sample_rate,
                        num_signals=self.iterations_per_test,
                        random_start=(2048, sample_rate),
                        tail_length=sample_rate,
                        required_length_multiple=2048,
                        signal_length=config.length,
                        generator=self.generator,
                        amplitude=np.power(10, config.decibel / 20),
                        decay=config.decay,
                        cache=False,
                    ),
                    name,
                )

            # Sinusoidal excitation
            elif config.source == "sinusoidal":
                name = f"sinusoidal_{config.length}_{config.decibel}dB"
                if config.decay:
                    name += "_decay"

                self._register_signal_generator(
                    StackedSinusoidal(
                        sample_rate=sample_rate,
                        num_signals=self.iterations_per_test,
                        random_start=(2048, sample_rate),
                        tail_length=sample_rate,
                        required_length_multiple=2048,
                        signal_length=config.length,
                        generator=self.generator,
                        amplitude=np.power(10, config.decibel / 20),
                        decay=config.decay,
                        cache=False,
                    ),
                    name,
                )

            # Harmonic sinusoidal excitation
            elif config.source == "harmonic":
                name = f"harmonic_sinusoidal_{config.length}_{config.decibel}dB"
                if config.decay:
                    name += "_decay"

                self._register_signal_generator(
                    HarmonicSinusoidal(
                        sample_rate=sample_rate,
                        num_signals=self.iterations_per_test,
                        random_start=(2048, sample_rate),
                        tail_length=sample_rate,
                        required_length_multiple=2048,
                        signal_length=config.length,
                        generator=self.generator,
                        amplitude=np.power(10, config.decibel / 20),
                        decay=config.decay,
                        cache=False,
                    ),
                    name,
                )

    def evaluate(self):
        """
        Perform suite of evaluation tests
        """
        self.eval_log = {}
        for name, signal_generator in self.signal_generators.items():
            print("")
            print("-" * 80)
            print(f"Running test: {name}")

            for i, (x, start, stop) in tqdm(
                enumerate(signal_generator), total=self.iterations_per_test
            ):
                x = x.to(self.model.current_device())[None, None, :]

                # Reset the model state
                self.model.reset()

                # Perform forward pass
                with torch.no_grad():
                    y = self.model.forward(x)
                    y = y[..., : x.shape[-1]]

                # Measure latency
                latency, _, onset_env = self._measure_latency(y[0, 0], start)

                # Measure offset
                offset_latency = None
                if hasattr(self, "offset_detector"):
                    offset_latency, offset, env = self._measure_offset_latency(
                        y[0, 0], stop
                    )

                if latency is not None:
                    self._update_log(
                        name,
                        latency,
                        offset_latency,
                        signal_generator.get_signal_info(),
                    )

            self._print_results(name)
        # self._save_results()

    def _register_signal_generator(
        self, signal_generator: TestSignalBase, name: str = None
    ):
        """
        Register a signal generator for the evaluation
        """
        if not hasattr(self, "signal_generators"):
            self.signal_generators = OrderedDict()
        self.signal_generators.update({name: signal_generator})

    def _measure_latency(self, signal: torch.Tensor, start: int, plot=False):
        """
        Measure the latency of the model
        """
        # Select portion of the signal starting from the expected onset
        start_pad = min(2048, start)
        cropped_signal = signal[start - start_pad :]

        onsets, onset_env = self.onset_detector(cropped_signal)
        if len(onsets) == 0:
            logger.warning("No onsets detected")
            return None, None

        onset = onsets[0]
        latency = onset - start_pad

        onset_env = np.pad(onset_env, (start - start_pad, 0), mode="constant")
        return latency, onset, onset_env

    def _measure_offset_latency(self, signal: torch.Tensor, stop: int):
        """
        Compute the offset time of the model
        """
        offset, envelope = self.offset_detector(signal)
        return offset - stop, offset, envelope

    def _update_log(
        self,
        name: str,
        latency: int,
        offset_latency: Optional[int] = None,
        info: Optional[dict] = None,
    ):
        """
        Update the evaluation log
        """
        if name not in self.eval_log:
            self.eval_log[name] = {}
            self.eval_log[name]["latency"] = []
            if offset_latency is not None:
                self.eval_log[name]["offset_latency"] = []
            if info is not None:
                self.eval_log[name]["signal"] = []

        self.eval_log[name]["latency"].append(int(latency))
        if offset_latency is not None:
            self.eval_log[name]["offset_latency"].append(int(offset_latency))
        if info is not None:
            self.eval_log[name]["signal"].append(info)

    def _print_results(self, name: str):
        """
        Print the results of the evaluation
        """
        print(f"Results for test: {name}")
        print(f"Latency: {np.mean(self.eval_log[name]['latency'])} samples")
        print(f"Std Dev: {np.std(self.eval_log[name]['latency'])} samples")

    def _compute_log_stats(self):
        """
        Compute the statistics of the evaluation log
        """
        stats = {}
        for name in self.eval_log:
            stats[name] = {}
            latency = {
                "mean": float(np.mean(self.eval_log[name]["latency"])),
                "std": float(np.std(self.eval_log[name]["latency"])),
                "min": float(np.min(self.eval_log[name]["latency"])),
                "max": float(np.max(self.eval_log[name]["latency"])),
            }
            stats[name]["latency"] = latency

            if "offset_latency" in self.eval_log[name]:
                offset = {
                    "mean": float(np.mean(self.eval_log[name]["offset_latency"])),
                    "std": float(np.std(self.eval_log[name]["offset_latency"])),
                    "min": float(np.min(self.eval_log[name]["offset_latency"])),
                    "max": float(np.max(self.eval_log[name]["offset_latency"])),
                }
                stats[name]["offset"] = offset

        return stats

    def save_results(self, outfile: str):
        """
        Save the results of the evaluation to a file
        """
        # Check the file and create directory if it doesn't exist
        outfile = Path(outfile)
        if outfile.suffix != ".json":
            logger.warning("Output file should have a .json extension, appending...")
            outfile = outfile.with_suffix(".json")

        if not outfile.parent.exists():
            outfile.parent.mkdir(parents=True)

        # Compute summary statistics statistics
        stats = self._compute_log_stats()

        # Save stats to file
        with open(outfile, "w") as f:
            stats["sample_rate"] = self.model.sample_rate()
            json.dump(stats, f, indent=4)

    def save_log(self, outfile: str):
        """
        Save the full log to a file
        """
        # Check the file and create directory if it doesn't exist
        outfile = Path(outfile)
        if outfile.suffix != ".json":
            logger.warning("Output file should have a .json extension, appending...")
            outfile = outfile.with_suffix(".json")

        if not outfile.parent.exists():
            outfile.parent.mkdir(parents=True)

        # Save the full log to a file
        with open(outfile, "w") as f:
            json.dump(self.eval_log, f)
