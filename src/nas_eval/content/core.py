from pathlib import Path
from typing import Dict
from typing import Literal
from typing import Tuple
from typing import Union

import librosa
import numpy as np
import torch
import torchcrepe
from tqdm import tqdm

from .metrics import Accuracy
from .metrics import MAE
from .metrics import MaskedMAE
from .metrics import MetricBase
from .metrics import NCC
from .metrics import PitchVoicingAccuracy
from .metrics import PitchVoicingFalseNegatives
from .metrics import PitchVoicingFalsePositives


class ContentEvalBase:
    metrics: Dict[str, Union[MetricBase, callable]]  # Must be defined in subclass
    _name: str  # Must be defined in subclass

    def __init__(
        self,
        sample_rate: int,
        align=None,
        cache_inputs: bool = False,
        concat_features: bool = True,
    ):
        self.sample_rate = sample_rate
        self.align = align or AlignmentIdentity()
        self.cache_inputs = cache_inputs
        self.concat_features = concat_features

    def __call__(
        self,
        input_audio: Dict[str, torch.Tensor] = None,
        output_audio: Dict[str, torch.Tensor] = None,
        input_feature: Dict[str, np.ndarray] = None,
        output_feature: Dict[str, np.ndarray] = None,
    ) -> Tuple[Dict[str, Dict[str, np.ndarray]], list]:
        """
        Perform evaluation of content. Extract features from inputs and outputs
        and perform evaluation based on the extracted features.
        """
        assert input_audio is not None or input_feature is not None, "No input"
        assert output_audio is not None or output_feature is not None, "No output"

        if input_feature is None:
            input_feature = self.compute_input_feature_dict(input_audio)

        if output_feature is None:
            output_feature = self.compute_feature_dict(output_audio)

        self.input_feature = input_feature
        self.output_feature = output_feature
        assert (
            input_feature.keys() == output_feature.keys()
        ), "Input and output keys do not match"
        return self.evaluate(input_feature, output_feature)

    @property
    def name(self):
        assert hasattr(self, "_name"), "Name must be defined in subclass"
        return self._name

    def compute_feature_dict(
        self, audio: Dict[str, torch.Tensor]
    ) -> Dict[str, np.ndarray]:
        """
        Calculate features for each audio file in the input dictionary
        and return a dictionary of features.
        """
        features = {}
        for key, waveform in tqdm(audio.items()):
            features[key] = self.compute_feature(waveform)
        return features

    def compute_input_feature_dict(
        self, audio: Dict[str, torch.Tensor]
    ) -> Dict[str, np.ndarray]:
        """
        Calculate features for each audio file in the input dictionary
        and return a dictionary of features.
        """
        if self.cache_inputs:
            if (
                hasattr(self, "_input_feature")
                and self._input_feature.keys() == audio.keys()
            ):
                print("Using cached input features")
                return self._input_feature
            self._input_feature = self.compute_feature_dict(audio)
            return self._input_feature
        return self.compute_feature_dict(audio)

    def compute_feature(self, audio: torch.Tensor):
        """
        Must be implemented in subclass. Compute features from audio tensor.
        """
        raise NotImplementedError()

    def evaluate(self, in_features: np.ndarray, out_features: np.ndarray):
        """
        Evaluate the features extracted from the input and output audio using
        the metrics defined in the subclass. Return a dictionary of results along
        with the order of the audio files evaluated.
        """
        assert len(self.metrics) > 0, "Metrics must be defined in subclass"
        results = {k: [] for k in self.metrics}
        eval_order = sorted(list(in_features.keys()))

        aligned_x = []
        aligned_y = []
        for key in eval_order:
            x, y = self.align(in_features[key], out_features[key])
            aligned_x.append(x)
            aligned_y.append(y)

        if self.concat_features:
            aligned_x = (np.concatenate(aligned_x, axis=0),)
            aligned_y = (np.concatenate(aligned_y, axis=0),)

        for x, y in zip(aligned_x, aligned_y):
            for name, metric in self.metrics.items():
                results[name].append(float(metric(x, y)))

        return results, eval_order

    def save_output_features(self, folder: str, suffix: str = ""):
        """
        Save output features to a folder with a suffix.
        """
        assert hasattr(self, "output_feature"), "No output features cached"
        self.save_features(self.output_feature, folder, suffix)

    def save_input_features(self, folder: str, suffix: str = ""):
        """
        Save cached input feature to a folder with a suffix.
        """
        assert hasattr(self, "input_feature"), "No input features cached"
        self.save_features(self.input_feature, folder, suffix)

    def save_features(self, features: Dict, folder: str, suffix: str = ""):
        folder = Path(folder)
        assert folder.is_dir(), f"Folder {folder} does not exist"
        for key, feature in features.items():
            out_file = Path(folder, f"{key}_{suffix}.npy")
            np.save(out_file, feature)


class PitchEval(ContentEvalBase):
    _name = "pitch"

    def __init__(
        self,
        sample_rate: int,
        frame_rate: float = 200.0,  # Hz
        method: Literal["crepe", "pyin"] = "crepe",
        device: Literal["cpu", "cuda"] = "cpu",
        **kwargs,
    ):
        pad = int(frame_rate / 2.0)  # Allow adjustment +/- 500ms (50% of frame rate)
        alignment = AlignmentMinL1(pad=pad, constant=0.0)
        super().__init__(sample_rate, align=alignment, **kwargs)
        self.method = method
        self.hop_length = int(sample_rate / frame_rate)

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            print("CUDA not available, using CPU")

        self.device = device
        self.metrics = {
            "rpa_0.5": Accuracy(tolerance=0.5),  # Tolerance in semitones
            "rpa_1.0": Accuracy(tolerance=1.0),  # Tolerance in semitones
            "rpa_2.0": Accuracy(tolerance=2.0),  # Tolerance in semitones
            "ncc": NCC(),
            "mae": MAE(),
            "mae_masked_output": MaskedMAE(from_output=True),
            "mae_masked_ref": MaskedMAE(from_ref=True),
            "mae_masked_both": MaskedMAE(from_ref=True, from_output=True),
            "pva": PitchVoicingAccuracy(),
            "fpr": PitchVoicingFalsePositives(),
            "fnr": PitchVoicingFalseNegatives(),
        }

    def compute_feature(self, audio: torch.Tensor):
        assert audio.ndim == 2
        assert audio.shape[0] == 1, "monophonic audio only"

        if self.method == "crepe":
            f0 = self.compute_crepe(audio)
        elif self.method == "pyin":
            f0 = self.compute_pyin(audio)

        pitch = librosa.hz_to_midi(f0)
        np.nan_to_num(pitch, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if np.any(np.isnan(pitch)) or np.any(np.isinf(pitch)):
            assert False, "Pitch contains NaN or Inf values"
        return pitch

    def compute_crepe(self, audio: torch.Tensor):
        f0, periodicity = torchcrepe.predict(
            audio,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            fmin=50.0,
            fmax=2000.0,
            model="full",
            device=self.device,
            batch_size=2048,
            return_periodicity=True,
        )

        # Filter pitch on silence & periodicity
        periodicity = torchcrepe.threshold.Silence(-60.0)(
            periodicity,
            audio=audio,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
        )
        f0 = torchcrepe.threshold.At(0.85)(f0, periodicity)
        f0 = f0.numpy()[0]
        f0 = np.nan_to_num(f0)
        return f0

    def compute_pyin(self, audio: torch.Tensor):
        audio = audio.numpy()[0]
        f0, _, _ = librosa.pyin(
            audio,
            fmin=50.0,
            fmax=2000.0,
            sr=self.sample_rate,
            hop_length=self.hop_length,
        )
        f0 = np.nan_to_num(f0)
        return f0


class LoudnessEval(ContentEvalBase):
    _name = "loudness"

    def __init__(
        self,
        sample_rate: int,
        n_fft: int = 2048,
        hop_size: int = 2048 // 8,
        activation: bool = False,
        **kwargs,
    ):
        pad = int(0.5 * sample_rate / hop_size)
        alignment = AlignmentMinL1(pad=pad, constant=0.0)
        # alignment = AlignmentMinLength()
        super().__init__(sample_rate, align=alignment, **kwargs)
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.activation = activation
        self.metrics = {
            "accuracy": Accuracy(tolerance=2.0),  # Tolerance of 2 dB
            "mae": MAE(),
            "ncc": NCC(),
        }

    def compute_feature(self, audio: torch.Tensor):
        # Monophonic audio only supported for now
        assert audio.ndim == 2
        assert audio.shape[0] == 1

        # Convert to 1D numpy array
        audio = audio.numpy()[0]

        # A-weighted loudness envlope
        mag = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_size)
        power = np.abs(mag) ** 2
        fft_freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)
        a_weight = librosa.A_weighting(fft_freqs)
        weighting = 10 ** (a_weight / 10)
        power = power * weighting[..., None]

        # Summarize over all frequencies
        power = np.mean(power, axis=0)
        loudness = librosa.power_to_db(power, ref=80.0)

        # Convert to an activation signal
        if self.activation:
            loudness = np.diff(loudness)
            loudness[loudness < 0.0] = 0.0

        return loudness


class AlignmentBase:
    """
    Base class for alignment methods. Subclasses must implement the align method.
    """

    def __call__(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.align(x, y)

    def align(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()


class AlignmentIdentity(AlignmentBase):
    """
    Return the input sequences as is
    """

    def align(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return x, y


class AlignmentMinLength(AlignmentBase):
    """
    Return the sequence trimmed to the length of the shorter two
    """

    def align(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = min(x.shape[-1], y.shape[-1])
        return x[..., :n], y[..., :n]


class AlignmentCorr(AlignmentBase):
    """
    Align entire signal using correlation. Pad y with pad number of constant values
    """

    def __init__(self, pad: int = 10, constant=0.0):
        self.pad = pad
        self.constant = constant

    def align(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y = np.pad(y, (self.pad, self.pad), constant_values=self.constant)
        c = np.correlate(x, y, mode="valid")
        c = self.pad + (self.pad - np.argmax(c))
        y = y[c : c + len(x)]
        return x, y


class AlignmentMinL1(AlignmentBase):
    """
    Align sequences to minimize their L1 distance
    """

    def __init__(self, pad: int = 10, constant=0.0):
        self.pad = pad
        self.constant = constant
        self.mae = MAE()

    def align(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert y.shape[0] >= x.shape[0], "y must be equal or longer than x"
        y = np.pad(y, (self.pad, self.pad), constant_values=self.constant)
        N = x.shape[-1]
        M = y.shape[-1] - N
        error = np.empty(M)
        for i in range(M):
            error[i] = self.mae(x, y[..., i : i + N])

        min_id = np.argmin(error)
        y = y[..., min_id : min_id + N]
        return x, y


class AlignmentDTW(AlignmentBase):
    def align(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray]:
        _, wp = librosa.sequence.dtw(x, y)
        aligned_x = x[wp[::-1, 0]]
        aligned_y = y[wp[::-1, 1]]

        print(aligned_x.shape)

        assert aligned_x.shape == aligned_y.shape
        return aligned_x, aligned_y


def align_sequences_dtw(self, x: np.ndarray, y: np.ndarray):
    # Align the sequences using DTW
    step_sizes = [[1, 1]]
    if len(x) < len(y):
        step_sizes.append([0, 1])
    else:
        step_sizes.append([1, 0])

    _, wp = librosa.sequence.dtw(x, y, step_sizes_sigma=step_sizes)
    aligned_x = x[wp[::-1, 0]]
    aligned_y = y[wp[::-1, 1]]

    assert aligned_x.shape == aligned_y.shape
    return aligned_x, aligned_y


def crop_sequences(self, x: np.ndarray, y: np.ndarray):
    """
    Crop the sequences to the minimum length
    """
    min_len = min(len(x), len(y))
    return x[:min_len], y[:min_len]
