"""
Module containing the metrics used to evaluate musical content preservation in audio
"""

import numpy as np


class MetricBase:
    """
    Base class for metrics used to evaluate audio content preservation.
    """

    def __call__(self, reference: np.ndarray, output: np.ndarray) -> float:
        """
        Evaluate the content preservation of audio signals x and y.
        """
        return self.evaluate(reference, output)

    def evaluate(self, reference: np.ndarray, output: np.ndarray) -> float:
        """
        Evaluate the content preservation of audio signals x and y.
        Must be implemented in subclass.
        """
        raise NotImplementedError()


class MAE(MetricBase):
    """
    Mean absolute error metric.
    """

    def evaluate(self, reference: np.ndarray, output: np.ndarray) -> float:
        return np.mean(np.abs(reference - output))


class MaskedMAE(MetricBase):
    """
    Applies masking on the signals (calculated based on a threshold from the other)
    before calculating the MAE.
    """

    def __init__(
        self, threshold: float = 0.0, from_ref: bool = False, from_output: bool = False
    ):
        assert from_ref or from_output, "No masking direction specified"
        self.threshold = threshold
        self.from_ref = from_ref
        self.from_output = from_output

    def evaluate(self, reference: np.ndarray, output: np.ndarray) -> float:
        if self.from_ref:
            output = output * (reference > self.threshold)

        if self.from_output:
            reference = reference * (output > self.threshold)

        return np.mean(np.abs(reference - output))


class PitchVoicingAccuracy(MetricBase):
    """
    Voicing accuracy metric. Calculates the proportion of frames where the pitch
    is non-zero.
    """

    def evaluate(self, reference: np.ndarray, output: np.ndarray) -> float:
        assert reference.ndim == 1
        assert reference.shape == output.shape
        accuracy = (reference > 0.0) == (output > 0.0)
        return accuracy.sum() / len(accuracy)


class PitchVoicingFalsePositives(MetricBase):
    def evaluate(self, reference: np.ndarray, output: np.ndarray) -> float:
        assert reference.ndim == 1
        assert reference.shape == output.shape
        fp_ratio = (reference > 0.0) < (output > 0.0)
        return fp_ratio.sum() / len(fp_ratio)


class PitchVoicingFalseNegatives(MetricBase):
    def evaluate(self, reference: np.ndarray, output: np.ndarray) -> float:
        assert reference.ndim == 1
        assert reference.shape == output.shape
        fp_ratio = (reference > 0.0) > (output > 0.0)
        return fp_ratio.sum() / len(fp_ratio)


class NCC(MetricBase):
    """
    Normalized cross-correlation metric.
    """

    def evaluate(self, reference: np.ndarray, output: np.ndarray) -> float:
        assert reference.ndim == 1
        assert reference.shape == output.shape
        ncc = np.sum(normalize(reference) * normalize(output))
        ncc = (1.0 / (len(reference) - 1)) * ncc
        return ncc


class Accuracy(MetricBase):
    """
    Accuracy metric. Performs frame-wise comparison of reference and output and
    returns the proportion of frames that are the same given a tolerance.
    """

    def __init__(self, tolerance: float):
        self.tolerance = tolerance

    def evaluate(self, reference: np.ndarray, output: np.ndarray) -> float:
        assert reference.ndim == 1
        assert reference.shape == output.shape

        # Compute the abs error between the reference and output
        error = np.abs(reference - output)

        # Compute the accuracy based on the error and tolerance
        accuracy = np.zeros_like(error)
        accuracy[np.where(error < self.tolerance)[0]] = 1.0
        accuracy = accuracy.sum() / len(error)

        return accuracy


def normalize(x: np.ndarray):
    std = np.std(x, ddof=1)
    x = x - np.mean(x)
    if std != 0.0:
        x = x / std
    return x
