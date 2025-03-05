"""
Basic example of how to evaluate the latency of a model using the NeuralLatencyEvaluator.

This example includes a simple model that adds a delay to the input to simulate latency.
This model is then wrapped in a ModelWrapper, which provides a unfirorm interface
for the evaluator. A suite of latency evaluations is  then perofmred on the model
using the NeuralLatencyEvaluator.
"""

import argparse
import sys

from nas_eval import ModelWrapper, NeuralLatencyEvaluator, TestSuiteConfig
import torch


class LatencyModel(torch.nn.Module):
    """
    Simple model that adds a delay to the input to simulate latency. Optionally,
    jitter can be simulated by randomly varying the delay value on each forward pass.
    """

    def __init__(
        self,
        delay: int = 0,  # Delay in samples
        jitter: int = 0,  # Jitter in samples
        sample_rate: int = 48000,  # Sample rate of the model
    ):
        super().__init__()
        self.delay = delay
        self.jitter = jitter
        self.sample_rate = sample_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add delay to the input tensor.
        """

        assert x.dim() == 3, "Input tensor must have shape (batch, channels, samples)"
        num_samples = x.shape[2]

        # Compute delay - add normally distributed jitter if jitter > 0
        delay = self.delay
        if self.jitter > 0:
            jitter = torch.randn(1).item() * self.jitter
            delay += int(jitter)

        # Add delay to the input by concatenating zeros to the start of the tensor
        x = torch.cat(
            [torch.zeros(x.shape[0], x.shape[1], delay, device=x.device), x], dim=2
        )
        return x[..., :num_samples]


class LatencyModelWrapper(ModelWrapper):
    """
    Implementation of ModelWrapper for the LatencyModel.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass of the model. Passes in audio with the
        shape (batch, channels, samples), and expects the same shape to be returned.
        """
        return self.model(x)

    def reset(self) -> None:
        """
        Reset the model state. If your model has internal state such as
        cached convolutions, these should be cleared here to ensure that
        the latency evaluation is consistent.
        """
        pass

    def sample_rate(self) -> int:
        """
        Return the samplerate of the model.
        """
        return self.model.sample_rate

    def to(self, device: torch.device) -> None:
        """
        Move the model to a device. Make sure to set the device attribute
        for this class so that the evaluator can access it.
        """
        self.model.to(device)
        self.device = device


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--device", help="Device to run the evaluation on", type=str, default="cpu"
    )
    parser.add_argument(
        "--save", help="Save the results to specified JSON file", type=str, default=None
    )
    parser.add_argument(
        "--log",
        help="Save the log of all individual latency values to specifed JSON file",
        type=str,
        default=None,
    )

    args = parser.parse_args(arguments)

    # Create a latency evaluation model from the loaded RAVE model
    model = LatencyModel(delay=2048, jitter=128)
    model = LatencyModelWrapper(model)

    # device
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

    # Send the model to the device
    model.to(device)

    # Create the evaluator -- this is optional, but here you can specifiy the
    # signals that are used to evaluate the latency of the model. Each config
    # item is a list, and all combinations of all lists are evaluated.
    test_suite = TestSuiteConfig(
        sources=["white_noise"],  # Signal types [white_noise, sinusoidal, harmonic]
        lengths=[4096],  # Length of the test signals
        decibels=[0.0, -6.0],  # Decibel levels of the test signals
        decays=[True],  # Apply exponential decay to the test signals
    )

    evaluator = NeuralLatencyEvaluator(model, test_suite=test_suite)
    evaluator.evaluate()

    # Save the results
    if args.save is not None:
        evaluator.save_results(args.save)

    # Save the log
    if args.log is not None:
        evaluator.save_log(args.log)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
