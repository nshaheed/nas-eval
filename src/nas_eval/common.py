import os
from pathlib import Path
from typing import Dict

from loguru import logger
import torch
import torchaudio


class ModelWrapper(torch.nn.Module):
    """
    Wrapper class for a neural audio synthesis model which is passed into the
    evaluation class. Must implement methods required to perform the evalaulation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Perform forward pass of the model.
        """
        raise NotImplementedError()

    def reset(self) -> None:
        """
        Reset the model state.
        """
        raise NotImplementedError()

    def sample_rate(self) -> int:
        """
        Return the samplerate of the model.
        """
        raise NotImplementedError()

    def current_device(self) -> torch.device:
        """
        Return the device the model is on.
        """
        if hasattr(self, "device"):
            return self.device
        else:
            raise RuntimeError("Model has not been moved to a device")


# Assemble timbre transfer target list -- single test audio -> timbre transfer
def assemble_targets_list_single(
    matrix_name: str,
    test_audios_dir: str,
    reconstructions_dir: str,
    include_output=False,
):
    references, sorted_paths = assemble_targets_lists(
        matrix_name, test_audios_dir, reconstructions_dir
    )
    input_timbre, output_timbre = matrix_name.split("-")
    filtered_references = [f for f in references if Path(f).name == input_timbre]
    if include_output:
        output_reference = [f for f in references if Path(f).name == output_timbre]
        if len(output_reference) == 0:
            logger.error(f"No output reference found for {output_timbre}")
            quit()
        filtered_references += output_reference
    return filtered_references, sorted_paths


def assemble_targets_lists(
    matrix_name: str, test_audios_dir: str, reconstructions_dir: str
):
    input_timbre, trained_timbre = matrix_name.split("-")

    # Initialize the list to hold the directories
    references = []

    # Iterate over the directories in the base directory
    for item in os.listdir(test_audios_dir):
        item_path = os.path.join(test_audios_dir, item)
        if os.path.isdir(item_path):  # Check if it's a directory
            references.append(item_path)

    cross_synthesis = []
    # Iterate over the directories in the base directory
    for item in os.listdir(reconstructions_dir):
        item_path = os.path.join(reconstructions_dir, item)
        # Check if it's a directory and if it contains the trained timbre
        if os.path.isdir(item_path) and trained_timbre in item:
            # Construct the full path for the input timbre
            source_path = os.path.join(item_path, input_timbre)
            # Check if the input directory exists within the trained timbre directory
            if os.path.isdir(source_path):
                cross_synthesis.append(source_path)

    # Sort the list based on the extracted numbers
    cross_synthesis.sort(key=lambda x: Path(x).parent.name)

    return references, cross_synthesis


def validate_input_dirs(references: Path, reconstructions: Path) -> bool:
    """
    Validate the input directories for the evaulations. These should be directories
    of wav files with the same number of files and the same names in each directory.
    """
    references = Path(references)
    reconstructions = Path(reconstructions)
    if not references.exists() and not references.is_dir():
        raise ValueError(f"Reference directory {references} does not exist.")
    if not reconstructions.exists() and not reconstructions.is_dir():
        raise ValueError(f"Reconstruction directory {reconstructions} does not exist.")

    # Iterate wav files in the directories and confirm the wav files match
    suffix = "*.[wW][aA][vV]"
    references_files = list(references.glob(suffix))
    reconstructions_files = list(reconstructions.glob(suffix))
    if len(references_files) == 0:
        raise RuntimeError(f"No wav files found in reference directory {references}")

    if len(references_files) != len(reconstructions_files):
        raise RuntimeError(
            f"Number of files in reference and reconstruction directories do not match. "
            f"Reference: {len(references_files)}, Reconstruction: {len(reconstructions_files)}"
        )

    recon_names = [Path(file).name for file in reconstructions_files]
    for ref_file in references_files:
        if Path(ref_file).name not in recon_names:
            raise RuntimeError(
                f"Reference file {ref_file} does not have a corresponding reconstruction."
            )


def infer_sample_rate(audio_dir: str) -> int:
    """
    Infer the sample rate of the audio files in the directory.
    """
    audio_dir = Path(audio_dir)
    suffix = "*.[wW][aA][vV]"
    audio_files = list(audio_dir.rglob(suffix))
    sample_rates = set()
    for audio_file in audio_files:
        _, sample_rate = torchaudio.load(audio_file)
        sample_rates.add(sample_rate)

    sample_rate = sample_rates.pop()
    if len(sample_rates) > 1:
        logger.warning(
            f"Audio files have different sample rates, using {sample_rate}. To set a specific sample rate, use the --sr flag."
        )
    else:
        logger.warning(
            f"Inferred sample rate: {sample_rate}. To set a specific sample rate, use the --sr flag."
        )

    return sample_rate


def load_audio_files(
    audio_dir: str, sr: int, num_samples: int = None
) -> Dict[str, torch.Tensor]:
    audio_dir = Path(audio_dir)
    suffix = "*.[wW][aA][vV]"
    audio_files = list(audio_dir.glob(suffix))
    audio = {}

    for audio_file in audio_files:
        if audio_file.name.startswith("."):
            logger.warning(f"Skipping hidden file {audio_file}")
            continue

        waveform, sample_rate = torchaudio.load(audio_file)
        if waveform.shape[0] != 1:
            waveform = waveform[:1, :]
        if sample_rate != sr:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, sr, lowpass_filter_width=512
            )
        if num_samples is None:
            audio[Path(audio_file).name] = waveform
        else:
            audio[Path(audio_file).name] = waveform[:, :num_samples]

    return audio
