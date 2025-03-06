<h1 align="center">nas-eval</h1>
<div align="center">
    Code for evaluating neural audio synthesizers.
</div>


## Setup

```bash
git clone https://github.com/jorshi/nas-eval.git
cd nas-eval
pip install .
```

## Content Evaluation

Content-based evaluations are computed on audio generated from your model. Use the command line interface conduct this evaluation.

### Pitch & Loudness Evaluation
How well does a model preserve pitch and loudness?

```
nas-eval pitch references/ reconstructions/
```

```
nas-eval loudness references/ reconstructions/
```

Computes error between pairs of reference and reconstruction audio files. To setup you'll need to have reference audio files in one folder and the reconstruction audio files in another with the **same** name. i.e.:

```
references/
    audio_1.wav
    audio_2.wav
    ...
reconstructions/
    audio_1.wav
    audio_2.wav
    ...
```


#### Optional Arguments
| Argument | Description                                                                                                                  |
|----------|------------------------------------------------------------------------------------------------------------------------------|
| --sr     | Sample rate to compute evalution. Will resample audio to this rate.                                                          |
| --pitch  | Pitch detection method. crepe or pyin. Defaults to crepe.                                                                   |
| --device | Device to run pitch evaluation on for crepe. Recommend using cuda if your system supports.                                   |
| --cache  | Cache features to disk when computing to avoid recomputation. These will be automatically reused during future computations. |
| --output | Output file to save results, will be saved as a JSON file.                                                                   |


### Timbre Transfer Evaluation

Uses Maximum Mean Discrepency (MMD) to compute the similarity 
between two sets of audio files. These do not need to be paired audio files. MMD is computed on MFCCs extracted and averaged over texture windows. 

References are the input audio and reconstrucions the output.
```bash
nas-eval timbre references/ reconstructions/
```

#### Example
For example, if you had trained a timbre transfer model on saxophone, and then want to evaluate timbre transfer to singing voice (svoice). 

First, run the svoice dataset through your timbre transfer model and save it to a folder, say `model_outputs/svoice`. This is svoice transferred to saxophone since your model was tranined on saxophone.

Evalaute timbre similarity between the svoice (input) and the model outputs.
```bash
nas-eval svoice/ model_outputs/svoice/
```

That returns a value of $0.16$. Now test timbre similarity between the saxophone (training data) and the model outputs/

```bash
nas-eval saxophone/ model_outputs/svoice/
```

This returns a value of $0.074$, which indicates that when the svoice is passed through the model, its outputs become closer in timbral similarity to the saxophone dataset.

#### Optional Arguments
| Argument | Description                                                         |
|----------|---------------------------------------------------------------------|
| --sr     | Sample rate to compute evalution. Will resample audio to this rate. |
| --save   | Save the MMD results into an .npy file.                             |

## Latency Evaluation

Latency is defined as a delayed response to a model input, observed at the output.
nas-eval provides functionality for wrapping your PyTorch NAS models and condusting a host of latency evaluations on synthetic data to profile latency reponse.

See an example of implementing latency evaluation on a PyTorch model in `examples/basic_latency.py`.

Three main classes are involved:
- `ModelWrapper` class wraps your model, allowing you to define model setup and forward calss, and provides a unified api for evaluation.
- `NeuralLatencyEvaluator` receives a wrapped model and performs a set of evaluations with synthetic test signals to profile latency.
- `TestSuiteConfig` is an optional class to configure the set of synthetic test
signals used.


## Cite Us

If you find this work useful please consider citing our paper:

```bibtex
@article{caspe2025designing,
    title={{Designing Neural Synthesizers for Low Latency Interaction}},
    author={Caspe, Franco and Shier, Jordie and Sandler, Mark and Saitis, Charis and McPherson, Andrew},
    journal={Journal of the Audio Engineering Society},
    year={2025}
}
```