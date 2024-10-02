# Auto Semantic Segmentation

Segment Anything based toolkit for semantic segmentation.

## Installation

1. Clone the repository locally.

```bash
git clone https://github.com/mdvdv/pylandmarker.git
```

2. Install basic requirements.

You should set the environment variable manually as follows if you want to build a local GPU environment for Grounded-SAM:

```bash
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/path/to/cuda-11.3/
```

Install Segment Anything:

```bash
python -m pip install -e segment_anything
```

Install Grounding DINO:

```bash
pip install --no-build-isolation -e GroundingDINO
```

Install diffusers:

```bash
pip install --upgrade diffusers[torch]
```

Install other requirements:

```bash
pip install -r requirements.txt
```

## Usage

In progress.

## Test

In progress.

## Documentation

In progress.

## Citation

```bash
@software{auto-semantic-segmentation,
    title = {auto-semantic-segmentation},
    author = {Medvedev, Anatolii},
    year = {2024},
    url = {https://github.com/mdvdv/auto-semantic-segmentation.git},
    version = {1.0.0}
}
```
