# AlbumentationsX

[![PyPI version](https://badge.fury.io/py/albumentationsx.svg)](https://badge.fury.io/py/albumentationsx)
![CI](https://github.com/albumentations-team/AlbumentationsX/workflows/CI/badge.svg)
[![PyPI Downloads](https://img.shields.io/pypi/dm/albumentationsx.svg?label=PyPI%20downloads)](https://pypi.org/project/albumentationsx/)

> 📣 **Stay updated!** [Subscribe to our newsletter](https://albumentations.ai/subscribe?utm_source=github&utm_medium=referral&utm_campaign=readme) for the latest releases, tutorials, and tips.

[![License: AGPL-3.0-only](https://img.shields.io/badge/License-AGPL--3.0--only-blue.svg)](https://github.com/albumentations-team/AlbumentationsX/blob/5bac7d5ae4b38f3c2a89c21dc088245613e9e99c/LICENSE)
[![Commercial License](https://img.shields.io/badge/Commercial_License-available-brightgreen)](https://albumentations.ai/pricing?utm_source=github&utm_medium=referral&utm_campaign=readme)

[![Docs](https://img.shields.io/badge/docs-albumentations.ai-blue)](https://albumentations.ai/docs/?utm_source=github&utm_medium=referral&utm_campaign=readme) [![Discord](https://img.shields.io/badge/Discord-join-7289da?logo=discord&logoColor=white)](https://discord.gg/AKPrrDYNAt) [![Twitter](https://img.shields.io/badge/Twitter-follow-1da1f2?logo=twitter&logoColor=white)](https://twitter.com/albumentations) [![LinkedIn](https://img.shields.io/badge/LinkedIn-connect-0077b5?logo=linkedin&logoColor=white)](https://www.linkedin.com/company/albumentations/) [![Reddit](https://img.shields.io/badge/Reddit-join-ff4500?logo=reddit&logoColor=white)](https://www.reddit.com/r/Albumentations/)

**AlbumentationsX** is a Python library for image augmentation. It provides high-performance, robust implementations and cutting-edge features for computer vision tasks. Image augmentation is used in deep learning and computer vision to increase the quality of trained models. The purpose of image augmentation is to create new training samples from the existing data.

## Citing

If AlbumentationsX supports your research, please cite
[AlbumentationsX: One Augmentation Pipeline for Images and Related Annotations](https://arxiv.org/abs/2608.11123).
Your citation makes the project's research impact visible to funders and helps sustain maintenance.

```bibtex
@article{iglovikov2026albumentationsx,
    title = {AlbumentationsX: One Augmentation Pipeline for Images and Related Annotations},
    author = {Iglovikov, Vladimir},
    journal = {arXiv preprint arXiv:2608.11123},
    year = {2026},
    doi = {10.48550/arXiv.2608.11123},
    url = {https://arxiv.org/abs/2608.11123}
}
```

<a id="-licensing-commercial-use-is-allowed"></a>

## Licensing

AlbumentationsX offers two license options:

- **Commercial license:** choose alternative permissions for proprietary software under an agreement covering your
  team, products, and deployments. [Request a quote](https://albumentations.ai/pricing?utm_source=github&utm_medium=referral&utm_campaign=readme)
  or email [vladimir@albumentations.ai](mailto:vladimir@albumentations.ai).
- **AGPL-3.0-only:** available at no charge. The AGPL permits commercial use subject to its terms.
  [Read the license guide](https://albumentations.ai/docs/license/).

Commercial, proprietary, internal, or production status alone does not require a commercial license.
See the [AGPL text](https://github.com/albumentations-team/AlbumentationsX/blob/5bac7d5ae4b38f3c2a89c21dc088245613e9e99c/LICENSE)
and [licensing details and history](https://github.com/albumentations-team/AlbumentationsX/blob/5bac7d5ae4b38f3c2a89c21dc088245613e9e99c/LICENSING.md)
for the applicable terms.

For supplier questionnaires and purchasing documents, see
[institutional procurement](https://github.com/albumentations-team/AlbumentationsX/blob/main/LICENSING.md#institutional-procurement).

## Quick Start

```bash
# Install the PyTorch build for your platform first. For Linux CPU-only:
pip install "torch>=2.13.0" --index-url https://download.pytorch.org/whl/cpu

# Then install AlbumentationsX with OpenCV.
pip install "albumentationsx[headless]"
```

```python
import albumentations as A

transform = A.Compose(
    [
        A.RandomCrop(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ]
)
```

---

Here is an example of how you can apply some [pixel-level](#pixel-level-transforms) augmentations to create new images from the original one:
![parrot](https://habrastorage.org/webt/bd/ne/rv/bdnerv5ctkudmsaznhw4crsdfiw.jpeg)

## Why AlbumentationsX

- **Complete Computer Vision Support**: Works with all major CV tasks
- **Simple, Unified API**: [One consistent interface](#a-simple-example) for all data types - RGB/grayscale/multispectral images, masks, bounding boxes, and keypoints.
- **Rich Augmentation Library**: [70+ high-quality augmentations](https://albumentations.ai/docs/reference/supported-targets-by-transform/?utm_source=github&utm_medium=referral&utm_campaign=readme) to enhance your training data.
- **Fast**: Consistently benchmarked as the [fastest augmentation library](https://albumentations.ai/docs/benchmarks/image-benchmarks/?utm_source=github&utm_medium=referral&utm_campaign=readme) also shown [below section](#performance-comparison), with optimizations for production use.
- **Deep Learning Integration**: Works with [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/), and other frameworks. Part of the [PyTorch ecosystem](https://pytorch.org/ecosystem/).
- **Created by Experts**: Built by [developers with deep experience in computer vision and machine learning competitions](#authors).

## Table of contents

- [Citing](#citing)
- [Licensing](#licensing)
- [Quick Start](#quick-start)
- [Why AlbumentationsX](#why-albumentationsx)
- [Authors](#authors)
- [Installation](#installation)
- [Documentation](#documentation)
- [A simple example](#a-simple-example)
- [List of augmentations](#list-of-augmentations)
  - [Pixel-level transforms](#pixel-level-transforms)
  - [Spatial-level transforms](#spatial-level-transforms)
  - [3D transforms](#3d-transforms)
- [Augmentation examples](#a-few-more-examples-of-augmentations)
- [Benchmark results](#benchmark-results)
- [Performance comparison](#performance-comparison)
- [Contribute](#-contribute)
- [License](#-license)
- [Contact](#-contact)
- [Newsletter](#-stay-connected)

## Authors

### Current Maintainer

[**Vladimir I. Iglovikov**](https://www.linkedin.com/in/iglovikov/) | [Kaggle Grandmaster](https://www.kaggle.com/iglovikov)

### Emeritus Core Team Members

[**Mikhail Druzhinin**](https://www.linkedin.com/in/mikhail-druzhinin-548229100/) | [Kaggle Expert](https://www.kaggle.com/dipetm)

[**Alex Parinov**](https://www.linkedin.com/in/alex-parinov/) | [Kaggle Master](https://www.kaggle.com/creafz)

[**Alexander Buslaev**](https://www.linkedin.com/in/al-buslaev/) | [Kaggle Master](https://www.kaggle.com/albuslaev)

[**Eugene Khvedchenya**](https://www.linkedin.com/in/cvtalks/) | [Kaggle Grandmaster](https://www.kaggle.com/bloodaxe)

## Installation

AlbumentationsX requires Python 3.11 or higher. To install the latest version from PyPI:

### Basic Installation

Install the PyTorch build for your CPU, CUDA, or MPS environment before installing AlbumentationsX. For a Linux
CPU-only environment:

```bash
pip install "torch>=2.13.0" --index-url https://download.pytorch.org/whl/cpu
```

For CUDA or macOS (MPS), use the matching command from the [PyTorch installation selector](https://pytorch.org/get-started/locally/).
AlbumentationsX does not choose or install a PyTorch accelerator build.

If you already have OpenCV installed (any variant), install AlbumentationsX:

```bash
pip install -U albumentationsx
```

### Installation with OpenCV

If you don't have OpenCV installed yet, choose the appropriate variant:

```bash
# For servers/Docker (no GUI support, lighter package)
pip install -U "albumentationsx[headless]"

# For local development with GUI support (cv2.imshow, etc.)
pip install opencv-python && pip install -U albumentationsx

# For OpenCV with extra algorithms (contrib modules)
pip install opencv-contrib-python && pip install -U albumentationsx

# For contrib + headless
pip install -U "albumentationsx[contrib-headless]"
```

**Note:** AlbumentationsX works with any OpenCV variant:

- `opencv-python` (full version with GUI)
- `opencv-python-headless` (no GUI, smaller size)
- `opencv-contrib-python` (with extra modules)
- `opencv-contrib-python-headless` (contrib + headless)

Choose the one that fits your needs. The library will detect whichever is installed.

`pip install albumentationsx` installs the base dependency set without PyTorch. It is useful for dependency-only
consumers such as documentation builds. Importing `albumentations` requires the PyTorch build you selected above.

Other installation options are described in the [documentation](https://albumentations.ai/docs/1-introduction/installation/?utm_source=github&utm_medium=referral&utm_campaign=readme).

## Documentation

The full documentation is available at **[https://albumentations.ai/docs/](https://albumentations.ai/docs/?utm_source=github&utm_medium=referral&utm_campaign=readme)**.
For security, release verification, and contributor guides, see the
[repository documentation](https://github.com/albumentations-team/AlbumentationsX/blob/main/docs/README.md).

For AI-assisted augmentation review, AlbumentationsX can also be used through MCP-capable hosts such as Claude Desktop,
Cursor, Claude Code, and Codex. The community
[AlbumentationsX MCP integration](https://github.com/albumentations-team/AlbumentationsX/blob/main/docs/integrations/mcp.md) lets assistants inspect transforms, validate pipelines,
render bounded local preview batches, compare preview runs, collect concrete feedback, and export reproducible
AlbumentationsX pipelines.

## A simple example

```python
import albumentations as A
import cv2

transform = A.Compose(
    [
        A.RandomCrop(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ]
)

image = cv2.imread("image.jpg", cv2.IMREAD_COLOR_RGB)

transformed = transform(image=image)
transformed_image = transformed["image"]
```

AlbumentationsX collects usage statistics by default to guide product development.
See [data collection and opt-out](https://github.com/albumentations-team/AlbumentationsX/blob/main/docs/privacy.md).

## List of augmentations

### Pixel-level transforms

- [AdditiveNoise](https://albumentations.ai/explore/transform/AdditiveNoise/)
- [AdvancedBlur](https://albumentations.ai/explore/transform/AdvancedBlur/)
- [BiasField](https://albumentations.ai/explore/transform/BiasField/)
- [AnnotationArtifacts](https://albumentations.ai/explore/transform/AnnotationArtifacts/)
- [AtmosphericFog](https://albumentations.ai/explore/transform/AtmosphericFog/)
- [AutoContrast](https://albumentations.ai/explore/transform/AutoContrast/)
- [Blur](https://albumentations.ai/explore/transform/Blur/)
- [CLAHE](https://albumentations.ai/explore/transform/CLAHE/)
- [ChannelDropout](https://albumentations.ai/explore/transform/ChannelDropout/)
- [ChannelShuffle](https://albumentations.ai/explore/transform/ChannelShuffle/)
- [ChannelSwap](https://albumentations.ai/explore/transform/ChannelSwap/)
- [ChromaticAberration](https://albumentations.ai/explore/transform/ChromaticAberration/)
- [ColorJitter](https://albumentations.ai/explore/transform/ColorJitter/)
- [Colorize](https://albumentations.ai/explore/transform/Colorize/)
- [Defocus](https://albumentations.ai/explore/transform/Defocus/)
- [Dithering](https://albumentations.ai/explore/transform/Dithering/)
- [Downscale](https://albumentations.ai/explore/transform/Downscale/)
- [Emboss](https://albumentations.ai/explore/transform/Emboss/)
- [Enhance](https://albumentations.ai/explore/transform/Enhance/)
- [Equalize](https://albumentations.ai/explore/transform/Equalize/)
- [ExposureMatching](https://albumentations.ai/explore/transform/ExposureMatching/)
- [FDA](https://albumentations.ai/explore/transform/FDA/)
- [FancyPCA](https://albumentations.ai/explore/transform/FancyPCA/)
- [FilmGrain](https://albumentations.ai/explore/transform/FilmGrain/)
- [FromFloat](https://albumentations.ai/explore/transform/FromFloat/)
- [GaussNoise](https://albumentations.ai/explore/transform/GaussNoise/)
- [GaussianBlur](https://albumentations.ai/explore/transform/GaussianBlur/)
- [GlassBlur](https://albumentations.ai/explore/transform/GlassBlur/)
- [HEStain](https://albumentations.ai/explore/transform/HEStain/)
- [Halftone](https://albumentations.ai/explore/transform/Halftone/)
- [HistogramMatching](https://albumentations.ai/explore/transform/HistogramMatching/)
- [HueSaturationValue](https://albumentations.ai/explore/transform/HueSaturationValue/)
- [ISONoise](https://albumentations.ai/explore/transform/ISONoise/)
- [Illumination](https://albumentations.ai/explore/transform/Illumination/)
- [ImageCompression](https://albumentations.ai/explore/transform/ImageCompression/)
- [InvertImg](https://albumentations.ai/explore/transform/InvertImg/)
- [KSpaceSpikeNoise](https://albumentations.ai/explore/transform/KSpaceSpikeNoise/)
- [LensFlare](https://albumentations.ai/explore/transform/LensFlare/)
- [MedianBlur](https://albumentations.ai/explore/transform/MedianBlur/)
- [ModeFilter](https://albumentations.ai/explore/transform/ModeFilter/)
- [MotionBlur](https://albumentations.ai/explore/transform/MotionBlur/)
- [MultiplicativeNoise](https://albumentations.ai/explore/transform/MultiplicativeNoise/)
- [Normalize](https://albumentations.ai/explore/transform/Normalize/)
- [PhotoMetricDistort](https://albumentations.ai/explore/transform/PhotoMetricDistort/)
- [PixelDistributionAdaptation](https://albumentations.ai/explore/transform/PixelDistributionAdaptation/)
- [PlanckianJitter](https://albumentations.ai/explore/transform/PlanckianJitter/)
- [PlasmaBrightnessContrast](https://albumentations.ai/explore/transform/PlasmaBrightnessContrast/)
- [PlasmaShadow](https://albumentations.ai/explore/transform/PlasmaShadow/)
- [Posterize](https://albumentations.ai/explore/transform/Posterize/)
- [RGBShift](https://albumentations.ai/explore/transform/RGBShift/)
- [RandomBrightnessContrast](https://albumentations.ai/explore/transform/RandomBrightnessContrast/)
- [RandomFog](https://albumentations.ai/explore/transform/RandomFog/)
- [RandomGamma](https://albumentations.ai/explore/transform/RandomGamma/)
- [RandomGravel](https://albumentations.ai/explore/transform/RandomGravel/)
- [RandomRain](https://albumentations.ai/explore/transform/RandomRain/)
- [RandomShadow](https://albumentations.ai/explore/transform/RandomShadow/)
- [RandomSnow](https://albumentations.ai/explore/transform/RandomSnow/)
- [RandomSunFlare](https://albumentations.ai/explore/transform/RandomSunFlare/)
- [RandomToneCurve](https://albumentations.ai/explore/transform/RandomToneCurve/)
- [RicianNoise](https://albumentations.ai/explore/transform/RicianNoise/)
- [RingingOvershoot](https://albumentations.ai/explore/transform/RingingOvershoot/)
- [SaltAndPepper](https://albumentations.ai/explore/transform/SaltAndPepper/)
- [Sharpen](https://albumentations.ai/explore/transform/Sharpen/)
- [ShotNoise](https://albumentations.ai/explore/transform/ShotNoise/)
- [Solarize](https://albumentations.ai/explore/transform/Solarize/)
- [Spatter](https://albumentations.ai/explore/transform/Spatter/)
- [StochasticConvolution](https://albumentations.ai/explore/transform/StochasticConvolution/)
- [Superpixels](https://albumentations.ai/explore/transform/Superpixels/)
- [ToFloat](https://albumentations.ai/explore/transform/ToFloat/)
- [ToGray](https://albumentations.ai/explore/transform/ToGray/)
- [ToRGB](https://albumentations.ai/explore/transform/ToRGB/)
- [ToSepia](https://albumentations.ai/explore/transform/ToSepia/)
- [UnsharpMask](https://albumentations.ai/explore/transform/UnsharpMask/)
- [Vignetting](https://albumentations.ai/explore/transform/Vignetting/)
- [ZoomBlur](https://albumentations.ai/explore/transform/ZoomBlur/)

### Spatial-level transforms

| Transform                                                                                         | Image | Images | Mask | Masks | BBoxes (HBB) | BBoxes (OBB) | Keypoints | Volume | Volumes | Mask3D | Masks3D |
| ------------------------------------------------------------------------------------------------- | :---: | :----: | :--: | :---: | :----------: | :----------: | :-------: | :----: | :-----: | :----: | :-----: |
| [Affine](https://albumentations.ai/explore/transform/Affine/)                                     | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [AtLeastOneBBoxRandomCrop](https://albumentations.ai/explore/transform/AtLeastOneBBoxRandomCrop/) | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [BBoxSafeRandomCrop](https://albumentations.ai/explore/transform/BBoxSafeRandomCrop/)             | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [BBoxSubsetSafeRandomCrop](https://albumentations.ai/explore/transform/BBoxSubsetSafeRandomCrop/) | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [CenterCrop](https://albumentations.ai/explore/transform/CenterCrop/)                             | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [CoarseDropout](https://albumentations.ai/explore/transform/CoarseDropout/)                       | ✓     | ✓      | ✓    | ✓     | ✓            |              | ✓         | ✓      | ✓       | ✓      | ✓       |
| [ConstrainedCoarseDropout](https://albumentations.ai/explore/transform/ConstrainedCoarseDropout/) | ✓     |        | ✓    |       | ✓            |              | ✓         | ✓      | ✓       | ✓      | ✓       |
| [CopyAndPaste](https://albumentations.ai/explore/transform/CopyAndPaste/)                         | ✓     |        | ✓    | ✓     | ✓            |              | ✓         |        |         |        |         |
| [Crop](https://albumentations.ai/explore/transform/Crop/)                                         | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [CropAndPad](https://albumentations.ai/explore/transform/CropAndPad/)                             | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [CropNonEmptyMaskIfExists](https://albumentations.ai/explore/transform/CropNonEmptyMaskIfExists/) | ✓     |        | ✓    |       | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [D4](https://albumentations.ai/explore/transform/D4/)                                             | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [ElasticTransform](https://albumentations.ai/explore/transform/ElasticTransform/)                 | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [Erasing](https://albumentations.ai/explore/transform/Erasing/)                                   | ✓     | ✓      | ✓    | ✓     | ✓            |              | ✓         | ✓      | ✓       | ✓      | ✓       |
| [FrequencyMasking](https://albumentations.ai/explore/transform/FrequencyMasking/)                 | ✓     | ✓      | ✓    | ✓     | ✓            |              | ✓         | ✓      | ✓       | ✓      | ✓       |
| [GridDistortion](https://albumentations.ai/explore/transform/GridDistortion/)                     | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [GridDropout](https://albumentations.ai/explore/transform/GridDropout/)                           | ✓     | ✓      | ✓    | ✓     | ✓            |              | ✓         | ✓      | ✓       | ✓      | ✓       |
| [GridElasticDeform](https://albumentations.ai/explore/transform/GridElasticDeform/)               | ✓     | ✓      | ✓    | ✓     | ✓            |              | ✓         | ✓      | ✓       | ✓      | ✓       |
| [GridMask](https://albumentations.ai/explore/transform/GridMask/)                                 | ✓     | ✓      | ✓    | ✓     | ✓            |              | ✓         | ✓      | ✓       | ✓      | ✓       |
| [GuidedCoarseDropout](https://albumentations.ai/explore/transform/GuidedCoarseDropout/)           | ✓     | ✓      | ✓    | ✓     | ✓            |              | ✓         |        |         |        |         |
| [HorizontalFlip](https://albumentations.ai/explore/transform/HorizontalFlip/)                     | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [Lambda](https://albumentations.ai/explore/transform/Lambda/)                                     | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [LetterBox](https://albumentations.ai/explore/transform/LetterBox/)                               | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [LongestMaxSize](https://albumentations.ai/explore/transform/LongestMaxSize/)                     | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [MaskDropout](https://albumentations.ai/explore/transform/MaskDropout/)                           | ✓     |        | ✓    |       | ✓            |              | ✓         | ✓      | ✓       | ✓      | ✓       |
| [Morphological](https://albumentations.ai/explore/transform/Morphological/)                       | ✓     | ✓      | ✓    | ✓     | ✓            |              | ✓         | ✓      | ✓       | ✓      | ✓       |
| [Mosaic](https://albumentations.ai/explore/transform/Mosaic/)                                     | ✓     |        | ✓    | ✓     | ✓            | ✓            | ✓         |        |         |        |         |
| [NoOp](https://albumentations.ai/explore/transform/NoOp/)                                         | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [OpticalDistortion](https://albumentations.ai/explore/transform/OpticalDistortion/)               | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [OverlayElements](https://albumentations.ai/explore/transform/OverlayElements/)                   | ✓     |        | ✓    |       |              |              |           |        |         |        |         |
| [Pad](https://albumentations.ai/explore/transform/Pad/)                                           | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [PadIfNeeded](https://albumentations.ai/explore/transform/PadIfNeeded/)                           | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [Perspective](https://albumentations.ai/explore/transform/Perspective/)                           | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [PiecewiseAffine](https://albumentations.ai/explore/transform/PiecewiseAffine/)                   | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [PixelDropout](https://albumentations.ai/explore/transform/PixelDropout/)                         | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [PixelSpread](https://albumentations.ai/explore/transform/PixelSpread/)                           | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [RandomCrop](https://albumentations.ai/explore/transform/RandomCrop/)                             | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [RandomCropFromBorders](https://albumentations.ai/explore/transform/RandomCropFromBorders/)       | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [RandomCropNearBBox](https://albumentations.ai/explore/transform/RandomCropNearBBox/)             | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [RandomGridShuffle](https://albumentations.ai/explore/transform/RandomGridShuffle/)               | ✓     | ✓      | ✓    | ✓     | ✓            |              | ✓         | ✓      | ✓       | ✓      | ✓       |
| [RandomResizedCrop](https://albumentations.ai/explore/transform/RandomResizedCrop/)               | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [RandomRotate90](https://albumentations.ai/explore/transform/RandomRotate90/)                     | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [RandomScale](https://albumentations.ai/explore/transform/RandomScale/)                           | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [RandomSizedBBoxSafeCrop](https://albumentations.ai/explore/transform/RandomSizedBBoxSafeCrop/)   | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [RandomSizedCrop](https://albumentations.ai/explore/transform/RandomSizedCrop/)                   | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [Resize](https://albumentations.ai/explore/transform/Resize/)                                     | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [Rotate](https://albumentations.ai/explore/transform/Rotate/)                                     | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [SafeRotate](https://albumentations.ai/explore/transform/SafeRotate/)                             | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [ShiftScaleRotate](https://albumentations.ai/explore/transform/ShiftScaleRotate/)                 | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [SmallestMaxSize](https://albumentations.ai/explore/transform/SmallestMaxSize/)                   | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [SquareSymmetry](https://albumentations.ai/explore/transform/SquareSymmetry/)                     | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [ThinPlateSpline](https://albumentations.ai/explore/transform/ThinPlateSpline/)                   | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [TimeMasking](https://albumentations.ai/explore/transform/TimeMasking/)                           | ✓     | ✓      | ✓    | ✓     | ✓            |              | ✓         | ✓      | ✓       | ✓      | ✓       |
| [TimeReverse](https://albumentations.ai/explore/transform/TimeReverse/)                           | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [Transpose](https://albumentations.ai/explore/transform/Transpose/)                               | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [VerticalFlip](https://albumentations.ai/explore/transform/VerticalFlip/)                         | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [WaterRefraction](https://albumentations.ai/explore/transform/WaterRefraction/)                   | ✓     | ✓      | ✓    | ✓     | ✓            | ✓            | ✓         | ✓      | ✓       | ✓      | ✓       |
| [XYMasking](https://albumentations.ai/explore/transform/XYMasking/)                               | ✓     | ✓      | ✓    | ✓     | ✓            |              | ✓         | ✓      | ✓       | ✓      | ✓       |

### 3D transforms

| Transform                                                                               | Volume | Volumes | Mask3D | Masks3D | Keypoints |
| --------------------------------------------------------------------------------------- | :----: | :-----: | :----: | :-----: | :-------: |
| [Affine3D](https://albumentations.ai/explore/transform/Affine3D/)                       | ✓      | ✓       | ✓      | ✓       | ✓         |
| [Anisotropy3D](https://albumentations.ai/explore/transform/Anisotropy3D/)               | ✓      | ✓       |        |         |           |
| [CenterCrop3D](https://albumentations.ai/explore/transform/CenterCrop3D/)               | ✓      | ✓       | ✓      | ✓       | ✓         |
| [CoarseDropout3D](https://albumentations.ai/explore/transform/CoarseDropout3D/)         | ✓      | ✓       | ✓      | ✓       | ✓         |
| [CubicSymmetry](https://albumentations.ai/explore/transform/CubicSymmetry/)             | ✓      | ✓       | ✓      | ✓       | ✓         |
| [ElasticTransform3D](https://albumentations.ai/explore/transform/ElasticTransform3D/)   | ✓      | ✓       | ✓      | ✓       | ✓         |
| [Flip3D](https://albumentations.ai/explore/transform/Flip3D/)                           | ✓      | ✓       | ✓      | ✓       | ✓         |
| [GridShuffle3D](https://albumentations.ai/explore/transform/GridShuffle3D/)             | ✓      | ✓       | ✓      | ✓       | ✓         |
| [MotionArtifact](https://albumentations.ai/explore/transform/MotionArtifact/)           | ✓      | ✓       |        |         |           |
| [Pad3D](https://albumentations.ai/explore/transform/Pad3D/)                             | ✓      | ✓       | ✓      | ✓       | ✓         |
| [PadIfNeeded3D](https://albumentations.ai/explore/transform/PadIfNeeded3D/)             | ✓      | ✓       | ✓      | ✓       | ✓         |
| [RandomCrop3D](https://albumentations.ai/explore/transform/RandomCrop3D/)               | ✓      | ✓       | ✓      | ✓       | ✓         |
| [RandomResizedCrop3D](https://albumentations.ai/explore/transform/RandomResizedCrop3D/) | ✓      | ✓       | ✓      | ✓       | ✓         |
| [RandomRotate90_3D](https://albumentations.ai/explore/transform/RandomRotate90_3D/)     | ✓      | ✓       | ✓      | ✓       | ✓         |
| [Resize3D](https://albumentations.ai/explore/transform/Resize3D/)                       | ✓      | ✓       | ✓      | ✓       | ✓         |

### Semantic segmentation on the Inria dataset

![inria](https://habrastorage.org/webt/su/wa/np/suwanpeo6ww7wpwtobtrzd_cg20.jpeg)

### Medical imaging

![medical](https://habrastorage.org/webt/1i/fi/wz/1ifiwzy0lxetc4nwjvss-71nkw0.jpeg)

### Object detection and semantic segmentation on the Mapillary Vistas dataset

![vistas](https://habrastorage.org/webt/rz/-h/3j/rz-h3jalbxic8o_fhucxysts4tc.jpeg)

### Keypoints augmentation

<img src="https://habrastorage.org/webt/e-/6k/z-/e-6kz-fugp2heak3jzns3bc-r8o.jpeg" width=100%>

## Benchmark Results

These results cover the library versions listed below.

### Image Benchmark Results

### System Information

- Platform: macOS-15.1-arm64-arm-64bit
- Processor: arm
- CPU Count: 16
- Python Version: 3.12.8

### Benchmark Parameters

- Number of images: 2000
- Runs per transform: 5
- Max warmup iterations: 1000

### Library Versions

- albumentationsx: 2.0.8
- augly: 1.0.0
- imgaug: 0.4.0
- kornia: 0.8.0
- torchvision: 0.20.1

## Performance Comparison

Number shows how many uint8 images per second can be processed on one CPU thread. Larger is better.
The Speedup column shows how many times faster AlbumentationsX is compared to the fastest other
library for each transform.

| Transform            | albumentationsx<br>2.0.8 | augly<br>1.0.0 | imgaug<br>0.4.0 | kornia<br>0.8.0 | torchvision<br>0.20.1 | Speedup<br>(AlbX/fastest other) |
|:---------------------|:-------------------------|:---------------|:----------------|:----------------|:----------------------|:--------------------------------|
| Affine               | **1445 ± 9**             | -              | 1328 ± 16       | 248 ± 6         | 188 ± 2               | 1.09x                           |
| AutoContrast         | **1657 ± 13**            | -              | -               | 541 ± 8         | 344 ± 1               | 3.06x                           |
| Blur                 | **7657 ± 114**           | 386 ± 4        | 5381 ± 125      | 265 ± 11        | -                     | 1.42x                           |
| Brightness           | **11985 ± 455**          | 2108 ± 32      | 1076 ± 32       | 1127 ± 27       | 854 ± 13              | 5.68x                           |
| CLAHE                | **647 ± 4**              | -              | 555 ± 14        | 165 ± 3         | -                     | 1.17x                           |
| CenterCrop128        | **119293 ± 2164**        | -              | -               | -               | -                     | N/A                             |
| ChannelDropout       | **11534 ± 306**          | -              | -               | 2283 ± 24       | -                     | 5.05x                           |
| ChannelShuffle       | **6772 ± 109**           | -              | 1252 ± 26       | 1328 ± 44       | 4417 ± 234            | 1.53x                           |
| CoarseDropout        | **18962 ± 1346**         | -              | 1190 ± 22       | -               | -                     | 15.93x                          |
| ColorJitter          | **1020 ± 91**            | 418 ± 5        | -               | 104 ± 4         | 87 ± 1                | 2.44x                           |
| Contrast             | **12394 ± 363**          | 1379 ± 25      | 717 ± 5         | 1109 ± 41       | 602 ± 13              | 8.99x                           |
| CornerIllumination   | **484 ± 7**              | -              | -               | 452 ± 3         | -                     | 1.07x                           |
| Elastic              | 374 ± 2                  | -              | **395 ± 14**    | 1 ± 0           | 3 ± 0                 | 0.95x                           |
| Equalize             | **1236 ± 21**            | -              | 814 ± 11        | 306 ± 1         | 795 ± 3               | 1.52x                           |
| Erasing              | **27451 ± 2794**         | -              | -               | 1210 ± 27       | 3577 ± 49             | 7.67x                           |
| GaussianBlur         | **2350 ± 118**           | 387 ± 4        | 1460 ± 23       | 254 ± 5         | 127 ± 4               | 1.61x                           |
| GaussianIllumination | **720 ± 7**              | -              | -               | 436 ± 13        | -                     | 1.65x                           |
| GaussianNoise        | **315 ± 4**              | -              | 263 ± 9         | 125 ± 1         | -                     | 1.20x                           |
| Grayscale            | **32284 ± 1130**         | 6088 ± 107     | 3100 ± 24       | 1201 ± 52       | 2600 ± 23             | 5.30x                           |
| HSV                  | **1197 ± 23**            | -              | -               | -               | -                     | N/A                             |
| HorizontalFlip       | **14460 ± 368**          | 8808 ± 1012    | 9599 ± 495      | 1297 ± 13       | 2486 ± 107            | 1.51x                           |
| Hue                  | **1944 ± 64**            | -              | -               | 150 ± 1         | -                     | 12.98x                          |
| Invert               | **27665 ± 3803**         | -              | 3682 ± 79       | 2881 ± 43       | 4244 ± 30             | 6.52x                           |
| JpegCompression      | **1321 ± 33**            | 1202 ± 19      | 687 ± 26        | 120 ± 1         | 889 ± 7               | 1.10x                           |
| LinearIllumination   | 479 ± 5                  | -              | -               | **708 ± 6**     | -                     | 0.68x                           |
| MedianBlur           | **1229 ± 9**             | -              | 1152 ± 14       | 6 ± 0           | -                     | 1.07x                           |
| MotionBlur           | **3521 ± 25**            | -              | 928 ± 37        | 159 ± 1         | -                     | 3.79x                           |
| Normalize            | **1819 ± 49**            | -              | -               | 1251 ± 14       | 1018 ± 7              | 1.45x                           |
| OpticalDistortion    | **661 ± 7**              | -              | -               | 174 ± 0         | -                     | 3.80x                           |
| Pad                  | **48589 ± 2059**         | -              | -               | -               | 4889 ± 183            | 9.94x                           |
| Perspective          | **1206 ± 3**             | -              | 908 ± 8         | 154 ± 3         | 147 ± 5               | 1.33x                           |
| PlankianJitter       | **3221 ± 63**            | -              | -               | 2150 ± 52       | -                     | 1.50x                           |
| PlasmaBrightness     | **168 ± 2**              | -              | -               | 85 ± 1          | -                     | 1.98x                           |
| PlasmaContrast       | **145 ± 3**              | -              | -               | 84 ± 0          | -                     | 1.71x                           |
| PlasmaShadow         | 183 ± 5                  | -              | -               | **216 ± 5**     | -                     | 0.85x                           |
| Posterize            | **12979 ± 1121**         | -              | 3111 ± 95       | 836 ± 30        | 4247 ± 26             | 3.06x                           |
| RGBShift             | **3391 ± 104**           | -              | -               | 896 ± 9         | -                     | 3.79x                           |
| Rain                 | **2043 ± 115**           | -              | -               | 1493 ± 9        | -                     | 1.37x                           |
| RandomCrop128        | **111859 ± 1374**        | 45395 ± 934    | 21408 ± 622     | 2946 ± 42       | 31450 ± 249           | 2.46x                           |
| RandomGamma          | **12444 ± 753**          | -              | 3504 ± 72       | 230 ± 3         | -                     | 3.55x                           |
| RandomResizedCrop    | **4347 ± 37**            | -              | -               | 661 ± 16        | 837 ± 37              | 5.19x                           |
| Resize               | **3532 ± 67**            | 1083 ± 21      | 2995 ± 70       | 645 ± 13        | 260 ± 9               | 1.18x                           |
| Rotate               | **2912 ± 68**            | 1739 ± 105     | 2574 ± 10       | 256 ± 2         | 258 ± 4               | 1.13x                           |
| SaltAndPepper        | **629 ± 6**              | -              | -               | 480 ± 12        | -                     | 1.31x                           |
| Saturation           | **1596 ± 24**            | -              | 495 ± 3         | 155 ± 2         | -                     | 3.22x                           |
| Sharpen              | **2346 ± 10**            | -              | 1101 ± 30       | 201 ± 2         | 220 ± 3               | 2.13x                           |
| Shear                | **1299 ± 11**            | -              | 1244 ± 14       | 261 ± 1         | -                     | 1.04x                           |
| Snow                 | **611 ± 9**              | -              | -               | 143 ± 1         | -                     | 4.28x                           |
| Solarize             | **11756 ± 481**          | -              | 3843 ± 80       | 263 ± 6         | 1032 ± 14             | 3.06x                           |
| ThinPlateSpline      | **82 ± 1**               | -              | -               | 58 ± 0          | -                     | 1.41x                           |
| VerticalFlip         | **32386 ± 936**          | 16830 ± 1653   | 19935 ± 1708    | 2872 ± 37       | 4696 ± 161            | 1.62x                           |

## 🤝 Contribute

We thrive on community collaboration! AlbumentationsX wouldn't be the powerful augmentation library it is without contributions from developers like you. Please see our [Contributing Guide](https://github.com/albumentations-team/AlbumentationsX/blob/main/CONTRIBUTING.md) to get started. A huge **Thank You** 🙏 to everyone who contributes!

[![AlbumentationsX open-source contributors](https://contrib.rocks/image?repo=albumentations-team/AlbumentationsX)](https://github.com/albumentations-team/AlbumentationsX/graphs/contributors)

We look forward to your contributions to help make the AlbumentationsX ecosystem even better!

## 📜 License

See [Licensing](#licensing) for the two options.
The [AGPL text](https://github.com/albumentations-team/AlbumentationsX/blob/5bac7d5ae4b38f3c2a89c21dc088245613e9e99c/LICENSE),
[licensing history](https://github.com/albumentations-team/AlbumentationsX/blob/5bac7d5ae4b38f3c2a89c21dc088245613e9e99c/LICENSING.md),
and [third-party notices](https://github.com/albumentations-team/AlbumentationsX/blob/5bac7d5ae4b38f3c2a89c21dc088245613e9e99c/THIRD_PARTY_NOTICES.md)
record the applicable terms. Earlier releases retain the permissions that accompanied them.

## 📞 Contact

For bug reports and feature requests related to AlbumentationsX, please visit [GitHub Issues](https://github.com/albumentations-team/AlbumentationsX/issues). For questions, discussions, and community support, join our active communities on [Discord](https://discord.gg/AKPrrDYNAt), [Twitter](https://twitter.com/albumentations), [LinkedIn](https://www.linkedin.com/company/albumentations/), and [Reddit](https://www.reddit.com/r/Albumentations/). We're here to help with all things AlbumentationsX!

---

## 📫 Stay Connected

Never miss updates, tutorials, and tips from the AlbumentationsX team! [Subscribe to our newsletter](https://albumentations.ai/subscribe?utm_source=github&utm_medium=referral&utm_campaign=readme).
