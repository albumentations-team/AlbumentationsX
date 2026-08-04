# AlbumentationsX

[![PyPI version](https://badge.fury.io/py/albumentationsx.svg)](https://badge.fury.io/py/albumentationsx)
![CI](https://github.com/albumentations-team/AlbumentationsX/workflows/CI/badge.svg)
[![PyPI Downloads](https://img.shields.io/pypi/dm/albumentationsx.svg?label=PyPI%20downloads)](https://pypi.org/project/albumentationsx/)

> 📣 **Stay updated!** [Subscribe to our newsletter](https://albumentations.ai/subscribe?utm_source=github&utm_medium=referral&utm_campaign=readme) for the latest releases, tutorials, and tips.

[![License: AGPL-3.0-only](https://img.shields.io/badge/License-AGPL--3.0--only-blue.svg)](LICENSE)
[![Commercial License](https://img.shields.io/badge/Commercial_License-available-brightgreen)](https://albumentations.ai/pricing?utm_source=github&utm_medium=referral&utm_campaign=readme)

[![Docs](https://img.shields.io/badge/docs-albumentations.ai-blue)](https://albumentations.ai/docs/?utm_source=github&utm_medium=referral&utm_campaign=readme) [![Discord](https://img.shields.io/badge/Discord-join-7289da?logo=discord&logoColor=white)](https://discord.gg/AKPrrDYNAt) [![Twitter](https://img.shields.io/badge/Twitter-follow-1da1f2?logo=twitter&logoColor=white)](https://twitter.com/albumentations) [![LinkedIn](https://img.shields.io/badge/LinkedIn-connect-0077b5?logo=linkedin&logoColor=white)](https://www.linkedin.com/company/albumentations/) [![Reddit](https://img.shields.io/badge/Reddit-join-ff4500?logo=reddit&logoColor=white)](https://www.reddit.com/r/Albumentations/)

**AlbumentationsX** is a Python library for image augmentation. It provides high-performance, robust implementations and cutting-edge features for computer vision tasks. Image augmentation is used in deep learning and computer vision to increase the quality of trained models. The purpose of image augmentation is to create new training samples from the existing data.

## GitAds Sponsored

[![Sponsored by GitAds](https://gitads.dev/v1/ad-serve?source=albumentations-team/albumentationsx@github)](https://gitads.dev/v1/ad-track?source=albumentations-team/albumentationsx@github)

## 📢 Licensing: commercial use is allowed

**AlbumentationsX can be used in commercial projects under the AGPL.** The
current public repository is available under **AGPL-3.0-only**, an open-source
license. The AGPL permits commercial use subject to its terms.

Albumentations, LLC also offers separately negotiated commercial licenses with
alternative, scope-specific permissions for the versions and uses covered by
an executed agreement. A commercial license is an option when a team needs
terms different from the AGPL. It is not automatically required because a
project is commercial, proprietary, in production, or internal.

Which terms fit depends on the deployment facts, including modification,
combination, copying or conveyance, and network interaction. Support,
warranties, maintenance, and service levels are included only when an executed
agreement or order form expressly says so. See the [AGPL text](LICENSE),
[licensing details and history](LICENSING.md), and
[third-party notices](THIRD_PARTY_NOTICES.md).

### Quick Start

```bash
# Install AlbumentationsX with OpenCV
pip install albumentationsx[headless]

# Or if you already have OpenCV installed
pip install albumentationsx
```

```python
import albumentations as A

# Create your augmentation pipeline
transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])
```

For commercial licensing inquiries, please visit [our pricing page](https://albumentations.ai/pricing?utm_source=github&utm_medium=referral&utm_campaign=readme).

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

- [AlbumentationsX](#albumentationsx)
  - [Why AlbumentationsX](#why-albumentationsx)
  - [Table of contents](#table-of-contents)
  - [Authors](#authors)
    - [Current Maintainer](#current-maintainer)
    - [Emeritus Core Team Members](#emeritus-core-team-members)
  - [Installation](#installation)
  - [Documentation](#documentation)
  - [A simple example](#a-simple-example)
  - [List of augmentations](#list-of-augmentations)
    - [Pixel-level transforms](#pixel-level-transforms)
    - [Spatial-level transforms](#spatial-level-transforms)
  - [A few more examples of **augmentations**](#a-few-more-examples-of-augmentations)
    - [Semantic segmentation on the Inria dataset](#semantic-segmentation-on-the-inria-dataset)
    - [Medical imaging](#medical-imaging)
    - [Object detection and semantic segmentation on the Mapillary Vistas dataset](#object-detection-and-semantic-segmentation-on-the-mapillary-vistas-dataset)
    - [Keypoints augmentation](#keypoints-augmentation)
  - [Benchmarking results](#benchmark-results)
    - [System Information](#system-information)
    - [Benchmark Parameters](#benchmark-parameters)
    - [Library Versions](#library-versions)
  - [Performance Comparison](#performance-comparison)
  - [🤝 Contribute](#-contribute)
  - [📜 License](#-license)
  - [📞 Contact](#-contact)
  - [Citing](#citing)

## Authors

### Current Maintainer

[**Vladimir I. Iglovikov**](https://www.linkedin.com/in/iglovikov/) | [Kaggle Grandmaster](https://www.kaggle.com/iglovikov)

### Emeritus Core Team Members

[**Mikhail Druzhinin**](https://www.linkedin.com/in/mikhail-druzhinin-548229100/) | [Kaggle Expert](https://www.kaggle.com/dipetm)

[**Alex Parinov**](https://www.linkedin.com/in/alex-parinov/) | [Kaggle Master](https://www.kaggle.com/creafz)

[**Alexander Buslaev**](https://www.linkedin.com/in/al-buslaev/) | [Kaggle Master](https://www.kaggle.com/albuslaev)

[**Eugene Khvedchenya**](https://www.linkedin.com/in/cvtalks/) | [Kaggle Grandmaster](https://www.kaggle.com/bloodaxe)

## Installation

AlbumentationsX requires Python 3.10 or higher. To install the latest version from PyPI:

### Basic Installation

If you already have OpenCV installed (any variant), simply install AlbumentationsX:

```bash
pip install -U albumentationsx
```

### Installation with OpenCV

If you don't have OpenCV installed yet, choose the appropriate variant:

```bash
# For servers/Docker (no GUI support, lighter package)
pip install -U albumentationsx[headless]

# For local development with GUI support (cv2.imshow, etc.)
pip install opencv-python && pip install -U albumentationsx

# For OpenCV with extra algorithms (contrib modules)
pip install opencv-contrib-python && pip install -U albumentationsx

# For contrib + headless
pip install -U albumentationsx[contrib-headless]
```

**Note:** AlbumentationsX works with any OpenCV variant:

- `opencv-python` (full version with GUI)
- `opencv-python-headless` (no GUI, smaller size)
- `opencv-contrib-python` (with extra modules)
- `opencv-contrib-python-headless` (contrib + headless)

Choose the one that fits your needs. The library will detect whichever is installed.

Other installation options are described in the [documentation](https://albumentations.ai/docs/1-introduction/installation/?utm_source=github&utm_medium=referral&utm_campaign=readme).

## Documentation

The full documentation is available at **[https://albumentations.ai/docs/](https://albumentations.ai/docs/?utm_source=github&utm_medium=referral&utm_campaign=readme)**.

For AI-assisted augmentation review, AlbumentationsX can also be used through MCP-capable hosts such as Claude Desktop,
Cursor, Claude Code, and Codex. The community
[AlbumentationsX MCP integration](docs/integrations/mcp.md) lets assistants inspect transforms, validate pipelines,
render bounded local preview batches, compare preview runs, collect concrete feedback, and export reproducible
AlbumentationsX pipelines.

## A simple example

```python
import albumentations as A
import cv2

# Declare an augmentation pipeline
transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

# Read an image with OpenCV and convert it to the RGB colorspace
image = cv2.imread("image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Augment an image
transformed = transform(image=image)
transformed_image = transformed["image"]
```

AlbumentationsX collects anonymous usage statistics to improve the library. This can be disabled with `ALBUMENTATIONS_OFFLINE=1` or `ALBUMENTATIONS_NO_TELEMETRY=1`.

## List of augmentations

### Pixel-level transforms

Pixel-level transforms will change just an input image and will leave any additional targets such as masks, bounding boxes, and keypoints unchanged. For volumetric data (a volume and 3D masks), these transforms are applied independently to each slice along the Z-axis (depth dimension), maintaining consistency across the volume. The list of pixel-level transforms:

- [AdditiveNoise](https://albumentations.ai/explore/transform/AdditiveNoise/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [AdvancedBlur](https://albumentations.ai/explore/transform/AdvancedBlur/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [AnnotationArtifacts](https://albumentations.ai/explore/transform/AnnotationArtifacts/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [AtmosphericFog](https://albumentations.ai/explore/transform/AtmosphericFog/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [AutoContrast](https://albumentations.ai/explore/transform/AutoContrast/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [Blur](https://albumentations.ai/explore/transform/Blur/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [CLAHE](https://albumentations.ai/explore/transform/CLAHE/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [ChannelDropout](https://albumentations.ai/explore/transform/ChannelDropout/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [ChannelShuffle](https://albumentations.ai/explore/transform/ChannelShuffle/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [ChannelSwap](https://albumentations.ai/explore/transform/ChannelSwap/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [ChromaticAberration](https://albumentations.ai/explore/transform/ChromaticAberration/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [ColorJitter](https://albumentations.ai/explore/transform/ColorJitter/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [Colorize](https://albumentations.ai/explore/transform/Colorize/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [Defocus](https://albumentations.ai/explore/transform/Defocus/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [Dithering](https://albumentations.ai/explore/transform/Dithering/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [Downscale](https://albumentations.ai/explore/transform/Downscale/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [Emboss](https://albumentations.ai/explore/transform/Emboss/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [Enhance](https://albumentations.ai/explore/transform/Enhance/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [Equalize](https://albumentations.ai/explore/transform/Equalize/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [ExposureMatching](https://albumentations.ai/explore/transform/ExposureMatching/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [FDA](https://albumentations.ai/explore/transform/FDA/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [FancyPCA](https://albumentations.ai/explore/transform/FancyPCA/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [FilmGrain](https://albumentations.ai/explore/transform/FilmGrain/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [FromFloat](https://albumentations.ai/explore/transform/FromFloat/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [GaussNoise](https://albumentations.ai/explore/transform/GaussNoise/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [GaussianBlur](https://albumentations.ai/explore/transform/GaussianBlur/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [GlassBlur](https://albumentations.ai/explore/transform/GlassBlur/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [HEStain](https://albumentations.ai/explore/transform/HEStain/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [Halftone](https://albumentations.ai/explore/transform/Halftone/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [HistogramMatching](https://albumentations.ai/explore/transform/HistogramMatching/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [HueSaturationValue](https://albumentations.ai/explore/transform/HueSaturationValue/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [ISONoise](https://albumentations.ai/explore/transform/ISONoise/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [Illumination](https://albumentations.ai/explore/transform/Illumination/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [ImageCompression](https://albumentations.ai/explore/transform/ImageCompression/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [InvertImg](https://albumentations.ai/explore/transform/InvertImg/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [LensFlare](https://albumentations.ai/explore/transform/LensFlare/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [MedianBlur](https://albumentations.ai/explore/transform/MedianBlur/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [ModeFilter](https://albumentations.ai/explore/transform/ModeFilter/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [MotionBlur](https://albumentations.ai/explore/transform/MotionBlur/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [MultiplicativeNoise](https://albumentations.ai/explore/transform/MultiplicativeNoise/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [Normalize](https://albumentations.ai/explore/transform/Normalize/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [PhotoMetricDistort](https://albumentations.ai/explore/transform/PhotoMetricDistort/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [PixelDistributionAdaptation](https://albumentations.ai/explore/transform/PixelDistributionAdaptation/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [PlanckianJitter](https://albumentations.ai/explore/transform/PlanckianJitter/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [PlasmaBrightnessContrast](https://albumentations.ai/explore/transform/PlasmaBrightnessContrast/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [PlasmaShadow](https://albumentations.ai/explore/transform/PlasmaShadow/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [Posterize](https://albumentations.ai/explore/transform/Posterize/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [RGBShift](https://albumentations.ai/explore/transform/RGBShift/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [RandomBrightnessContrast](https://albumentations.ai/explore/transform/RandomBrightnessContrast/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [RandomFog](https://albumentations.ai/explore/transform/RandomFog/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [RandomGamma](https://albumentations.ai/explore/transform/RandomGamma/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [RandomGravel](https://albumentations.ai/explore/transform/RandomGravel/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [RandomRain](https://albumentations.ai/explore/transform/RandomRain/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [RandomShadow](https://albumentations.ai/explore/transform/RandomShadow/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [RandomSnow](https://albumentations.ai/explore/transform/RandomSnow/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [RandomSunFlare](https://albumentations.ai/explore/transform/RandomSunFlare/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [RandomToneCurve](https://albumentations.ai/explore/transform/RandomToneCurve/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [RingingOvershoot](https://albumentations.ai/explore/transform/RingingOvershoot/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [SaltAndPepper](https://albumentations.ai/explore/transform/SaltAndPepper/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [Sharpen](https://albumentations.ai/explore/transform/Sharpen/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [ShotNoise](https://albumentations.ai/explore/transform/ShotNoise/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [Solarize](https://albumentations.ai/explore/transform/Solarize/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [Spatter](https://albumentations.ai/explore/transform/Spatter/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [Superpixels](https://albumentations.ai/explore/transform/Superpixels/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [TextImage](https://albumentations.ai/explore/transform/TextImage/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [ToFloat](https://albumentations.ai/explore/transform/ToFloat/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [ToGray](https://albumentations.ai/explore/transform/ToGray/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [ToRGB](https://albumentations.ai/explore/transform/ToRGB/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [ToSepia](https://albumentations.ai/explore/transform/ToSepia/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [UnsharpMask](https://albumentations.ai/explore/transform/UnsharpMask/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [Vignetting](https://albumentations.ai/explore/transform/Vignetting/?utm_source=github&utm_medium=referral&utm_campaign=readme)
- [ZoomBlur](https://albumentations.ai/explore/transform/ZoomBlur/?utm_source=github&utm_medium=referral&utm_campaign=readme)

### Spatial-level transforms

Spatial-level transforms will simultaneously change both an input image as well as additional targets such as masks, bounding boxes, and keypoints. For volumetric data (a volume and 3D masks), these transforms are applied independently to each slice along the Z-axis (depth dimension), maintaining consistency across the volume. The following table shows which additional targets are supported by each transform:

- Volume: 3D array of shape (D, H, W) or (D, H, W, C) where D is depth, H is height, W is width, and C is number of channels (optional)
- Mask3D: Binary or multi-class 3D mask of shape (D, H, W) where each slice represents segmentation for the corresponding volume slice

| Transform                                                                                         | Image | Mask | BBoxes (HBB) | BBoxes (OBB) | Keypoints | Volume | Mask3D |
| ------------------------------------------------------------------------------------------------- | :---: | :--: | :----------: | :----------: | :-------: | :----: | :----: |
| [Affine](https://albumentations.ai/explore/transform/Affine/?utm_source=github&utm_medium=referral&utm_campaign=readme)                                     | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [AtLeastOneBBoxRandomCrop](https://albumentations.ai/explore/transform/AtLeastOneBBoxRandomCrop/?utm_source=github&utm_medium=referral&utm_campaign=readme) | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [BBoxSafeRandomCrop](https://albumentations.ai/explore/transform/BBoxSafeRandomCrop/?utm_source=github&utm_medium=referral&utm_campaign=readme)             | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [CenterCrop](https://albumentations.ai/explore/transform/CenterCrop/?utm_source=github&utm_medium=referral&utm_campaign=readme)                             | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [CoarseDropout](https://albumentations.ai/explore/transform/CoarseDropout/?utm_source=github&utm_medium=referral&utm_campaign=readme)                       | ✓     | ✓    | ✓            |              | ✓         | ✓      | ✓      |
| [ConstrainedCoarseDropout](https://albumentations.ai/explore/transform/ConstrainedCoarseDropout/?utm_source=github&utm_medium=referral&utm_campaign=readme) | ✓     | ✓    | ✓            |              | ✓         | ✓      | ✓      |
| [CopyAndPaste](https://albumentations.ai/explore/transform/CopyAndPaste/?utm_source=github&utm_medium=referral&utm_campaign=readme)                         | ✓     | ✓    | ✓            |              | ✓         |        |        |
| [Crop](https://albumentations.ai/explore/transform/Crop/?utm_source=github&utm_medium=referral&utm_campaign=readme)                                         | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [CropAndPad](https://albumentations.ai/explore/transform/CropAndPad/?utm_source=github&utm_medium=referral&utm_campaign=readme)                             | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [CropNonEmptyMaskIfExists](https://albumentations.ai/explore/transform/CropNonEmptyMaskIfExists/?utm_source=github&utm_medium=referral&utm_campaign=readme) | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [D4](https://albumentations.ai/explore/transform/D4/?utm_source=github&utm_medium=referral&utm_campaign=readme)                                             | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [ElasticTransform](https://albumentations.ai/explore/transform/ElasticTransform/?utm_source=github&utm_medium=referral&utm_campaign=readme)                 | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [Erasing](https://albumentations.ai/explore/transform/Erasing/?utm_source=github&utm_medium=referral&utm_campaign=readme)                                   | ✓     | ✓    | ✓            |              | ✓         | ✓      | ✓      |
| [FrequencyMasking](https://albumentations.ai/explore/transform/FrequencyMasking/?utm_source=github&utm_medium=referral&utm_campaign=readme)                 | ✓     | ✓    | ✓            |              | ✓         | ✓      | ✓      |
| [GridDistortion](https://albumentations.ai/explore/transform/GridDistortion/?utm_source=github&utm_medium=referral&utm_campaign=readme)                     | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [GridDropout](https://albumentations.ai/explore/transform/GridDropout/?utm_source=github&utm_medium=referral&utm_campaign=readme)                           | ✓     | ✓    | ✓            |              | ✓         | ✓      | ✓      |
| [GridElasticDeform](https://albumentations.ai/explore/transform/GridElasticDeform/?utm_source=github&utm_medium=referral&utm_campaign=readme)               | ✓     | ✓    | ✓            |              | ✓         | ✓      | ✓      |
| [GridMask](https://albumentations.ai/explore/transform/GridMask/?utm_source=github&utm_medium=referral&utm_campaign=readme)                                 | ✓     | ✓    | ✓            |              | ✓         | ✓      | ✓      |
| [HorizontalFlip](https://albumentations.ai/explore/transform/HorizontalFlip/?utm_source=github&utm_medium=referral&utm_campaign=readme)                     | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [Lambda](https://albumentations.ai/explore/transform/Lambda/?utm_source=github&utm_medium=referral&utm_campaign=readme)                                     | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [LetterBox](https://albumentations.ai/explore/transform/LetterBox/?utm_source=github&utm_medium=referral&utm_campaign=readme)                               | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [LongestMaxSize](https://albumentations.ai/explore/transform/LongestMaxSize/?utm_source=github&utm_medium=referral&utm_campaign=readme)                     | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [MaskDropout](https://albumentations.ai/explore/transform/MaskDropout/?utm_source=github&utm_medium=referral&utm_campaign=readme)                           | ✓     | ✓    | ✓            |              | ✓         | ✓      | ✓      |
| [Morphological](https://albumentations.ai/explore/transform/Morphological/?utm_source=github&utm_medium=referral&utm_campaign=readme)                       | ✓     | ✓    | ✓            |              | ✓         | ✓      | ✓      |
| [Mosaic](https://albumentations.ai/explore/transform/Mosaic/?utm_source=github&utm_medium=referral&utm_campaign=readme)                                     | ✓     | ✓    | ✓            | ✓            | ✓         |        |        |
| [NoOp](https://albumentations.ai/explore/transform/NoOp/?utm_source=github&utm_medium=referral&utm_campaign=readme)                                         | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [OpticalDistortion](https://albumentations.ai/explore/transform/OpticalDistortion/?utm_source=github&utm_medium=referral&utm_campaign=readme)               | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [OverlayElements](https://albumentations.ai/explore/transform/OverlayElements/?utm_source=github&utm_medium=referral&utm_campaign=readme)                   | ✓     | ✓    |              |              |           |        |        |
| [Pad](https://albumentations.ai/explore/transform/Pad/?utm_source=github&utm_medium=referral&utm_campaign=readme)                                           | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [PadIfNeeded](https://albumentations.ai/explore/transform/PadIfNeeded/?utm_source=github&utm_medium=referral&utm_campaign=readme)                           | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [Perspective](https://albumentations.ai/explore/transform/Perspective/?utm_source=github&utm_medium=referral&utm_campaign=readme)                           | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [PiecewiseAffine](https://albumentations.ai/explore/transform/PiecewiseAffine/?utm_source=github&utm_medium=referral&utm_campaign=readme)                   | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [PixelDropout](https://albumentations.ai/explore/transform/PixelDropout/?utm_source=github&utm_medium=referral&utm_campaign=readme)                         | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [PixelSpread](https://albumentations.ai/explore/transform/PixelSpread/?utm_source=github&utm_medium=referral&utm_campaign=readme)                           | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [RandomCrop](https://albumentations.ai/explore/transform/RandomCrop/?utm_source=github&utm_medium=referral&utm_campaign=readme)                             | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [RandomCropFromBorders](https://albumentations.ai/explore/transform/RandomCropFromBorders/?utm_source=github&utm_medium=referral&utm_campaign=readme)       | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [RandomCropNearBBox](https://albumentations.ai/explore/transform/RandomCropNearBBox/?utm_source=github&utm_medium=referral&utm_campaign=readme)             | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [RandomGridShuffle](https://albumentations.ai/explore/transform/RandomGridShuffle/?utm_source=github&utm_medium=referral&utm_campaign=readme)               | ✓     | ✓    | ✓            |              | ✓         | ✓      | ✓      |
| [RandomResizedCrop](https://albumentations.ai/explore/transform/RandomResizedCrop/?utm_source=github&utm_medium=referral&utm_campaign=readme)               | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [RandomRotate90](https://albumentations.ai/explore/transform/RandomRotate90/?utm_source=github&utm_medium=referral&utm_campaign=readme)                     | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [RandomScale](https://albumentations.ai/explore/transform/RandomScale/?utm_source=github&utm_medium=referral&utm_campaign=readme)                           | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [RandomSizedBBoxSafeCrop](https://albumentations.ai/explore/transform/RandomSizedBBoxSafeCrop/?utm_source=github&utm_medium=referral&utm_campaign=readme)   | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [RandomSizedCrop](https://albumentations.ai/explore/transform/RandomSizedCrop/?utm_source=github&utm_medium=referral&utm_campaign=readme)                   | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [Resize](https://albumentations.ai/explore/transform/Resize/?utm_source=github&utm_medium=referral&utm_campaign=readme)                                     | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [Rotate](https://albumentations.ai/explore/transform/Rotate/?utm_source=github&utm_medium=referral&utm_campaign=readme)                                     | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [SafeRotate](https://albumentations.ai/explore/transform/SafeRotate/?utm_source=github&utm_medium=referral&utm_campaign=readme)                             | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [ShiftScaleRotate](https://albumentations.ai/explore/transform/ShiftScaleRotate/?utm_source=github&utm_medium=referral&utm_campaign=readme)                 | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [SmallestMaxSize](https://albumentations.ai/explore/transform/SmallestMaxSize/?utm_source=github&utm_medium=referral&utm_campaign=readme)                   | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [SquareSymmetry](https://albumentations.ai/explore/transform/SquareSymmetry/?utm_source=github&utm_medium=referral&utm_campaign=readme)                     | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [ThinPlateSpline](https://albumentations.ai/explore/transform/ThinPlateSpline/?utm_source=github&utm_medium=referral&utm_campaign=readme)                   | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [TimeMasking](https://albumentations.ai/explore/transform/TimeMasking/?utm_source=github&utm_medium=referral&utm_campaign=readme)                           | ✓     | ✓    | ✓            |              | ✓         | ✓      | ✓      |
| [TimeReverse](https://albumentations.ai/explore/transform/TimeReverse/?utm_source=github&utm_medium=referral&utm_campaign=readme)                           | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [Transpose](https://albumentations.ai/explore/transform/Transpose/?utm_source=github&utm_medium=referral&utm_campaign=readme)                               | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [VerticalFlip](https://albumentations.ai/explore/transform/VerticalFlip/?utm_source=github&utm_medium=referral&utm_campaign=readme)                         | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [WaterRefraction](https://albumentations.ai/explore/transform/WaterRefraction/?utm_source=github&utm_medium=referral&utm_campaign=readme)                   | ✓     | ✓    | ✓            | ✓            | ✓         | ✓      | ✓      |
| [XYMasking](https://albumentations.ai/explore/transform/XYMasking/?utm_source=github&utm_medium=referral&utm_campaign=readme)                               | ✓     | ✓    | ✓            |              | ✓         | ✓      | ✓      |

### 3D transforms

3D transforms operate on volumetric data. Spatial transforms can also modify associated 3D masks and keypoints, while
volume-intensity transforms leave those targets unchanged.

Where:

- Volume: 3D array of shape (D, H, W) or (D, H, W, C) where D is depth, H is height, W is width, and C is number of channels (optional)
- Mask3D: Binary or multi-class 3D mask of shape (D, H, W) where each slice represents segmentation for the corresponding volume slice

| Transform                                                                           | Volume | Mask3D | Keypoints |
| ----------------------------------------------------------------------------------- | :----: | :----: | :-------: |
| [Anisotropy3D](https://albumentations.ai/explore/transform/Anisotropy3D/)           | ✓      |        |           |
| [CenterCrop3D](https://albumentations.ai/explore/transform/CenterCrop3D/)           | ✓      | ✓      | ✓         |
| [CoarseDropout3D](https://albumentations.ai/explore/transform/CoarseDropout3D/)     | ✓      | ✓      | ✓         |
| [CubicSymmetry](https://albumentations.ai/explore/transform/CubicSymmetry/)         | ✓      | ✓      | ✓         |
| [Flip3D](https://albumentations.ai/explore/transform/Flip3D/)                       | ✓      | ✓      | ✓         |
| [GridShuffle3D](https://albumentations.ai/explore/transform/GridShuffle3D/)         | ✓      | ✓      | ✓         |
| [Pad3D](https://albumentations.ai/explore/transform/Pad3D/)                         | ✓      | ✓      | ✓         |
| [PadIfNeeded3D](https://albumentations.ai/explore/transform/PadIfNeeded3D/)         | ✓      | ✓      | ✓         |
| [RandomCrop3D](https://albumentations.ai/explore/transform/RandomCrop3D/)           | ✓      | ✓      | ✓         |
| [RandomRotate90_3D](https://albumentations.ai/explore/transform/RandomRotate90_3D/) | ✓      | ✓      | ✓         |
| [Resize3D](https://albumentations.ai/explore/transform/Resize3D/)                   | ✓      | ✓      | ✓         |

## A few more examples of **augmentations**

### Semantic segmentation on the Inria dataset

![inria](https://habrastorage.org/webt/su/wa/np/suwanpeo6ww7wpwtobtrzd_cg20.jpeg)

### Medical imaging

![medical](https://habrastorage.org/webt/1i/fi/wz/1ifiwzy0lxetc4nwjvss-71nkw0.jpeg)

### Object detection and semantic segmentation on the Mapillary Vistas dataset

![vistas](https://habrastorage.org/webt/rz/-h/3j/rz-h3jalbxic8o_fhucxysts4tc.jpeg)

### Keypoints augmentation

<img src="https://habrastorage.org/webt/e-/6k/z-/e-6kz-fugp2heak3jzns3bc-r8o.jpeg" width=100%>

## Benchmark Results

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

We thrive on community collaboration! AlbumentationsX wouldn't be the powerful augmentation library it is without contributions from developers like you. Please see our [Contributing Guide](CONTRIBUTING.md) to get started. A huge **Thank You** 🙏 to everyone who contributes!

[![AlbumentationsX open-source contributors](https://contrib.rocks/image?repo=albumentations-team/AlbumentationsX)](https://github.com/albumentations-team/AlbumentationsX/graphs/contributors)

We look forward to your contributions to help make the AlbumentationsX ecosystem even better!

## 📜 License

The current public repository is licensed under **AGPL-3.0-only**. Earlier
AlbumentationsX releases retain the license terms recorded in the
[licensing details and history](LICENSING.md). The AGPL permits commercial use subject
to its terms.

For alternative, scope-specific terms from Albumentations, LLC, visit the
[pricing page](https://albumentations.ai/pricing?utm_source=github&utm_medium=referral&utm_campaign=readme).
The [AGPL text](LICENSE), [licensing details](LICENSING.md), and
[third-party notices](THIRD_PARTY_NOTICES.md) contain the complete
repository-level details.

## 📞 Contact

For bug reports and feature requests related to AlbumentationsX, please visit [GitHub Issues](https://github.com/albumentations-team/AlbumentationsX/issues). For questions, discussions, and community support, join our active communities on [Discord](https://discord.gg/AKPrrDYNAt), [Twitter](https://twitter.com/albumentations), [LinkedIn](https://www.linkedin.com/company/albumentations/), and [Reddit](https://www.reddit.com/r/Albumentations/). We're here to help with all things AlbumentationsX!

## Citing

If you find this library useful for your research, please consider citing [Albumentations: Fast and Flexible Image Augmentations](https://www.mdpi.com/2078-2489/11/2/125):

```bibtex
@Article{info11020125,
    AUTHOR = {Buslaev, Alexander and Iglovikov, Vladimir I. and Khvedchenya, Eugene and Parinov, Alex and Druzhinin, Mikhail and Kalinin, Alexandr A.},
    TITLE = {Albumentations: Fast and Flexible Image Augmentations},
    JOURNAL = {Information},
    VOLUME = {11},
    YEAR = {2020},
    NUMBER = {2},
    ARTICLE-NUMBER = {125},
    URL = {https://www.mdpi.com/2078-2489/11/2/125},
    ISSN = {2078-2489},
    DOI = {10.3390/info11020125}
}
```

---

## 📫 Stay Connected

Never miss updates, tutorials, and tips from the AlbumentationsX team! [Subscribe to our newsletter](https://albumentations.ai/subscribe?utm_source=github&utm_medium=referral&utm_campaign=readme).

<!-- GitAds-Verify: 99ZXCN5GQ9CQN3QEMO5H4RAOI8C5YTKV -->
