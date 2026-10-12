# MPFA-DETR: Multi-Directional Perception and High-Low Frequency Aggregation for Small Object Detection in UAV Images

**Official PyTorch Implementation of MPFA-DETR**







**Repository:** https://github.com/Xoneop/MPFA-DETR

## Overview

Small object detection in unmanned aerial vehicle (UAV) imagery remains challenging due to large scale variations, dense object distributions, complex backgrounds, and weak visual cues of tiny targets.

To address these challenges, we propose **MPFA-DETR**, an efficient DETR-based framework that enhances small-object representation through multi-directional perception, multi-scale spatial modeling, state-space feature interaction, and adaptive high–low frequency aggregation.

MPFA-DETR collaboratively captures frequency, directional, and scale-aware information while preserving fine-grained visual details. Extensive experiments demonstrate its effectiveness across multiple UAV object detection benchmarks.

## Abstract

Small object detection in unmanned aerial vehicle (UAV) imagery remains challenging due to large scale variations, dense object distributions, complex backgrounds, and weak visual cues of tiny targets. To address these issues, we propose MPFA-DETR, an efficient DETR-based framework that enhances small-object representation by jointly modeling long-range dependencies, directional spatial context, and complementary frequency information. The proposed method collaboratively captures frequency, directional, and scale-aware cues, where frequency modulation and direction-aware aggregation emphasize discriminative fine-grained responses, while multi-scale spatial modeling provides complementary contextual information under different receptive fields. Meanwhile, channel-collaborative selective scanning strengthens cross-channel interaction and leverages state-space modeling to capture both local structures and long-range dependencies. Residual feature propagation further preserves fine-grained information, while a dedicated post-wavelet high–low frequency decomposition and adaptive selection formulation dynamically balances high-frequency edge-texture details and low-frequency semantic structures. Extensive experiments on VisDrone, UAVOD-10, and AI-TOD demonstrate that MPFA-DETR achieves competitive detection accuracy and computational efficiency across diverse UAV detection scenarios.

## Highlights

The main characteristics of MPFA-DETR include:

- **Multi-Directional Perception:** Enhances fine-grained object representations by capturing directional spatial context and discriminative feature responses.
- **Frequency Modulation:** Exploits complementary frequency-domain information to strengthen feature representations for small objects.
- **Multi-Scale Spatial Modeling:** Captures contextual information under different receptive fields to handle significant object-scale variations.
- **Channel-Collaborative Selective Scanning:** Leverages state-space modeling to strengthen cross-channel interactions and capture both local structures and long-range dependencies.
- **Residual Feature Propagation:** Preserves fine-grained information throughout feature transformation and aggregation.
- **Adaptive High–Low Frequency Aggregation:** Uses wavelet-based decomposition and adaptive selection to balance high-frequency edge-texture information and low-frequency semantic structures.

## Method

MPFA-DETR is developed as an efficient DETR-based framework for UAV small object detection.

The proposed framework incorporates several complementary mechanisms to improve fine-grained representation learning.

### Multi-Directional Perception and Spatial Modeling

Frequency modulation and direction-aware aggregation jointly emphasize discriminative responses in challenging UAV imagery. Multi-scale spatial modeling further captures contextual information across different receptive fields, allowing the network to represent objects with varying scales.

### Channel-Collaborative Selective Scanning

Channel-collaborative selective scanning strengthens interactions across feature channels while exploiting state-space modeling to capture local structural information and long-range dependencies.

### Residual Feature Propagation

Residual feature propagation is incorporated to preserve fine-grained feature information and facilitate effective representation learning.

### High–Low Frequency Aggregation

A dedicated post-wavelet high–low frequency decomposition and adaptive selection mechanism is introduced to dynamically balance high-frequency edge-texture details and low-frequency semantic structures.

These complementary components enable MPFA-DETR to improve small-object representation in complex aerial scenarios.

## Experimental Environment

All experiments were conducted on an **NVIDIA GeForce RTX 4090 GPU** under Ubuntu 22.04.

### Software and Hardware Configuration

| Component               | Configuration           |
| ----------------------- | ----------------------- |
| Operating System        | Ubuntu 22.04            |
| Python                  | 3.10                    |
| PyTorch                 | 2.1.2                   |
| CUDA                    | 11.8                    |
| GPU                     | NVIDIA GeForce RTX 4090 |
| Backbone                | ResNet-18 / ResNet-50   |
| Batch Size              | 4                       |
| Maximum Epochs          | 500                     |
| Early Stopping          | Enabled                 |
| Early Stopping Patience | 20                      |

### Model Variants

Two variants of MPFA-DETR were designed:

- **MPFA-DETR (ResNet-18):** Used for comparisons on other datasets, ablation studies, and visualization analysis.
- **MPFA-DETR (ResNet-50):** A variant employing ResNet-50 as its backbone.

The framework was trained with a batch size of 4 for up to 500 epochs. An early stopping strategy with a patience value of 20 was adopted.

## Installation

### 1. Clone the Repository

```
git clone https://github.com/Xoneop/MPFA-DETR.git
cd MPFA-DETR
```

### 2. Create the Conda Environment

```
conda create -n mpfa-detr python=3.10 -y
conda activate mpfa-detr
```

### 3. Install PyTorch

Install PyTorch 2.1.2 with CUDA 11.8 support:

```
pip install torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install Additional Dependencies

If the repository provides a `requirements.txt` file, install the remaining dependencies using:

```
pip install -r requirements.txt
```

Please ensure that the dependency versions are compatible with PyTorch 2.1.2 and CUDA 11.8.

### 5. Verify the Environment

Run the following command to verify the PyTorch and CUDA installation:

```
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

Expected output for the specified environment:

```
PyTorch: 2.1.2+cu118
CUDA: 11.8
CUDA Available: True
GPU: NVIDIA GeForce RTX 4090
```

**Note:** An NVIDIA driver compatible with CUDA 11.8 is required. The expected output assumes successful installation and GPU detection.

## Datasets

We evaluate MPFA-DETR on multiple UAV object detection benchmarks.

### VisDrone

VisDrone is a widely used UAV imagery benchmark containing challenging scenes with large scale variations, dense object distributions, and complex backgrounds.

Dataset website: https://github.com/VisDrone/VisDrone-Dataset

### AI-TOD

AI-TOD is an aerial image object detection benchmark particularly challenging for tiny object detection.

### UAVOD-10

UAVOD-10 is used to evaluate object detection performance in UAV imagery under different scenes and target scales.

### UAVVaste

UAVVaste is included as an additional experimental benchmark for evaluating the detection performance of MPFA-DETR.

Please prepare each dataset according to its official annotations and the data-loading configuration used in the repository.

## Experimental Results

We report Average Precision (AP), AP at IoU 0.50 (AP50), and AP at IoU 0.75 (AP75) to evaluate detection performance.

### 1. Results on VisDrone

**Overall performance comparison of MPFA-DETR and state-of-the-art detectors on the VisDrone dataset.**

| Model                | AP       | AP50     | AP75     |
| -------------------- | -------- | -------- | -------- |
| CEASC (2023)         | 28.7     | 50.7     | 28.4     |
| YOLC (2024)          | 28.9     | 51.4     | 28.3     |
| FBRT-YOLO (2025)     | 30.1     | 48.4     | 31.7     |
| UAV-DETR (2025)      | 31.8     | 51.4     | 33.0     |
| **MPFA-DETR (Ours)** | **32.3** | **51.8** | **33.7** |

MPFA-DETR achieves the best performance among the listed methods, obtaining **32.3 AP, 51.8 AP50, and 33.7 AP75**.

Compared with UAV-DETR, MPFA-DETR improves AP, AP50, and AP75 by **0.5, 0.4, and 0.7 percentage points**, respectively.

These results demonstrate the effectiveness of MPFA-DETR in handling challenging UAV detection scenarios.

### 2. Results on AI-TOD

**Overall performance comparison between MPFA-DETR and UAV-DETR on the AI-TOD dataset.**

| Model                | AP       | AP50     | AP75      |
| -------------------- | -------- | -------- | --------- |
| UAV-DETR (2025)      | 11.4     | 29.4     | 0.06      |
| **MPFA-DETR (Ours)** | **15.4** | **37.2** | **10.36** |

MPFA-DETR achieves **15.4 AP and 37.2 AP50**, outperforming UAV-DETR by **4.0 AP and 7.8 AP50**.

The experimental results indicate that MPFA-DETR provides improved detection performance on challenging aerial tiny-object detection scenarios.

### 3. Results on UAVOD-10

**Overall performance comparison between MPFA-DETR and UAV-DETR on the UAVOD-10 dataset.**

| Model                | AP       | AP50     | AP75     |
| -------------------- | -------- | -------- | -------- |
| UAV-DETR (2025)      | 26.8     | 45.8     | 27.0     |
| **MPFA-DETR (Ours)** | **27.8** | **48.6** | **28.2** |

MPFA-DETR obtains **27.8 AP, 48.6 AP50, and 28.2 AP75**, achieving improvements of **1.0, 2.8, and 1.2 percentage points**, respectively, over UAV-DETR.

The results further demonstrate the effectiveness of the proposed framework across different UAV object detection benchmarks.

### 4. Results on UAVVaste

**Performance comparison of MPFA-DETR on the UAVVaste dataset.**

| Model            | AP       | AP50     | mAP75    |
| ---------------- | -------- | -------- | -------- |
| FBRT-YOLO (2025) | 47.5     | 78.7     | **53.7** |
| UAV-DETR (2025)  | **49.1** | 80.8     | —        |
| MPFA-DETR (Ours) | 48.5     | **81.6** | 52.7     |

MPFA-DETR achieves the highest AP50 of **81.6** among the compared methods, surpassing UAV-DETR by 0.8 percentage points.

Although its overall AP is slightly lower than UAV-DETR, MPFA-DETR maintains competitive detection performance and demonstrates effectiveness on this additional benchmark.

## Training

MPFA-DETR adopts a DETR-based detection framework with two backbone configurations.

The reported training setup uses:

| Parameter               | Value                 |
| ----------------------- | --------------------- |
| Backbone options        | ResNet-18 / ResNet-50 |
| Batch size              | 4                     |
| Maximum epochs          | 500                   |
| Early stopping patience | 20                    |
| GPU                     | RTX 4090              |
| PyTorch                 | 2.1.2                 |
| CUDA                    | 11.8                  |

The actual training commands depend on the configuration files and training entry points included in the repository.

The corresponding commands should be documented here together with the dataset paths and model configuration files to enable reproducibility.

## Evaluation

The detection performance is evaluated using standard Average Precision metrics.

- **AP:** Average Precision averaged over the standard IoU thresholds.
- **AP50:** Average Precision at IoU = 0.50.
- **AP75:** Average Precision at IoU = 0.75.

The experimental results for different benchmarks are summarized in the tables above.

Evaluation should use the corresponding dataset annotations, checkpoint files, and model configurations.

## Visualization

Qualitative visualization is useful for analyzing the performance of MPFA-DETR under challenging UAV scenarios, including:

- Small and tiny objects
- Dense object distributions
- Complex backgrounds
- Significant scale variations
- Weak visual cues

Detection visualization examples and architectural illustrations can be added to this section.

## Model Weights

Pretrained checkpoints and corresponding evaluation configurations will be documented here when available.

| Model     | Backbone  | Checkpoint  |
| --------- | --------- | ----------- |
| MPFA-DETR | ResNet-18 | To be added |
| MPFA-DETR | ResNet-50 | To be added |

## Citation

If you find MPFA-DETR useful in your research, please consider citing our work.

```
@misc{MPFADETR,
  title = {MPFA-DETR: Multi-Directional Perception and High-Low Frequency Aggregation for Small Object Detection in UAV Images},
  note = {Publication details to be updated}
}
```

The complete citation information will be updated when available.

## Acknowledgements

We appreciate the contributions of the open-source computer vision community and the authors of the object detection frameworks and benchmark datasets used in this research.

## Contact

For questions, suggestions, or discussions, please open an issue in the GitHub repository:

https://github.com/Xoneop/MPFA-DETR

**MPFA-DETR: Multi-Directional Perception and High-Low Frequency Aggregation for Small Object Detection in UAV Images**
