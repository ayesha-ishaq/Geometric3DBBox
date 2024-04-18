# Geometric3DBBox

Geometric3DBBox is a Python toolkit for 3D bounding box estimation using geometric deep learning models. The code has been tested with PyTorch 3.8.0.

## Installation

To install the Geometric3DBBox repository and its dependencies, follow these steps:

### Clone the Repository

Clone the repository and all required submodules using the following command:
```bash
git clone --recurse-submodules https://github.com/ayesha-ishaq/Geometric3DBBox.git
```

### Set Up the Environment

#### Conda Environment

Create and activate a new Conda environment:
```bash
conda create -n geom3d python==3.8.11
conda activate geom3d
```

Install PyTorch and related packages:
```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Install additional Python packages:
```bash
pip install nuscenes-devkit matplotlib pandas open3d
```

#### Additional Installation

For additional functionality, follow the installation guidelines for:
- [Lang SAM](https://github.com/luca-medeiros/lang-segment-anything.git)
- [MMRotate](https://github.com/open-mmlab/mmrotate.git)

### NuScenes Dataset

Download the NuScenes dataset from the [official website](https://www.nuscenes.org/download).

## Usage

### Inference Demo

To run inference on a validation point cloud, use the Jupyter notebook provided:
```bash
demo.ipynb
```
In the notebook, replace `dataset_dir` with the path to your NuScenes folder, set `cfg` to your configuration file (e.g., `oriented_rcnn_r50_fpn_1x_dota_le90.py`), and set `checkpoint` to the path of your Oriented R-CNN weights.

## Acknowledgements

This implementation of Geometric3DBBox uses resources and tools from:
- [Lang SAM](https://github.com/luca-medeiros/lang-segment-anything.git)
- [MMRotate](https://github.com/open-mmlab/mmrotate.git)
- [NuScenes dev kit](https://github.com/nutonomy/nuscenes-devkit.git)
``
