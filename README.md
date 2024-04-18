## Geometric3DBBox

The code has been tested on PyTorch 3.8.0
## Installation
First, clone this repository and the git submodules:
```
git clone --recurse-submodules [https://github.com/ayesha-ishaq/Geometric3DBBox.git](https://github.com/ayesha-ishaq/Geometric3DBBox.git)
```

### Conda environment
Basic installation:
```
conda create -n geom3d python==3.8.11
conda activate geom3d
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install nuscenes-devkit matplotlib pandas 
```
Other installation:
Follow installation guidelines for [Lang SAM](https://github.com/luca-medeiros/lang-segment-anything.git) and [MMRotate](https://github.com/open-mmlab/mmrotate.git)

### NuScenes Dataset
Download NuScenes dataset [here](https://www.nuscenes.org/download).

### Inference demo:
To run inference on any validation point cloud use[<pre> demo.ipynb </pre>](demo.ipynb), in the file replace the <pre> dataset_dir </pre> with the path to your NuScenes folder, and set the <pre> cfg </pre> to the [configuration file](oriented_rcnn_r50_fpn_1x_dota_le90.py), lastly set the <pre> checkpoint </pre> to Oriented R-CNN weights path.

### Acknowledgement
The implementation of Geometric3DBBox relies on resources from <a href="https://github.com/luca-medeiros/lang-segment-anything.git">Lang SAM</a>, <a href="https://github.com/open-mmlab/mmrotate.git">MMRotate</a>, and <a href="https://github.com/nutonomy/nuscenes-devkit.git">NuScenes dev kit</a>.
