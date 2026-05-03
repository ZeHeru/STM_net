# STM_net in PyTorch

PyTorch implementation for reconstructing pristine molecular orbital images from
scanning tunneling microscope (STM) images.

Paper: [Reconstructing Pristine Molecular Orbitals from Scanning Tunneling
Microscope Images via Artificial Intelligence Approaches](https://doi.org/10.1021/jacsau.5c00310),
JACS Au, 2025.

## Prerequisites

- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started

### Installation

- Clone this repo:

```bash
git clone https://github.com/ZeHeru/STM_net.git
cd STM_net
```

- Install [PyTorch](https://pytorch.org) and the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

If you use Conda, create an environment first and then install the requirements:

```bash
conda create -n stm-net python=3.10
conda activate stm-net
python -m pip install -r requirements.txt
```

### Data Layout

For paired training, use `--dataset_mode aligned`. Each image in the phase
directory should concatenate the input image A and target image B horizontally:

```text
datasets/your_dataset/
  train/
    sample_001.png  # [A | B]
    sample_002.png
  test/
    sample_101.png
```

Use `--direction AtoB` when the left half is the STM image and the right half is
the target orbital image. Add `--co` to enable centroid-orientation alignment
before resizing/cropping.

### Training

Example U-Net training command:

```bash
python train.py \
  --dataroot datasets/your_dataset \
  --name stm_net_unet \
  --model unet \
  --netG standard_unet \
  --dataset_mode aligned \
  --direction AtoB \
  --input_nc 1 \
  --output_nc 1 \
  --load_size 286 \
  --crop_size 256 \
  --batch_size 16 \
  --n_epochs 500 \
  --n_epochs_decay 500 \
  --lr 0.0002 \
  --display_id -1
```

Example pix2pix training command:

```bash
python train.py \
  --dataroot datasets/your_dataset \
  --name stm_net_pix2pix \
  --model pix2pix \
  --netG unet_256 \
  --dataset_mode aligned \
  --direction AtoB \
  --input_nc 1 \
  --output_nc 1 \
  --display_id -1
```

Checkpoints are written to `checkpoints/<experiment_name>/`.

### Inference

Run inference on one image with a trained generator:

```bash
python test.py \
  --dataroot path/to/stm_image.png \
  --name stm_net_unet \
  --model test \
  --netG standard_unet \
  --input_nc 1 \
  --output_nc 1 \
  --epoch latest \
  --results_dir results/
```

Results are written to `results/<experiment_name>/`.

## Citation

If you use this repository, please cite:

```bibtex
@article{Zhu2025STMNet,
  author = {Zhu, Yu and Xue, Renjie and Ren, Hao and Chen, Yicheng and Yan, Wenjie and Wu, Bingzheng and Duan, Sai and Zhang, Haiming and Chi, Lifeng and Xu, Xin},
  title = {Reconstructing Pristine Molecular Orbitals from Scanning Tunneling Microscope Images via Artificial Intelligence Approaches},
  journal = {JACS Au},
  year = {2025},
  volume = {5},
  number = {7},
  pages = {3163--3170},
  doi = {10.1021/jacsau.5c00310}
}
```
