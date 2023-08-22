# Semalign

## Overall

- Calibration in pixel wise level. Semalgin uses image sementic-segmentation as spvnas and 3D lidar points segmentation as sdcnet.

## Index

- [News](#News)
- [Dependancy](#Dependancy)
- [Install](#Install)
- [Parameter](#Parameter)
- [Demo](#Demo)
- [Results](#Results)

---
## News
[2022-07] release semalign ver 0.1 (base line )


## Dependancy

- 1.1 SDCNET
  PyTorch implementation of our CVPR2019 paper (oral) on achieving state-of-the-art semantic segmentation results using Deeplabv3-Plus like architecture with a WideResNet38 trunk. We present a video prediction-based methodology to scale up training sets by synthesizing new training samples and propose a novel label relaxation technique to make training objectives robust to label noise.

- 1.2 SPVNAS
  SPVNAS achieves state-of-the-art performance on the SemanticKITTI leaderboard (as of July 2020) and outperforms MinkowskiNet with 3x speedup, 8x MACs reduction.

## Install


```
(in workspace folder)
$ git clone <semalign>
$ docker build -t semalign -f semalign/docker/Dockerfile .
```
- wegiht :
- data :

## Demo

```
python maino.py --demo-image YOUR_IMG --snapshot ./pretrained_models/cityscapes_best.pth --save-dir YOUR_SAVE_DIR
```

## Results

- Kitti

![before](https://user-images.githubusercontent.com/44966311/183828010-ddbe8319-c8a6-4aba-9266-1e591535b388.jpg)


---
