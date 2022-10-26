# CMU Panoptic Dataset

Please visit [the official website](http://domedb.perception.cs.cmu.edu/) and download the files for 2D and 3D joints as well as the videos. We use `161029_flute1` sequence. You can find them [here](http://domedb.perception.cs.cmu.edu/161029_flute1.html). By downloading the sequence with the official toolbox, you have the following files.

```
third_party/
└── Panoptic
    ├── 161029_flute1
    │   ├── calibration_161029_flute1.json
    │   ├── hdFace3d.tar
    │   ├── hdHand3d.tar
    │   ├── hdPose3d_stage1_coco19.tar
    │   ├── hdVideos
    │   │   ├── hd_00_00.mp4
    │   │   ├── ...
    │   │   └── hd_00_30.mp4
    │   └── vgaVideos
    │       └── ...
    └── README.md
```

When you are ready for the dataset, you can run

```
sh ./prepare_panoptic.sh ./data/A088_P088_G088 88 88 88 noise_88_0 Panoptic ./third_party/Panoptic/161029_flute1 ./data/A088_P088_G088/poses_from_vp3d
```
