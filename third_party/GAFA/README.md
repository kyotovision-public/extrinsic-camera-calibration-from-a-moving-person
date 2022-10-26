# GAFA Dataset

Please visit [the official website](https://vision.ist.i.kyoto-u.ac.jp/research/gafa/) and download "Lab" sequence. By unpacking the downloaded ZIP file, you should have the following directory structure. We use the last 400 frames of cameras 1, 2, 5, 6, and 7 of subscenario 2 in `1013_2` sequence for evaluation.

```
third_party
└── GAFA
    ├── GoPro_2700_linear_intrinsic.npz
    ├── README.md
    ├── lab
    │   └── 1013_2
    │       ├── Camera_1_2
    │       │   ├── 000000.jpg
    │       │   ├── ...
    │       │   └── 000529.jpg
    │       ├── Camera_1_2.pkl
    │       ├── Camera_2_2
    │       │   ├── 000000.jpg
    │       │   ├── ...
    │       │   └── 000529.jpg
    │       ├── Camera_2_2.pkl
    │       ├── Camera_5_2
    │       │   ├── 000000.jpg
    │       │   ├── ...
    │       │   └── 000529.jpg
    │       ├── Camera_5_2.pkl
    │       ├── Camera_6_2
    │       │   ├── 000000.jpg
    │       │   ├── ...
    │       │   └── 000529.jpg
    │       ├── Camera_6_2.pkl
    │       ├── Camera_7_2
    │       │   ├── 000000.jpg
    │       │   ├── ...
    │       │   └── 000529.jpg
    │       └── Camera_7_2.pkl
    ├── manual_pts.csv
    └── prepare_GAFA.py
```

When you are ready for the dataset, you can run

```
sh ./prepare_gafa.sh ./data/A077_P077_G077 77 77 77 noise_77_0 GAFA ./third_party/GAFA/lab/1013_2  ./data/A077_P077_G077/poses_from_vp3d
```
