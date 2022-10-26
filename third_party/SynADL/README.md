# KIST SynADL Dataset

This directory contains the files from [KIST SynADL](https://ai4robot.github.io/ElderSim/) dataset.  You can either 1) prepare a subset of the dataset to test our code or 2) prepare the entire dataset from the official site.

## 1. Subset for testing

On the courtesy of the authors of KIST SynADL dataset, we can provide a subset of KIST SynADL dataset to test our code without downloading the entire dataset. You can download [this ZIP file](https://drive.google.com/file/d/1Njq4gk3DedWiIknGW75f2-g-QAZ9PFzz/view?usp=sharing) and extract the files as follows.

Please be reminded that the copyright of the files in the ZIP file belongs to the authors of KIST SynADL and the original license of KIST SynADL applies to the files.  That is, by downloading the zip file you agree [KIST SynADL EULA](https://ai4robot.github.io/ElderSim/).

The directory after unzipping the files should be as follows.  Please notice that the video files are not included in the ZIP file.  To include videos in the visualization of the results, please download the videos from the original website and copy `A023_P102_G003_C02[1357].avi` in `RGB` directory as shown in the next section.

```
./SynADL/
├── Openpose
│   ├── 2DJ
│   │   ├── A023_P102_G003_C021.json
│   │   ├── A023_P102_G003_C023.json
│   │   ├── A023_P102_G003_C025.json
│   │   └── A023_P102_G003_C027.json
│   └── 3DJ
│       ├── A023_P102_G003_C021.json
│       ├── A023_P102_G003_C023.json
│       ├── A023_P102_G003_C025.json
│       └── A023_P102_G003_C027.json
└── README.md
```


When you are ready for the dataset, you can run
```
sh ./prepare_synadl.sh ./third_party/SynADL 23 102 3 ./data/
```
as instructed in [`../../README.md`](../../README.md).


## 2. Original dataset

Please download `OpenPose.zip` and `RGB.zip` from [KIST SynADL website](https://ai4robot.github.io/ElderSim/).  After unpacking the downloaded files, you should have the following directory structure.

```
./SynADL/
├── Openpose
│   ├── 2DJ
│   │   ├── A001_P101_G001_C001.json
│   │   ├── ...
│   │   └── A055_P115_G004_C028.json
│   └── 3DJ
│       ├── A001_P101_G001_C001.json
│       ├── ...
│       └── A055_P115_G004_C028.json
├── RGB
│   ├── A001_P101_G001_C001.avi
│   ├── ...
│   └── A055_P115_G004_C028.avi
└── README.md
```

