# Human3.6M Dataset

## For evaluations without self-supervised fine-tuning of VideoPose3D

Please visit [the official website](http://vision.imar.ro/human3.6m/) and download the files for 2D and 3D joints as well as the videos. You need files of the "_S11 Walking 1_" sequence which can be found in the following tgz files.

- `Train/subject/Videos_S11.tgz`
- `Train/subject/Poses_D2_Positions_S11.tgz`
- `Train/subject/Poses_D3_Positions_S11.tgz`

By unpacking these files you should have the following directory structure.

```
── third_party
   └── H36M
         └── S11
            ├── MyPoseFeatures
            │   ├── D2_Positions
            │   │   ├── Walking 1.54138969.cdf
            │   │   ├── Waiting 1.55011271.cdf
            │   │   ├── Walking 1.58860488.cdf
            │   │   └── Walking 1.60457274.cdf
            │   └── D3_Positions
            │       └── Walking 1.cdf
            └── Videos
               ├── Walking 1.54138969.mp4
               ├── Walking 1.55011271.mp4
               ├── Walking 1.58860488.mp4
               └── Walking 1.60457274.mp4
```

## For self-supervised fine-tuning of VideoPose3D

To fine-tune VideoPose3D, we use both GAFA and Human3.6M dataset. Please follow [`third_party/VideoPose3D/DATASETS.md`](third_party/VideoPose3D/DATASETS.md) and prepare the data for VideoPose3D. As a result, you need to have two NPZ files as follows.

```
── third_party
   └── H36M
         ├── S11
         │  ├── MyPoseFeatures
         │  │   ├── D2_Positions
         │  │   │   ├── Walking 1.54138969.cdf
         │  │   │   ├── Waiting 1.55011271.cdf
         │  │   │   ├── Walking 1.58860488.cdf
         │  │   │   └── Walking 1.60457274.cdf
         │  │   └── D3_Positions
         │  │       └── Walking 1.cdf
         │  └── Videos
         │     ├── Walking 1.54138969.mp4
         │     ├── Walking 1.55011271.mp4
         │     ├── Walking 1.58860488.mp4
         │     └── Walking 1.60457274.mp4
         └── retrain
            ├── data_2d_h36m_detectron_pt_coco.npz
            └── data_3d_h36m.npz
```

When you are ready for the dataset, you can run

```
sh ./prepare_h36m.sh ./data/A099_P099_G099 99 99 99 noise_99_0 H36M ./third_party/H36M poses_from_vp3d ./data/A099_P099_G099/poses_from_vp3d
```
