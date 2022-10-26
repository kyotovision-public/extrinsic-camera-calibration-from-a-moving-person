import numpy as np
import yaml
import json
import os

import glob
import cv2

from cv2 import aruco
import sys
import pickle as pkl
import pandas as pd
import pycalib

sys.path.append("./")
from util import (
    save_cam,
    OP_KEY,
    project_cv2_nan,
    get_ccs,
    save_joint,
    save_skelton_w_op,
    invRT,
    convert_coco_to_op,
    convert_coco_to_op_score,
    triangulate_all,
    project_cv2,
    get_poses_from_vp3d,
)

NFRAME = 2850
import argument

COCO19_KEY = {
    "Neck": 0,
    "Nose": 1,
    "MidHip": 2,
    "LShoulder": 3,
    "LElbow": 4,
    "LWrist": 5,
    "LHip": 6,
    "LKnee": 7,
    "LAnkle": 8,
    "RShoulder": 9,
    "RElbow": 10,
    "RWrist": 11,
    "RHip": 12,
    "RKnee": 13,
    "RAnkle": 14,
    "LEye": 15,
    "LEar": 16,
    "REye": 17,
    "REar": 18,
}


# def panoptic_main(aid,pid,gid,camera_ids,gt_dir,prefix_panoptic,width, height):
def panoptic_main(
    aid,
    pid,
    gid,
    camera_ids,
    gt_dir,
    third_party_dir,
    prefix,
    original_path,
    target_dir,
    width,
    height,
):

    Ks = []
    distCoefs = []
    Rw2cs = []
    tw2cs = []

    Ks = []
    distCoefs = []
    scale = 0.01
    path_3d_joint = f"{original_path}/hdPose3d_stage1_coco19"
    intrinsic = f"{original_path}/calibration_161029_flute1.json"

    with open(intrinsic, "r") as fp:
        P = json.load(fp)
        target = ["00_03", "00_12", "00_18", "00_22"]

        for target_i in target:
            for camera_i in P["cameras"]:
                if camera_i["name"] == target_i:
                    cam_info = camera_i

                    Ks.append(np.array(cam_info["K"]))
                    distCoefs.append(np.array(cam_info["distCoef"]))
                    tw2cs.append(np.array(cam_info["t"]))

                    Rw2cs.append(np.array(cam_info["R"]))
        Ks = np.array(Ks)
        distCoefs = np.array(distCoefs)
        tw2cs = np.array(tw2cs) * scale
        Rw2cs = np.array(Rw2cs)

        Json_3DJOINTs = sorted(glob.glob(f"{path_3d_joint}/*.json"))

        # Flute scene

        ss = 4 * 60 + 25  # 4:25
        to = ss + 95  # 1:35
        offset = 180
        frame_ss = ss * 30  # 30fps
        frame_to = to * 30  # 30fps
        target_poses = Json_3DJOINTs[frame_ss - offset : frame_to - offset - 1]
        X_gt_w = []
        for fname in target_poses:

            with open(fname, "r") as fp:
                P = json.load(fp)
                X_gt_w_i = P["bodies"][0]["joints19"]

                X_gt_w_i = np.array(X_gt_w_i) * scale
                X_gt_w_i = X_gt_w_i.reshape(19, -1)
                X_gt_w_i = X_gt_w_i[:, :3]

                X_gt_w.append(X_gt_w_i)
        X_gt_w = np.array(X_gt_w)  ## (2849, 19, 3)

        file_poses = sorted(glob.glob(f"{prefix}/poses_from_vp3d/data_*.npz"))
        file_scores = sorted(glob.glob(f"{prefix}/poses_from_vp3d/Camera_*mp4.npz"))
        from collections import deque

        # print(file_scores)
        new_file_poses = deque(file_poses)
        new_file_scores = deque(file_scores)
        new_file_poses.rotate(1)
        new_file_scores.rotate(1)
        # print(new_file_poses)
        # print(new_file_scores)
        x2d_coco, x2d_coco_scores = get_poses_from_vp3d(
            third_party_dir, list(new_file_poses), list(new_file_scores), NFRAME
        )

        Nc, Nf, Nj, _ = x2d_coco.shape
        X2d_op = []
        x2d_op_scroes = []
        for i in range(Nc):
            X2d_op.append(convert_coco_to_op(x2d_coco[i]))
            x2d_op_scroes.append(convert_coco_to_op_score(x2d_coco_scores[i]))

        X2d_op = np.array(X2d_op)
        x2d_op_scroes = np.array(x2d_op_scroes)

        X3d_w_op_new = np.full(
            (X_gt_w.shape[0], 25, X_gt_w.shape[2]), np.nan, dtype=np.float32
        )
        vis_mask = np.full((Nc, X_gt_w.shape[0], 25), False, dtype=bool)
        for k, v in COCO19_KEY.items():
            X3d_w_op_new[:, OP_KEY[k], :] = X_gt_w[:, COCO19_KEY[k], :]
            vis_mask[:, :, OP_KEY[k]] = True

        X3d_c_op_new = get_ccs(Rw2cs, tw2cs, X3d_w_op_new, vis_mask)

        x2d_repro_op_new = project_cv2_nan(
            Rw2cs, tw2cs, Ks, X3d_w_op_new, width, height
        )

        save_joint(
            f"{target_dir}/2d_joint", X2d_op, x2d_op_scroes, aid, pid, gid, camera_ids
        )

        save_joint(
            f"{gt_dir}/2d_joint", x2d_repro_op_new, vis_mask, aid, pid, gid, camera_ids
        )

        save_joint(
            f"{gt_dir}/3d_joint", X3d_c_op_new, vis_mask, aid, pid, gid, camera_ids
        )

        save_cam(Rw2cs, tw2cs, Ks, distCoefs, gt_dir, gid, camera_ids)
        save_skelton_w_op(X3d_w_op_new, gt_dir, gid)
        save_joint(
            f"{target_dir}/2d_joint", X2d_op, x2d_op_scroes, aid, pid, gid, camera_ids
        )
        save_cam(Rw2cs, tw2cs, Ks, distCoefs, target_dir, gid, camera_ids)
        save_skelton_w_op(X3d_w_op_new, target_dir, gid)


if __name__ == "__main__":

    args = argument.parse_args()
    PREFIX = args.prefix + "/" + args.target
    AID = args.aid
    PID = args.pid
    GID = args.gid
    DATASET = args.dataset
    with open("./config/config.yaml") as file:
        config = yaml.safe_load(file.read())

    camera_ids = config[DATASET]["camera_ids"]
    width = config[DATASET]["width"]
    height = config[DATASET]["height"]

    prefix = args.prefix
    gt_dir = f"{prefix}/gt_subset"
    third_party_dir = f"./third_party/{DATASET}"
    original_path = args.src_original
    # poses_from_vp3d  = f"{third_party_dir}/poses_from_vp3d"
    AID = args.aid
    PID = args.pid
    GID = args.gid
    target_dir = f"{prefix}/{args.target}"

    os.makedirs(f"{gt_dir}/2d_joint", exist_ok=True)
    os.makedirs(f"{gt_dir}/3d_joint", exist_ok=True)
    os.makedirs(f"{target_dir}/2d_joint", exist_ok=True)

    panoptic_main(
        AID,
        PID,
        GID,
        camera_ids,
        gt_dir,
        third_party_dir,
        prefix,
        original_path,
        target_dir,
        width,
        height,
    )
