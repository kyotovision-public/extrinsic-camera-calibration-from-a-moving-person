from math import dist
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
import copy
from scipy.spatial.transform import Rotation

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
    invRT_batch,
    convert_h36m_to_op,
)

vp3d_path = os.path.abspath(os.path.join("./third_party/VideoPose3D"))
if vp3d_path not in sys.path:
    sys.path.append(vp3d_path)


import argument


from common.h36m_dataset import h36m_cameras_extrinsic_params
from common.h36m_dataset import h36m_cameras_intrinsic_params

# PATH_CAMERAS = f"./../../data/scene/real/h36m"
TARGET_SUBJECT = "S11"
NFRAME = 1637
SCENE = "Walking 1"


def h36m_main(
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

    # Cmaeras
    cameras = copy.deepcopy(h36m_cameras_extrinsic_params)
    cameras_intrinsics = copy.deepcopy(h36m_cameras_intrinsic_params)

    cam_mats = []
    distCoefs = []
    R_c2w = []
    t_c2w = []
    for camera_i, intrinsic_i in zip(cameras[TARGET_SUBJECT], cameras_intrinsics):
        orientation = camera_i["orientation"]
        translation = camera_i["translation"]
        R_c2w.append(
            Rotation.from_quat(
                [orientation[1], orientation[2], orientation[3], orientation[0]]
            ).as_matrix()
        )
        t_c2w.append(np.array(translation)[:, None] / 1000)
        cam_mat = np.eye(3)
        cam_mat[0, 0] = intrinsic_i["focal_length"][0]
        cam_mat[1, 1] = intrinsic_i["focal_length"][1]
        cam_mat[0, 2] = intrinsic_i["center"][0]
        cam_mat[1, 2] = intrinsic_i["center"][1]
        distCoef = np.array(
            [
                intrinsic_i["radial_distortion"][0],
                intrinsic_i["radial_distortion"][1],
                intrinsic_i["tangential_distortion"][0],
                intrinsic_i["tangential_distortion"][1],
                intrinsic_i["radial_distortion"][2],
            ]
        )
        cam_mats.append(cam_mat)
        distCoefs.append(distCoef)
    R_c2w = np.array(R_c2w)

    t_c2w = np.array(t_c2w)
    R_w2c, t_w2c = invRT_batch(R_c2w, t_c2w)
    t_w2c = t_w2c[:, :, None]
    cam_mats = np.array(cam_mats)
    distCoefs = np.array(distCoefs)
    file_poses = sorted(glob.glob(f"{prefix}/poses_from_vp3d/data_*.npz"))
    file_scores = sorted(glob.glob(f"{prefix}/poses_from_vp3d/*mp4.npz"))
    x2d_coco, x2d_coco_scores = get_poses_from_vp3d(
        third_party_dir, file_poses, file_scores, NFRAME
    )

    files3djoints = f"{original_path}/retrain/data_3d_h36m.npz"

    # GT 3D joints
    data3d = np.load(files3djoints, allow_pickle=True)
    X_gt_w = data3d["positions_3d"].item()[TARGET_SUBJECT][SCENE]

    X2d_op = []
    x2d_op_scroes = []
    print(x2d_coco[0].shape)
    Nc = len(camera_ids)
    for i in range(Nc):
        X2d_op.append(convert_coco_to_op(x2d_coco[i]))
        x2d_op_scroes.append(convert_coco_to_op_score(x2d_coco_scores[i]))

    X2d_op = np.array(X2d_op)
    x2d_op_scroes = np.array(x2d_op_scroes)

    X3d_w_op = convert_h36m_to_op(X_gt_w)
    vis_mask = [(np.sum(~np.isnan(X3d_w_op), axis=2) > 0) for i in range(Nc)]
    vis_mask = np.array(vis_mask)
    X3d_c_op_new = get_ccs(R_w2c, t_w2c, X3d_w_op, vis_mask)

    x2d_repro_op_new = project_cv2(R_w2c, t_w2c, cam_mats, X3d_w_op, width, height)
    # x2d_repro_op_new = project_cv2(Rs_w2c, ts_w2c, Ks, dist_all,
    #                                X3d_w_op_new,width,height)
    save_joint(
        f"{target_dir}/2d_joint", X2d_op, x2d_op_scroes, aid, pid, gid, camera_ids
    )
    save_joint(f"{gt_dir}/3d_joint", X3d_c_op_new, vis_mask, aid, pid, gid, camera_ids)
    save_joint(
        f"{gt_dir}/2d_joint", x2d_repro_op_new, vis_mask, aid, pid, gid, camera_ids
    )
    save_cam(R_w2c, t_w2c, cam_mats, distCoefs, gt_dir, gid, camera_ids)
    save_cam(R_w2c, t_w2c, cam_mats, distCoefs, target_dir, gid, camera_ids)
    save_skelton_w_op(X3d_w_op, gt_dir, gid)
    save_skelton_w_op(X3d_w_op, target_dir, gid)


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
    targret_dir = f"{prefix}/{args.target}"

    os.makedirs(f"{gt_dir}/2d_joint", exist_ok=True)
    os.makedirs(f"{gt_dir}/3d_joint", exist_ok=True)
    os.makedirs(f"{targret_dir}/2d_joint", exist_ok=True)

    h36m_main(
        AID,
        PID,
        GID,
        camera_ids,
        gt_dir,
        third_party_dir,
        prefix,
        original_path,
        targret_dir,
        width,
        height,
    )
