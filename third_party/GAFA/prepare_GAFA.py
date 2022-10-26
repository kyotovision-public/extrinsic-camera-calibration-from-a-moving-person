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

import argument

SCENARIOS = "lab"
SUBSCENARIOS = 2
PATH_INTRINSIC = "./third_party/GAFA/GoPro_2700_linear_intrinsic.npz"
PATH_MANUAL_PTS = "./third_party/GAFA/manual_pts.csv"
PATH_VERIFICATION = "./third_party/GAFA/verification/"
TH_DIST_PIXEL = 10
TH_MIN_VIEWS = 1
NFRAME = 400
idx_col_csv = {
    "1_2": (0, 20),
    "2_2": (21, 39),
    "5_2": (40, 66),
    "6_2": (67, 91),
    "7_2": (92, 113),
}
rgbs = np.random.randint(255, size=(500, 3))


class Camera:
    def __init__(self, cam_id, path_caminfo, path_intrinsic, scenario):
        self.cam_id = cam_id
        self.path_cam_info = path_caminfo
        self.scenario = scenario

        (
            self.cam_mat,
            self.dist_coeffs,
            self.R_c2w,
            self.t_c2w,
            self.joints2ds,
        ) = self.load_caminfo(cam_id, path_caminfo, path_intrinsic)
        # print(self.R_c2w)
        self.R_w2c, self.t_w2c = invRT(self.R_c2w, self.t_c2w)
        self.manual_kpts = pd.read_csv(PATH_MANUAL_PTS)
        self.img0 = cv2.imread(f"{path_caminfo}/Camera_{self.cam_id}/{0:0=6}.jpg")

        self.aruco_corners = self.get_aruco_corners(
            self.img0,
            self.cam_id,
            self.cam_mat,
            self.dist_coeffs,
            scenario,
            PATH_VERIFICATION,
        )

    def load_caminfo(self, cam_id, path_cam_info, path_intrinsic):
        # cam_params_file = f'{self.path}/Camera_{self.cam_id}.pkl'
        cam_params_file = f"{path_cam_info}/Camera_{self.cam_id}.pkl"
        # check_file(cam_params_file)
        # check_file(path_intrinsic)
        intrinsics = np.load(path_intrinsic)
        cam_mat = intrinsics["cam_mat"]
        # dist_coeffs = intrinsics['dist_coeffs']
        dist_coeffs = np.zeros(5)

        with open(cam_params_file, "rb") as cam_info:
            fdata = pkl.load(cam_info)
            R_cam2w = fdata["R_cam2w"]
            t_cam2w = fdata["t_cam2w"]
            joints2ds = fdata["keypoints2d"]

        return cam_mat, dist_coeffs, R_cam2w, t_cam2w, joints2ds

    def get_aruco_corners(self, im, cam_id, cam_mat, dist_coeffs, scenario, out_dir):
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        aruco_dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        aruco_param = aruco.DetectorParameters_create()
        aruco_corners, aruco_ids, rejected = aruco.detectMarkers(
            im_gray,
            aruco_dictionary,
            cameraMatrix=cam_mat,
            distCoeff=dist_coeffs,
            parameters=aruco_param,
        )
        aruco_corners = np.array(aruco_corners)[:, 0, :, :]
        aruco_ids = np.array(aruco_ids)[:, 0]

        if (cam_id == "5_2") and (scenario == "lab"):
            wrong_detected_aruco_idx = np.where(aruco_ids == 9)[0][0]
            aruco_ids = np.delete(aruco_ids, wrong_detected_aruco_idx, axis=0)
            aruco_corners = np.delete(aruco_corners, wrong_detected_aruco_idx, axis=0)

        Max_Index = 100
        ARUCO_CORNERS = 4
        K = Max_Index * ARUCO_CORNERS
        p2d = np.full((K, 2), np.nan)

        for cs, i in zip(aruco_corners, aruco_ids):
            for j, c in enumerate(cs):
                p2d[i * ARUCO_CORNERS + j] = c

        idx_from_to = idx_col_csv[cam_id]
        df_kpts = self.manual_kpts[idx_from_to[0] : idx_from_to[1]]
        df_kpts = df_kpts.set_index("Cam1")
        manual_p2d = self.get_maunal_kpts(df_kpts)
        p2d = np.concatenate([p2d, manual_p2d])

        for x2d_i, rgb_i in zip(p2d, rgbs):

            if not np.isnan(x2d_i).all():
                cv2.circle(
                    im,
                    (int(x2d_i[0]), int(x2d_i[1])),
                    radius=5,
                    color=(int(rgb_i[0]), int(rgb_i[1]), int(rgb_i[2])),
                    thickness=3,
                )

        cv2.imwrite(f"{out_dir}/aruco_Camera{cam_id}.jpg", im)
        return p2d

    def get_maunal_kpts(self, df_kpts):
        x2d = np.full((40, 2), np.nan)
        key_id = df_kpts.index.to_numpy().astype(int)
        kpts = df_kpts.iloc[:, 0:2].to_numpy().astype(int)
        x2d[key_id] = kpts
        return x2d


def calc_RT(original_gafa, camera_ids):

    cams = []
    aruco_corners_all = []
    cam_mats = []
    dist_all = []
    for cam_i in camera_ids:
        cam = Camera(
            f"{cam_i}_{SUBSCENARIOS}", original_gafa, PATH_INTRINSIC, SCENARIOS
        )
        cams.append(cam)
        cam_mats.append(cam.cam_mat)
        dist_all.append(cam.dist_coeffs)
        aruco_corners_all.append(cam.aruco_corners)
    dist_all = np.array(dist_all)
    cam_mats = np.array(cam_mats)
    Ps = [
        cam_i.cam_mat @ np.hstack((cam_i.R_w2c, cam_i.t_w2c[:, None])) for cam_i in cams
    ]
    cam_mat = cams[0].cam_mat
    dist_coeffs = cams[0].dist_coeffs
    Ps = np.array(Ps)

    CxKx2 = np.array(aruco_corners_all)

    KxCx2 = CxKx2.transpose(1, 0, 2)

    X3ds = []
    x2ds = []
    X3ds_ids = []
    cam_ids = []
    corres = dict()  # (cam_id, x2d_idx) = x3d_ids
    for i, Cx2 in enumerate(KxCx2):
        mask = ~np.isnan(Cx2).all(axis=1)
        if mask.sum() >= 2:

            pt2d = Cx2[mask]
            P = Ps[mask]
            # unpt2d = cv2.undistortPoints(pt2d, cam_mat, dist_coeffs, P=cam_mat)
            # unpt2d = unpt2d[:, 0, :]

            cam_id = np.where(mask == True)[0]
            X3d = pycalib.calib.triangulate(pt2d, P)
            for pt2d_i, cam_id_i in zip(pt2d, cam_id):
                x2ds.append(pt2d_i)
                cam_ids.append(cam_id_i)
                X3ds_ids.append(len(X3ds))
                corres[cam_id_i, i] = len(X3ds)
            X3ds.append(X3d[:3])

    X3ds = np.array(X3ds)
    x2ds = np.array(x2ds)
    cam_ids = np.array(cam_ids)
    X3ds_ids = np.array(X3ds_ids)

    camera_params = prepare_ba(cams)

    mask = pycalib.ba.make_mask(True, True)

    cam_opt, X_opt, tt, ret = pycalib.ba.bundle_adjustment(
        camera_params, X3ds, cam_ids, X3ds_ids, x2ds, mask=mask
    )

    Rs_w2c_opt, ts_w2c_opt, cam_mats_opt, dists_opt = decode_ba(cam_opt)

    validate(cams, Rs_w2c_opt, ts_w2c_opt, cam_mats_opt, X3ds, X_opt, corres, dists_opt)

    return Rs_w2c_opt, ts_w2c_opt, cam_mats_opt, dist_all


def project(R_w2c, t_w2c, X, cam_mat, cam_dist):

    rvec = cv2.Rodrigues(R_w2c)[0]
    projected_points, _ = cv2.projectPoints(X[None, :], rvec, t_w2c, cam_mat, cam_dist)

    projected_points = projected_points.reshape(-1, 2)
    return projected_points


def validate(cams, Rs_w2c, t_w2c, cam_mats_opt, X3ds, X3ds_opt, corres, dist_coeffs):

    Max_Index = 200
    ARUCO_CORNERS = 4
    K = Max_Index * ARUCO_CORNERS

    for i, (cam_i, R_w2c_opt_i, t_w2c_opt_i, cam_mat_opt_i, dist_coeff_i) in enumerate(
        zip(cams, Rs_w2c, t_w2c, cam_mats_opt, dist_coeffs)
    ):
        p2d = np.full((K, 2), np.nan)  # p2d = np.full((K, 2), np.nan)  ##
        im = cam_i.img0
        # (cam_id, x2d_idx) = x3d_ids
        corres_i = dict(filter(lambda x: x[0][0] == i, corres.items()))
        X3ds_ids_ba = np.array(list(corres_i.values()))

        x2ds_ba = project(
            R_w2c_opt_i, t_w2c_opt_i, X3ds_opt[X3ds_ids_ba], cam_mat_opt_i, dist_coeff_i
        )
        x2ds = project(
            cam_i.R_w2c,
            cam_i.t_w2c,
            X3ds[X3ds_ids_ba],
            cam_i.cam_mat,
            cam_i.dist_coeffs,
        )

        X2ds_ids_ba = np.array(list(map(lambda x: x[1], corres_i)))
        p2d[X2ds_ids_ba] = x2ds_ba

        truth = cam_i.aruco_corners[X2ds_ids_ba]
        for x2d_proj, aruco_corner in zip(x2ds_ba, x2ds):
            cv2.circle(
                im,
                (int(x2d_proj[0]), int(x2d_proj[1])),
                radius=5,
                color=[0, 255, 0],
                thickness=3,
            )
            cv2.circle(
                im,
                (int(aruco_corner[0]), int(aruco_corner[1])),
                radius=5,
                color=[0, 0, 255],
                thickness=3,
            )

        cv2.imwrite("{}/result_{}.jpg".format(PATH_VERIFICATION, cam_i.cam_id), im)
        print(
            f" cam {i} : repro. error = {np.linalg.norm(x2ds - truth, axis=1).mean()}"
        )
        print(
            f" cam {i} : repro. error after ba = {np.linalg.norm(x2ds_ba - truth, axis=1).mean()}"
        )


def prepare_ba(cams):
    # Camera parameters
    camera_params = []
    for cam_i in cams:
        c = pycalib.ba.encode_camera_param(
            cam_i.R_w2c, cam_i.t_w2c, cam_i.cam_mat, np.zeros(5)
        )

        camera_params.append(c)
    camera_params = np.array(camera_params)
    return camera_params


def decode_ba(cam_opt):
    R_w2c_opt = []
    t_w2c_opt = []
    cam_mat_opt = []
    dists_opt = []
    for cam_opt_i in cam_opt:
        c = pycalib.decode_camera_param(cam_opt_i)  # changed
        R_w2c_opt.append(c[0])
        t_w2c_opt.append(c[1])
        cam_mat_opt.append(c[2])
        dists_opt.append(c[3])

    R_w2c_opt = np.array(R_w2c_opt)
    t_w2c_opt = np.array(t_w2c_opt)
    cam_mat_opt = np.array(cam_mat_opt)
    dists_opt = np.array(dists_opt)

    return R_w2c_opt, t_w2c_opt, cam_mat_opt, dists_opt


def triangulate_without_conf(p2d, s2d, K, R_w2c, t_w2c):

    assert p2d.ndim == 4
    assert s2d.ndim == 3
    Nc, Nf, Nj, _ = p2d.shape

    mask = s2d > 0

    X = triangulate_all(
        K, R_w2c, t_w2c, p2d.reshape(Nc, Nf * Nj, 2), mask.reshape(Nc, Nf * Nj)
    )

    X = X.reshape(Nf, Nj, 3)
    return X


def GAFA_main(
    aid,
    pid,
    gid,
    camera_ids,
    gt_dir,
    prefix,
    third_party_dir,
    original_gafa,
    targret_dir,
    width,
    height,
):

    Rs_w2c, ts_w2c, Ks, dist_all = calc_RT(original_gafa, camera_ids)
    file_poses = sorted(glob.glob(f"{prefix}/poses_from_vp3d/data_*.npz"))
    file_scores = sorted(glob.glob(f"{prefix}/poses_from_vp3d/*mp4.npz"))
    x2d_coco, x2d_coco_scores = get_poses_from_vp3d(
        third_party_dir, file_poses, file_scores, NFRAME
    )

    Nc, Nf, Nj, _ = x2d_coco.shape
    X2d_op = []
    x2d_op_scroes = []
    for i in range(len(camera_ids)):
        X2d_op.append(convert_coco_to_op(x2d_coco[i]))
        x2d_op_scroes.append(convert_coco_to_op_score(x2d_coco_scores[i]))

    X2d_op = np.array(X2d_op)
    x2d_op_scroes = np.array(x2d_op_scroes)

    X3d_w_op = triangulate_without_conf(X2d_op, x2d_op_scroes, Ks, Rs_w2c, ts_w2c)

    x2d_repro_op = project_cv2(Rs_w2c, ts_w2c, Ks, X3d_w_op, width, height)

    # mask
    x2d_repro_op = x2d_repro_op.astype(np.float32)
    vis_mask_2d = (
        np.linalg.norm(X2d_op - x2d_repro_op, axis=3) <= TH_DIST_PIXEL
    )  # distance in  pixel
    # more than 2 views
    vis_mask_3d = np.sum(vis_mask_2d, axis=0) > TH_MIN_VIEWS

    vis_mask_2d_3d = []
    for vis_mask_2d_i in vis_mask_2d:
        vis_mask_2d_3d.append(np.bitwise_and(vis_mask_2d_i, vis_mask_3d))
    vis_mask_2d_3d = np.array(vis_mask_2d_3d)

    X3d_w_op_new = triangulate_without_conf(X2d_op, vis_mask_2d_3d, Ks, Rs_w2c, ts_w2c)

    # X3d_w_op_new = triangulate_with_conf(
    # X2d_op, x2d_op_scroes, Ks, Rs_w2c, ts_w2c,  vis_mask_2d_3d)

    X3d_c_op_new = get_ccs(Rs_w2c, ts_w2c, X3d_w_op_new, vis_mask_2d_3d)

    x2d_repro_op_new = project_cv2(Rs_w2c, ts_w2c, Ks, X3d_w_op_new, width, height)
    x2d_repro_op_new[~vis_mask_2d_3d] = np.nan
    assert np.isnan(x2d_repro_op_new).sum() / 2 == np.isnan(X3d_c_op_new).sum() / 3
    # vis_mask_3d = np.sum(vis_mask_2d_3d, axis=0) > TH_MIN_VIEWS
    X3d_w_op_new[~vis_mask_3d] = np.nan

    save_joint(
        f"{gt_dir}/2d_joint",
        x2d_repro_op_new,
        vis_mask_2d_3d,
        aid,
        pid,
        gid,
        camera_ids,
    )
    save_joint(
        f"{gt_dir}/3d_joint", X3d_c_op_new, vis_mask_2d_3d, aid, pid, gid, camera_ids
    )

    save_joint(
        f"{targret_dir}/2d_joint", X2d_op, x2d_op_scroes, aid, pid, gid, camera_ids
    )
    save_joint(
        f"{targret_dir}/2d_joint_coco",
        x2d_coco,
        x2d_coco_scores,
        aid,
        pid,
        gid,
        camera_ids,
    )
    save_cam(Rs_w2c, ts_w2c, Ks, dist_all, gt_dir, gid, camera_ids)
    save_cam(Rs_w2c, ts_w2c, Ks, dist_all, targret_dir, gid, camera_ids)
    save_skelton_w_op(X3d_w_op_new, gt_dir, gid)
    save_skelton_w_op(X3d_w_op_new, targret_dir, gid)


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
    original_gafa = args.src_original
    # poses_from_vp3d  = f"{third_party_dir}/poses_from_vp3d"
    AID = args.aid
    PID = args.pid
    GID = args.gid
    targret_dir = f"{prefix}/{args.target}"

    os.makedirs(f"{gt_dir}/2d_joint", exist_ok=True)
    os.makedirs(f"{gt_dir}/3d_joint", exist_ok=True)
    os.makedirs(f"{targret_dir}/2d_joint", exist_ok=True)
    os.makedirs(f"{targret_dir}/2d_joint_coco", exist_ok=True)

    GAFA_main(
        AID,
        PID,
        GID,
        camera_ids,
        gt_dir,
        prefix,
        third_party_dir,
        original_gafa,
        targret_dir,
        width,
        height,
    )
