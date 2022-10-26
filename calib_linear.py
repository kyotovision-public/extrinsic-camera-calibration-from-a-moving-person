#%%
import cv2
import numpy as np
import sys
import json
import itertools
import os
import scipy as sp

# from numba import jit
from argument import parse_args
from util import *
import pycalib

# module_path = os.path.abspath(os.path.join('./pycalib/'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

# import pycalib


def collinearity_w2c(R_w2c, n, idx_v, idx_t, num_v, num_t):
    nmat = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])

    t0 = num_v * 3
    # A = np.zeros((3, (num_v + num_t)*3))
    A = sp.sparse.lil_matrix((3, (num_v + num_t) * 3), dtype=np.float64)
    A[:, idx_v * 3 : idx_v * 3 + 3] = nmat @ R_w2c
    A[:, t0 + idx_t * 3 : t0 + idx_t * 3 + 3] = nmat

    return A


def coplanarity_w2c(Ra, Rb, na, nb, idx_t1, idx_t2, num_t):
    rows = na.shape[0]
    assert na.shape[1] == 3
    assert nb.shape[0] == rows
    assert nb.shape[1] == 3

    m = np.cross(na @ Ra, nb @ Rb)
    # A = np.zeros((rows, num_t * 3))
    A = sp.sparse.lil_matrix((rows, num_t * 3), dtype=np.float64)
    A[:, idx_t1 * 3 : idx_t1 * 3 + 3] = m @ Ra.T
    A[:, idx_t2 * 3 : idx_t2 * 3 + 3] = -m @ Rb.T
    return A


def calib_linear(v_CxNx3, n_CxMx3):
    C = v_CxNx3.shape[0]
    N = v_CxNx3.shape[1]
    M = n_CxMx3.shape[1]
    assert v_CxNx3.shape[2] == 3
    assert n_CxMx3.shape[0] == C
    assert n_CxMx3.shape[2] == 3

    # Rotation
    v_Nx3C = np.hstack(v_CxNx3)
    Y, D, Zt = np.linalg.svd(v_Nx3C)
    # print('ratio =', np.sum(D[:3])/np.sum(D))
    V = Y[:, :3] @ np.diag(D[:3]) / np.sqrt(C)
    R_all = np.sqrt(C) * Zt[:3, :]
    # make R0 be I (also correct handedness)
    Rx = np.linalg.inv(R_all[:3, :3])
    R_all = Rx @ R_all
    assert np.linalg.det(R_all[:3, :3]) > 0

    R_w2c_list = R_all.T.reshape((-1, 3, 3))

    # Translation
    A = []
    for idx_t, (R, n) in enumerate(zip(R_w2c_list, n_CxMx3)):
        for idx_v in range(n.shape[0]):
            A.append(collinearity_w2c(R, n[idx_v, :], idx_v, idx_t, M, C))
    # A = np.vstack(A)
    A = sp.sparse.vstack(A)

    B = []
    for ((a, Ra, na), (b, Rb, nb)) in itertools.combinations(
        zip(range(C), R_w2c_list, n_CxMx3), 2
    ):
        B.append(coplanarity_w2c(Ra, Rb, na, nb, a, b, C))
    # B = np.vstack(B)
    B = sp.sparse.vstack(B)

    C = sp.sparse.lil_matrix((A.shape[0] + B.shape[0], A.shape[1]), dtype=np.float64)
    # C = np.zeros( (A.shape[0] + B.shape[0], A.shape[1]) )
    C[: A.shape[0]] = A
    C[A.shape[0] :, -B.shape[1] :] = B

    # _, _, vt = np.linalg.svd(A.T@A)
    # k = vt.T[:,-4:]
    # vt of svd(B) == v[:,::-1] of eigh(B.T@B)
    w, v = sp.linalg.eigh(
        (C.T @ C).toarray(), subset_by_index=(0, 5), overwrite_a=True, overwrite_b=True
    )
    # w, v = np.linalg.eigh((C.T@C))
    # w, v = sp.sparse.linalg.eigs(C.T@C, 5, which='SM')
    if w[3] / w[4] > 1e-4:
        print(f"WARN: degenerate case (only 4 eigenvalues should be zero): lambda={w}")
        # sys.exit(0)

    # null-space has 4-dim = any-translation for x/y/z + global-scale
    k = v[:, :4]

    # find a set of coeffs to make t0 be (0, 0, 0)
    _, s, vt = np.linalg.svd(k[-B.shape[1] : -B.shape[1] + 3, :])  # t0
    t = k @ vt[3, :].T  # vt[3] is the coeffs to make t0 zero
    X = t[: -B.shape[1]].reshape((-1, 3))
    t = t[-B.shape[1] :]
    s = np.linalg.norm(t[3:6])
    t = t / s
    X = X / s
    t_w2c_list = t.reshape((-1, 3))

    # z-test to fix the global sign ambiguity
    R1 = R_w2c_list[0]
    R2 = R_w2c_list[1]
    t1 = t_w2c_list[0]
    t2 = t_w2c_list[1]
    n1 = n_CxMx3[0]
    n2 = n_CxMx3[1]
    sign, Np, Nn = z_test_w2c(R1, t1, R2, t2, n1, n2)

    t_w2c_list = sign * t_w2c_list
    X = sign * X

    # recompute X
    P = np.concatenate((R_w2c_list, t_w2c_list[:, :, None]), axis=2)
    X2 = []
    for i in range(M):
        x = pycalib.calib.triangulate(n_CxMx3[:, i, :], P)
        X2.append(x)
    X2 = np.array(X2)[:, :3]

    return R_w2c_list, t_w2c_list.reshape((-1, 3, 1)), X2


def main_linear(
    dirname, gid, aid, pid, bone_idx, joint_idx, bObs_mask, *, frame_skip=15
):

    if bObs_mask:
        CAMID, K, _, _, _, p3d, s3d, p2d, s2d, frames = load_eldersim(
            dirname, gid, aid, pid, joint2d_dir="2d_joint_mask"
        )
    else:
        CAMID, K, _, _, _, p3d, s3d, p2d, s2d, frames = load_eldersim(
            dirname, gid, aid, pid
        )
    # skip frames
    p3d = p3d[:, ::frame_skip, :, :]
    p2d = p2d[:, ::frame_skip, :, :]
    s3d = s3d[:, ::frame_skip, :]
    s2d = s2d[:, ::frame_skip, :]

    # valid joints
    mask_CxNxJ = (s2d > 0) * (s3d == 1)

    # get mask of joints visible from all the cameras
    mask_vis_NxJ = visible_from_all_cam(mask_CxNxJ)

    # get valid orientations == oriented points visible from all the cameras
    vc = joints2orientations(p3d, mask_vis_NxJ, bone_idx)
    # assert v.shape[1] == 3
    assert vc.shape[0] == len(CAMID)
    assert vc.shape[2] == 3

    """
    vc = []
    for r in R_w2c:
        vc.append(v @ r.T)
    vc = np.array(vc)
    """

    # get valid projections
    y = joints2projections(p2d, mask_vis_NxJ, joint_idx)
    assert y.shape[2] == 2

    # convert y to homogeneous coord
    n = np.ones((y.shape[0], y.shape[1], 3), dtype=np.float64)
    n[:, :, :2] = y
    # print(n)

    # left-multiply K0^-1 to get n
    # n = np.einsum('ij,klj->kli', np.linalg.inv(K0), n)
    ni = []
    for i in range(len(K)):
        ni.append(n[i] @ np.linalg.inv(K[i]).T)
    n = np.array(ni)

    print(f"vc={vc.shape}, n={n.shape}")

    R_w2c_est, t_w2c_est, p3d_w_est = calib_linear(vc, n)

    # print(R_w2c[0].T @ v[0,10,:].T)
    # print(R_w2c[1].T @ v[1,10,:].T)
    # print( R_w2c[0].T @ (p3d[0,1,:,:].T - t_w2c[0][:,None]))
    # print( R_w2c[1].T @ (p3d[1,1,:,:].T - t_w2c[1][:,None]))

    # X01 = triangulate(R_w2c[0], t_w2c[0], R_w2c[1], t_w2c[1], n[0], n[1])
    # print(X01.shape)
    # x01 = K0 @ (R_w2c[0] @ X01.T + t_w2c[0][:,None])
    # x01 = x01 / x01[2,:]
    # print(y [0] - x01[:2,:].T)

    E = []
    for k, R, t, pts2d in zip(K, R_w2c_est, t_w2c_est, y):
        p = k @ (R @ p3d_w_est.T + t)
        p = p / p[2, :]
        e = pts2d - p[:2, :].T
        E.append(e)
    e = np.mean(np.linalg.norm(e, axis=1))

    return R_w2c_est, t_w2c_est, p3d_w_est, e, CAMID, K


if __name__ == "__main__":

    args = parse_args()
    PREFIX = args.prefix + "/" + args.target
    AID = args.aid
    PID = args.pid
    GID = args.gid
    DATASET = args.dataset

    OBS_MASK = args.obs_mask
    print(f"dataset={DATASET}")
    os.makedirs(args.prefix + "/results/", exist_ok=True)
    if OBS_MASK:
        JSON_OUT = (
            args.prefix
            + "/results/"
            + "linear_"
            + args.target.split("_")[1]
            + "_"
            + args.target.split("_")[2]
            + "_mask"
            + ".json"
        )
    else:
        JSON_OUT = (
            args.prefix
            + "/results/"
            + "linear_"
            + args.target.split("_")[1]
            + "_"
            + args.target.split("_")[2]
            + ".json"
        )
    FRAME_SKIP = args.frame_skip
    R, t, X, e, CAMID, K = main_linear(
        PREFIX, GID, AID, PID, OP_BONE, OP_KEY_SUB, OBS_MASK, frame_skip=FRAME_SKIP
    )

    with open(JSON_OUT, "w") as fp:
        out = {
            #            'E_reproj': e,
            "CAMID": CAMID.tolist(),
            "K": K.tolist(),
            "R_w2c": R.tolist(),
            "t_w2c": t.tolist(),
        }
        json.dump(out, fp, indent=2, ensure_ascii=True)

    print(f"Reprojection error: {e:e}")
    print(" ")
