#!/bin/bash
if [ $# != 11 ]; then
    echo
    echo $0
    echo [Usage] PREFIX AID, PID, GID, EID, FRAME_SKIP LAMBDA1 LAMBDA2 TARGET DATASET OBS_MASK SAVE_OBS_MASK
    echo [e.g.] sh ./ba.sh ./data/A023_P102_G003 23 102 3 20 1. 1. linear_3_0 SynADL false false
  
    exit
fi


PREFIX=${1}
AID=${2}
PID=${3}
GID=${4}
FRAME_SKIP=${5}
LAMBDA1=${6}
LAMBDA2=${7}
TARGET=${8}     
DATASET=${9}
OBS_MASK=${10}
SAVE_OBS_MASK=${11}

python3 ba.py --prefix ${PREFIX} --aid ${AID} --pid ${PID} --gid ${GID} --frame_skip ${FRAME_SKIP} --ba_lambda1 ${LAMBDA1} --ba_lambda2 ${LAMBDA2} --target ${TARGET} --dataset ${DATASET} --obs_mask $OBS_MASK --th_obs_mask 20 --save_obs_mask ${SAVE_OBS_MASK}
