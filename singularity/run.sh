#/bin/bash -x

DIR="$(dirname $0)"
singularity shell --nv "$DIR/human_calib.sif"
