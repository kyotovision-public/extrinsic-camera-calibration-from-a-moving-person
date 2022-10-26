#/bin/bash -x

DIR="$(dirname $0)"
singularity build --fakeroot "${DIR}/human_calib.sif" "${DIR}/human_calib.def"
