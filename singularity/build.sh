#/bin/bash -x

DIR="$(dirname $0)"
singularity build --fakeroot "${DIR}/env.sif" "${DIR}/env.def"
