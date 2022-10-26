#/bin/bash -x

DIR="$(dirname $0)"
singularity shell --nv "$DIR/env.sif"
