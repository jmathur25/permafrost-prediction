set -eux

IMAGE_NAME=docker.io/hysds/isce2_live_dev:latest
CONTAINER_NAME=permafrost_prediction_isce2
CURRENT_DIR="$(dirname "$0")"
REPO_ROOT=$CURRENT_DIR/../../

docker create \
    --name $CONTAINER_NAME \
    -v $REPO_ROOT:/permafrost-prediction \
    -v $CURRENT_DIR/isce2:/opt/isce2/src/isce2 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --device /dev/dri:/dev/dri \
    -e DISPLAY=$DISPLAY \
    -v $XAUTHORITY:/tmp/.XAuthority \
    -e XAUTHORITY=/tmp/.XAuthority \
    --network=host \
    --tty \
    $IMAGE_NAME

docker start $CONTAINER_NAME

docker exec -it --env=TERM $CONTAINER_NAME bash -l

