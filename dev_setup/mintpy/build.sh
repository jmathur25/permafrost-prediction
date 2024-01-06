
cd "$(dirname "$0")"

IMAGE_NAME=ubuntu_22.04_mintpy:latest
docker build -f Dockerfile -t $IMAGE_NAME .
