
cd "$(dirname "$0")"

docker build -f Dockerfile -t ubuntu_22.04_mintpy:latest .
