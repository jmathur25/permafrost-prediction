
# Setup

InSAR processing is fairly complex and involves many dependencies. To standardize and also streamline development, I've prepared these Dockerfiles. Eventually, the goal is to have a single Docker image to do all development. However, at the moment the MintPy Dockerfile was simpler and works if you do not need to modify ISCE2.

First, update submodules:
```
git submodule update --init --recursive
```

TODO: below Dockerfiles do not install permafrost dependencies yet

TODO: push premade images?

# MintPy
This is the Docker image used to create the results in the paper. The Dockerfile is based on this [repo](https://github.com/yunjunz/conda_envs). I found it worked much better than MintPy's provided Dockerfile. Run:
```bash
cd mintpy
bash build.sh
bash start.sh
cd /permafrost-prediction
pip3 install -r requirements.txt
```

You should only need to do this once. From now on, you can always get a terminal in the Docker container by doing:
```bash
docker exec -it --env=TERM permafrost_prediction_mintpy bash -l
```

If the container stops due to something like reboot, restart it with:
```bash
docker start permafrost_prediction_mintpy
```

# ISCE2
TODO: This has not been tested recently. This will be useful if ISCE2 needs to be modified.

# Dockerfile Design
These Dockerfiles have been written to facilitate development. They bind-mount important folders into the container so that they can be edited/recompiled and therefore developed. The advantage is that you don't need to rebuild the Dockerfile every time you want to test a change to an important dependency (such as MintPy or isce2). Instead, you can edit those dependencies in container and recompile in container. Furthermore, since it is bind-mounted, your work persists even if the container dies because the files safely live on the host. To accomplish this, the Dockerfiles have to be written a specific way. Because Docker only permits bind-mounting folders with read-only permissions at build time, you cannot use the build step to compile in a bind-mounted folder. You must do some of the dependency setp in the Dockerfile, and then have an `init` script that runs when you first enter the Dockerfile. This `init` script assumes you bind-mounted certain folders to certain places. Hence, the build and run scripts for these Dockerfiles are coupled.

# VSCode
I recommended you use VSCode for development. You can then use the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension to open a window into the container.

