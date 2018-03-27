# Tensorflow Deconvolution for Microscopy


## Installation

**WIP** - This should eventually look like this:

- pip install a package for building TF graphs and running them from python 
- maven artifact for using pre-built graphs from java
- docker examples 
    - \* need to figure out how to extend the Tensorflow dynamic Dockerfile build script to include a few extra steps


### Docker Instructions

**WIP** - Eventually much of this will end up in a dockerfile but for now here is a start on getting things stood up:

```
docker pull tensorflow/tensorflow:latest-gpu
docker run -td --name tf tensorflow/tensorflow:latest-gpu
docker exec -it tf bash

# to stop
docker stop id
docker rm id

# docker steps
apt-get update
apt-get install -y git
apt-get install -y vim
mkdir /repos; cd /repos
git clone https://github.com/eric-czech/flowdec.git
cd codex-analysis/deconvolution/tfdecon/python/
pip install .


apt-get install -y default-jdk
apt-get install -y maven
```

## Examples

### Python 

- [C. Elegans](python/examples/CElegans%20Deconvolution.ipynb) - Deconvolution of 712x672x104 acquisition
- [Hollow Bars](python/examples/Hollow%20Bars%20Deconvolution.ipynb) - Deconvolution of 256x256x128 (rows x cols x z) synthetic data
- [Graph Export](python/examples/Algorithm%20Graph%20Export.ipynb) - Defining and exporting Tensorflow graphs

### Java

- [Multi-GPU Example](java/tf-decon/src/main/java/org/hammerlab/tfdecon/examples/MultiGPUExample.java) - Prototype example for how to (hopefully) be able to execute deconvolution against multiple GPUs in parallel

## TODO

- Decide what to do with datasets too big to fit in repo
- Determine whether or not PSF generation utilities are worth the effort
- Make docker files for environment initilization
    - Extend tensorflow Dockerfile script?
    - Finish setup.py
    - Deploy to maven central?
- Test multi-gpu on some linux machine via java
- Compare results between DL2, TF, and microvolution (for time and accuracy)
    - Attempt to get microvolution working in standalone examples (for more direct comparisons of time and accuracy)
- Tensorboard monitoring in frozen graphs?
    - Does any of this work in java?
- Unit tests:
    - fftshift
    - cropping
    - padding
- Address TODOs in code
