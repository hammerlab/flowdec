# Flowdec

**Flowdec** is a library containing [Tensorflow](https://github.c.om/tensorflow/tensorflow) (TF) implementations of image and signal deconvolution algorithms.  Currently, only [Richardson-Lucy Deconvolution](https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution) has been implemented but others may come in the future.

Flowdec is designed to construct and execute TF graphs in python as well as use frozen, exported graphs from other languages (e.g. Java).

Here are a few of the major advantages and disadvantages of Flowdec at the moment:

*Advantages*

- **Support for Windows, Mac, and Linux** - Because Tensorflow can run on these platforms, so can Flowdec
- **Client Support for Java, Go, C++, and Python** - Using Flowdec graphs from Python and Java has been tested, but theoretically they could also be used by any [Tensorflow API Client Libraries](https://www.tensorflow.org/api_docs/)
- **GPU Accleration** - Executing [Tensorflow graphs on GPUs](https://www.tensorflow.org/programmers_guide/using_gpu) is trivial and will happen by default w/ Flowdec if you meet all of the Tensorflow requirements for this (i.e. CUDA Toolkit installed, Nvidia drivers, etc.)
- **Image Dimensions** - Flowdec can support 1, 2, or 3 dimensional images/signals
- **Multi-GPU Usage** - This has yet to be tested, but theoretically this is possible since TF can do it (and this [Multi-GPU Example](java/tf-decon/src/main/java/org/hammerlab/tfdecon/examples/MultiGPUExample.java) is a start)
- **Image Preprocessing** - A trickier part of deconvolution implementations is dealing with image padding and cropping necessary to use faster FFT implementations -- in Flowdec, image padding using the reflection of the image along each axis can be specified manually or by letting it automatically round up and pad to the nearest power of 2 (which will enable use of faster Cooley-Tukey algorithm instead of the Bluestein algorithm provided by Nvidia cuFFT used by TF).
- **Performance** - There are some other similar open source solutions that run *partially* with GPU acceleration, though this generally means that only FFT and iFFT operations run on GPUs while all other operations run on the CPU -- Experiments with Flowdec show that this means that the same 3D image may take ~8 mins to run on a single CPU, ~40s to run with another open source solution using only FFT/iFFT GPU accleration, and ~1s with Flowdec/Tensorflow using GPU accleration for all operations.
- **Visualizing Iterations** - Another difficulty with iterative deconvolution algorithms is in determining when they should stop.  With Richardson Lucy, this is usually done somewhat subjectively based on visualizing results for different iteration counts and Flowdec at least helps with this by letting ```observer``` functions be given that take intermediate results of the deconvolution process to be written out to image sequences or stacks for manual inspection.  Future work may include using [Tensorboard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) to do this instead but for now, it has been difficult to get image summaries working within TF "while" loops.

*Disadvantages*

- **Point Spread Functions** - Flowdec does not yet generate point spread functions so these must be built and supplied by something such as [PSFGenerator](http://bigwww.epfl.ch/algorithms/psfgenerator/).
- **No Blind Deconvolution** - Currently, nothing in this arena has been attempted but since much recent research on this subject is centered around solutions in deep learning, Tensorflow will hopefully make for a good platform in the future.


## Basic Usage

Here is a basic example demonstrating how Flowdec can be used in a single signal deconvolution:

See full example notebook [here](python/examples/Astronaut%20Deconvolution.ipynb)

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import data as sk_data
from skimage import color as sk_color
from scipy.signal import fftconvolve
from flowdec import data as fd_data
from flowdec import restoration as fd_restoration

# Load skimage "Astro" 2D image and generate a fake PSF
img = sk_color.rgb2gray(sk_data.astronaut())
psf = np.ones((5, 5)) / 25

# Add blur and noise to Image
np.random.seed(1)
img_blur = fftconvolve(img, psf, 'same') + (np.random.poisson(lam=25, size=img.shape) - 10) / 255.

# Wrap data in an "Acqusition" instance
acquisition = fd_data.Acquisition(data=img_blur, kernel=psf)

# Initialize Tensorflow graph for 2-dimensional deconvolution 
algo = fd_restoration.RichardsonLucyDeconvolver(n_dims=img.ndim).initialize()

# Run 30 RL deconvolution iterations
img_decon = algo.run(acquisition, niter=30).data

# Alternatively:
# res = tf_restoration.richardson_lucy(acquisition, niter=30)) will
# accomplish the same thing in a single step but this generally a bad
# idea to use in a loop because the TF graph is redefined each time

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 5))
plt.gray()
ax[0].imshow(img)
ax[0].set_title('Original Data')
ax[1].imshow(img_blur)
ax[1].set_title('Noisy data')
ax[2].imshow(img_decon, vmin=img_blur.min(), vmax=img_blur.max())
ax[2].set_title('Restoration using\nRichardson-Lucy')
```

![Astro Example](docs/images/astro.png "Astro")


## More Examples

### Python 

- [C. Elegans](python/examples/CElegans%20Deconvolution.ipynb) - Deconvolution of 712x672x104 acquisition
- [Hollow Bars](python/examples/Hollow%20Bars%20Deconvolution.ipynb) - Deconvolution of 256x256x128 (rows x cols x z) synthetic data
- [Graph Export](python/examples/Algorithm%20Graph%20Export.ipynb) - Defining and exporting Tensorflow graphs

### Java

- [Multi-GPU Example](java/tf-decon/src/main/java/org/hammerlab/tfdecon/examples/MultiGPUExample.java) - Prototype example for how to (hopefully) be able to execute deconvolution against multiple GPUs in parallel

## Installation

The project can be installed, ideally in a python 3.6 environment (though it should work in 3.5 too), by running:

```bash
git clone https://github.com/hammerlab/flowdec.git
cd flowdec
pip install .
```


### Docker Instructions

A local docker image can be built by running:

```bash
cd flowdec  # Note: not flowdec/docker, just cd flowdec

docker build --no-cache -t flowdec -f docker/Dockerfile .

# If on a system that supports nvidia-docker, the GPU-enabled version can be built instead via:
# nvidia-docker build --no-cache -t flowdec -f docker/Dockerfile.gpu .
```

The image can then be run using:

```bash
# Run in background (port mapping is host:container if 8888 is already taken)
docker run -td -p 8888:8888 --name flowdec flowdec
docker exec -ti --name flowdec bash

# Run in foreground
docker run -ti -p 8888:8888 flowdec
```

The Flowdec dockerfile extends the [Tensorflow DockerHub Images](https://hub.docker.com/r/tensorflow/tensorflow/) so its usage is similar, where running it in the foreground automatically starts jupyter notebook and prints a link to connect to it via a browser on the host system.

#### Docker Development Notes

List of Tensorflow tags on docker hub to inherit dockerfile from: https://hub.docker.com/r/tensorflow/tensorflow/tags/

Building a CPU only image:

```bash
cd flowdec
docker build --no-cache -t flowdec -f docker/Dockerfile .
```

Building a GPU image:

```bash
cd docker
nvidia-docker build --no-cache -t flowdec -f Dockerfile.gpu .
```


## TODO

- Add PSF generator
- Add and test java within docker image: ```apt-get install -y default-jdk maven```
- Test multi-gpu on some linux machine via java
- Tensorboard monitoring during iterations instead of python function injection
