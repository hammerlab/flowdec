

import os
from os import path as osp
import tensorflow as tf
import timeit
from skimage import io
from tfdecon import restoration as tfdecon_restoration
from tfdecon import data as tfdecon_data

# in_path = 'C:/Users/User/data/deconvolution/output/test-stack-1/'
# data = io.imread(osp.join(in_path, 'data.tif'))
# kern = io.imread(osp.join(in_path, 'psf.tif'))

in_path = 'C:/Users/User/data/deconvolution/data/bars/'
data = io.concatenate_images(io.imread_collection(osp.join(in_path, 'Bars-G10-P30/*.tif')))
kern = io.concatenate_images(io.imread_collection(osp.join(in_path, 'PSF-Bars/*.tif')))

print('data shape = {}, kernel shape = {}'.format(data.shape, kern.shape))

algo = tfdecon_restoration.RichardsonLucyDeconvolver(
    data.ndim, real_domain_fft=False, pad_mode='log2'
    ).initialize()

times = []
def run_deconvolution():
    return algo.run(
        tfdecon_data.Acquisition(data=data, kernel=kern), 25, 
        session_config=tf.ConfigProto(
            device_count={'GPU': 1},
            device_filters=['/device:GPU:1']
        )
    )

for i in range(5):
    times.append(timeit.timeit(run_deconvolution, number=1))

res = run_deconvolution()

print('Result shape = {}'.format(res.data.shape))
print('Times: {}'.format('\n'.join([str(t) for t in times])))

io.imsave(os.path.join(in_path, 'result.tif'), res.data)

print('Done')