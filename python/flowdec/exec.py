""" Utilities for executing external commands """
import tempfile
import numpy as np
from skimage import io
import os

# TODO: Decide whether or not to actually incorporate DL2 jar with project
DL2_JAR_PATH = '/Users/eczech/.m2/repository/org/hammerlab/dl2-decon/0.0.1-SNAPSHOT/dl2-decon-0.0.1-SNAPSHOT.jar'


def run_dl2(acq, n_iter, pad_mode):
    tmp_data_path = tempfile.mkstemp(prefix='dl2-data-', suffix='.tif')[1]
    tmp_kern_path = tempfile.mkstemp(prefix='dl2-kern-', suffix='.tif')[1]
    output_path = tempfile.mkstemp(prefix='dl2-output-', suffix='.tif')[1]

    io.imsave(tmp_data_path, acq.data.astype(np.float32))
    io.imsave(tmp_kern_path, acq.kernel.astype(np.float32))

    cmd = 'java -jar {jar} -i {data_path} -p {kern_path} -o {out_path} -n {n_iter} -d {pad_mode}'.format(
        jar=DL2_JAR_PATH, data_path=tmp_data_path, kern_path=tmp_kern_path,
        out_path=output_path, n_iter=n_iter, pad_mode=pad_mode
    )
    code = os.system(cmd)
    if code != 0:
        raise RuntimeError('Failed to run DL2 command (exit code = {}) "{}"'.format(cmd, code))
    return io.imread(output_path)