import tensorflow as tf
import sys


def gaussian_kernel(size: int, mean: float, std: float):
	d = tf.distributions.Normal(mean, std)
	vals = d.prob(tf.range(size, dtype = tf.float32))
	gauss_kernel = tf.einsum('i,j,k->ijk',
							vals,
							vals,
							vals)             
	# normalise gauss kernel to sum = 1
	kerngausnorm = gauss_kernel / tf.reduce_mean(gauss_kernel)
	return kerngausnorm

def squareIt(tensor):
    tensorpow2 = tf.multiply(tensor, tensor)
    return tensorpow2

sess = tf.compat.v1.Session()
with sess.as_default():
    tensor = gaussian_kernel(5, 2.0, 1.0)
    tensor2 = gaussian_kernel(5, 3.0, 1.0)
    tensor = tf.dtypes.cast(tensor, tf.float32)
    tensor2 = tf.dtypes.cast(tensor2, tf.float32)
    tensorSquared = squareIt(tensor)
    tensorPlusTen = tensor + 10.0
    # expand its dimensionality to fit into conv3d, input and filter have different dimension orders, filter has no batch but in and out channels as 2 last dims
    tensor_expand = tf.expand_dims(tensor, 0)
    tensor_expand = tf.expand_dims(tensor_expand, -1)
    tensor_filter = tf.expand_dims(tensor2, -1)
    tensor_filter = tf.expand_dims(tensor_filter, -1)
    # why does tf.nn.conv3d output a tensor containing only 1 value???eg  [[[[[]35234.325]]]] is it the sum of the real 3D image output? 
    tensorConvolved = tf.compat.v1.nn.conv3d(tensor_expand, filter=tensor_filter, strides=[1,1,1,1,1], padding="VALID", data_format='NDHWC')
    print_op = tf.print("3D Gauss Kernel? :\n", tensor, " Sum = ", tf.reduce_sum(tensor), " Max = ", tf.reduce_max(tensor), "\n",
                        "3D Gauss Kernel squared? :\n", tensorSquared, " Sum = ", tf.reduce_sum(tensorSquared), " Max = ", tf.reduce_max(tensorSquared), "\n",
                        "3D Gauss Kernel convolved? :\n", tensorConvolved, " Sum = ",
                        tf.reduce_sum(tensorConvolved), " Max = ", tf.reduce_max(tensorConvolved), " DimsShape = ", tf.shape(tensorConvolved), "\n",
                        " Plus10 :\n", tensorPlusTen,
                        output_stream=sys.stdout)
    with tf.control_dependencies([print_op]):
        the_tensor = tensor * 1.0
        the_tensor2 = tensor2 * 1.0
        the_tensorpow2 = tensorSquared * 1.0
        the_tensorConvolved = tensorConvolved * 1.0
    sess.run(the_tensorConvolved)

