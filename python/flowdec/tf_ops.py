
import tensorflow as tf
import numpy as np


def tf_print(t, transform=None):
    """Inject graph operation to print a tensors underlying value (or transformation of it)"""
    def log_value(x):
        print('{} - {}'.format(t.name, x if transform is None else transform(x)))
        return x
    log_op = tf.py_func(log_value, [t], [t.dtype], name=t.name.split(':')[0])[0]
    with tf.control_dependencies([log_op]):
        r = tf.identity(t)
    return r


def tf_observer(tensors, observer_fn):
    """Inject graph operation to observe but not alter tensor values
    
    Args:
        t: List of tensors to send to observer
        observer_fn: Function with signature ```fn(tensors)``` where tensors will be a list
            of numpy arrays or scalars representing the underlying tensor values; return
            value from function will be ignored
    Returns:
        Input tensors wrapped with dependencies forcing passage of data to observer
    """
    def _observe(*args):
        observer_fn(*args)
        return np.array([0], dtype=np.int32)
    observe_op = tf.py_func(_observe, tensors, tf.int32, stateful=True, name='observer')[0]
    with tf.control_dependencies([observe_op]):
        ts = [tf.identity(t) for t in tensors]
    return ts


def pad_around_center(t, target_shape, mode='CONSTANT', constant_values=0):
    """Center tensor data within a (possibly) larger tensor

    Args:
        t: Tensor to center
        target_shape: Target shape of resulting tensor; must be >= shape of `t` in all dimensions
        mode: One of ['CONSTANT', 'SYMMETRIC', 'REFLECT']; see
            https://www.tensorflow.org/api_docs/python/tf/pad for details
        constant_values: Constant value to use for padding with 'CONSTANT' mode (default 0)
    Returns:
        Tensor with shape matching given shape and where necessary new values padded in
    """
    t_shape = tf.shape(t)
    lopad = (target_shape - t_shape + 1) // 2
    hipad = target_shape - t_shape - lopad
    return tf.pad(t, tf.stack([lopad, hipad], axis=1), mode=mode, constant_values=constant_values)


def unpad_around_center(t, source_shape):
    """Remove padding around centered tensor

    Args:
        t: Tensor to extract data from
        source_shape: Shape of original tensor before padding around tensor (presumably via `tf_ops.pad_around_center`);
            must be <= shape of `t` in all dimensions
    Returns:
        Tensor with shape matching source_shape with padding around edges removed
    """
    begin = (tf.shape(t) - source_shape + 1) // 2
    return tf.slice(t, begin, source_shape)