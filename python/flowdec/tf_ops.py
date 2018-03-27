
import tensorflow as tf


def tf_print(t, transform=None):
    """Inject graph operation to print a tensors underlying value (or transformation of it)"""
    def log_value(x):
        print('{} - {}'.format(t.name, x if transform is None else transform(x)))
        return x
    log_op = tf.py_func(log_value, [t], [t.dtype], name=t.name.split(':')[0])[0]
    with tf.control_dependencies([log_op]):
        r = tf.identity(t)
    return r


def pad_around_center(t, target_shape, fill=0):
    """Center tensor data within a (possibly) larger tensor

    Args:
        t: Tensor to center
        target_shape: Target shape of resulting tensor; must be >= shape of `t` in all dimensions
        fill: Constant value to use for padding (default 0)
    Returns:
        Tensor with shape matching given shape and where necessary new values padded in (as `fill` value)
    """
    t_shape = tf.shape(t)
    lopad = (target_shape - t_shape + 1) // 2
    hipad = target_shape - t_shape - lopad
    return tf.pad(t, tf.stack([lopad, hipad], axis=1), constant_values=fill)


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