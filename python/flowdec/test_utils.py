"""Utility methods for use in unit testing"""
import tensorflow as tf


def exec_tf(fn):
    g = tf.Graph()
    with g.as_default():
        tf_res = fn()
    with tf.Session(graph=g) as sess:
        return sess.run(tf_res)