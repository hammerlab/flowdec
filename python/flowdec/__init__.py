import os.path as osp

pkg_dir = osp.abspath(osp.dirname(__file__))
data_dir = osp.normpath(osp.join(pkg_dir, 'datasets'))
tf_graph_dir = osp.normpath(osp.join(pkg_dir, '../../tensorflow'))
