"""Export Project TensorFlow Graphs for use in other TF client APIs"""
from flowdec import restoration as fd_restoration
import flowdec
import shutil
import os
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    for args in [
        ['richardsonlucy', 1, 'complex'],
        ['richardsonlucy', 2, 'complex'],
        ['richardsonlucy', 3, 'complex']
    ]:
        algo_name, ndims, domain = args
        logger.info('Building graph export for arguments {}'.format(args))

        graph_dir = '{}-{}-{}d'.format(algo_name, domain, ndims)
        export_dir = os.path.abspath(os.path.join(flowdec.tf_graph_dir, graph_dir))

        if os.path.exists(export_dir):
            shutil.rmtree(export_dir)

        algo = fd_restoration.RichardsonLucyDeconvolver(
            ndims, pad_mode='log2', real_domain_fft=(domain == 'real')
        ).initialize()

        algo.graph.save(export_dir, save_as_text=False)

# rsync -rP ~/repos/hammer/flowdec/tensorflow/* ~/repos/imagej/ops-experiments/ops-experiments-tensorflow/src/main/resources/tensorflow/graphs/