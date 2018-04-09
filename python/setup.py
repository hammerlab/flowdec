from setuptools import setup
import os

if os.getenv('TF_GPU', 'false') == 'true':
    requires = ['tensorflow-gpu>=1.6.0']
else:
    requires = ['tensorflow>=1.6.0']

with open('requirements.txt', 'r') as fd:
    requires += [l.strip() for l in fd.readlines()]

if __name__ == '__main__':
    setup(
        name='flowdec',
        version='0.0.1',
        description="TensorFlow Implementations of Signal Deconvolution Algorithms",
        author="Eric Czech",
        author_email="eric@hammerlab.org",
        url="",
        license="http://www.apache.org/licenses/LICENSE-2.0.html",
        classifiers=[
            'Environment :: Console',
            'Operating System :: OS Independent',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
        ],
        install_requires=requires,
        packages=['flowdec', 'flowdec.nb'],
        package_data={'flowdec': ['datasets/*/*.tif']}
    )
