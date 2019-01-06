from setuptools import setup
import os

if os.getenv('TF_GPU', 'false') == 'true':
    requires = ['tensorflow-gpu>=1.6.0']
else:
    requires = ['tensorflow>=1.6.0']

try:
    with open('requirements.txt', 'r') as fd:
        requires += [l.strip() for l in fd.readlines()]
except FileNotFoundError:
    # for when running in a tox virtualenv
    requires += ['scikit-image', 'matplotlib', 'requests']

if __name__ == '__main__':
    setup(
        name='flowdec',
        version='0.0.1',
        description="TensorFlow Implementations of Signal Deconvolution Algorithms",
        author="Eric Czech",
        author_email="eric@hammerlab.org",
        url="https://github.com/hammerlab/flowdec",
        license="http://www.apache.org/licenses/LICENSE-2.0.html",
        classifiers=[
            'Environment :: Console',
            'Operating System :: OS Independent',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3.6'
        ],
        install_requires=requires,
        packages=['flowdec', 'flowdec.cmd', 'flowdec.nb'],
        package_data={'flowdec': ['datasets/*/*.tif']},
        entry_points={
            'console_scripts': [
                'deconvolution = flowdec.cmd.deconvolution:main'
            ]
        }
    )
