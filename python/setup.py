from setuptools import setup
import logging
import os.path as osp
import os

readme_path = osp.realpath(osp.join(osp.dirname(__file__), '..', 'README.md'))

try:
    with open(readme_path, 'r') as f:
        readme_markdown = f.read()
    logging.info("Successfully loaded readme markdown from %s" % readme_path)
except:
    logging.warning("Failed to load readme markdown from %s" % readme_path)
    readme_markdown = ""

try:
    with open('requirements.txt', 'r') as fd:
        requires = [l.strip() for l in fd.readlines()]
except FileNotFoundError:
    # for when running in a tox virtualenv
    requires = ['scikit-image', 'matplotlib', 'requests']

if __name__ == '__main__':
    setup(
        name='flowdec',
        version='1.0.7',
        description="TensorFlow Implementations of Signal Deconvolution Algorithms",
        long_description=readme_markdown,
        long_description_content_type="text/markdown",
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
        extras_require={
            "tf": ["tensorflow>=1.14.0"],
            "tf_gpu": ["tensorflow-gpu>=1.14.0"],
        },
        packages=['flowdec', 'flowdec.cmd', 'flowdec.nb'],
        package_data={'flowdec': ['datasets/*/*.tif']},
        entry_points={
            'console_scripts': [
                'deconvolution = flowdec.cmd.deconvolution:main'
            ]
        }
    )
