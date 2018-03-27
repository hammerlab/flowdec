from setuptools import setup


with open('requirements.txt', 'r') as fd:
    requires = [l.strip() for l in fd.readlines()]

if __name__ == '__main__':
    setup(
        name='flowdec',
        version='0.0.1',
        description="Tensorflow Deconvolution for Microscopy Data",
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
        package_data={},
        install_requires=requires,
        packages=['flowdec']
    )
