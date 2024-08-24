from setuptools import setup, find_packages

setup(
    name='quadrilateral-fitter',
    version='1.9',
    author='Eric-Canas',
    author_email='eric@ericcanas.com',
    url='https://github.com/Eric-Canas/quadrilateral-fitter',
    description='QuadrilateralFitter is an efficient and easy-to-use Python library for fitting irregular '
                'quadrilaterals from irregular polygons or any noisy data.',
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    keywords='quadrilateral, fitter, polygon, shape analysis, geometry',
    platforms='any',
    license='MIT',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'shapely',
        'numpy',
        'scipy',
    ],

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Utilities',
    ],
)