from setuptools import setup, find_packages

setup(
    name='trapezoid-fitter',
    version='1.0',
    author='Eric-Canas',
    author_email='eric@ericcanas.com',
    url='https://github.com/Eric-Canas/trapezoid-fitter',
    description='Trapezoid Fitter is an efficient and easy to use algorithm for fitting'
                ' trapezoids from irregular polygons.',
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'scipy',
        'shapely'
    ],

    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities',
    ],
)