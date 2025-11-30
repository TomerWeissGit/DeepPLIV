from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='deeppliv',
    version='0.1.0',
    description='Neural network-based two-stage package for causal inference in partially-linear instrumental variable settings',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Tomer Weiss',
    author_email='Tomerweiss248@gmail.com',
    url='https://github.com/tomerweiss/deeppliv',
    packages=find_packages(exclude=['examples', 'examples.*', 'tests', 'tests.*']),
    install_requires=[
        'numpy>=2.0.2',
        'torch>=2.5.1',
        'matplotlib>=3.9.2',
        'pandas>=2.2.3',
        'seaborn>=0.13.2',
        'scikit-learn>=1.5.2',
        'scipy>=1.13.1',
        'typing_extensions>=4.12.2',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0',
            'pytest-cov>=4.0',
        ],
        'examples': [
            'optuna>=4.1.0',
            'scikit-optimize>=0.10.2',
            'statsmodels>=0.14.5',
        ],
        'aws': [
            'boto3>=1.40.18',
            'aioboto3>=7.0.0',
            'aiofiles>=24.1.0',
            'python-dotenv>=1.1.1',
        ],
    },
    python_requires='>=3.10',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    license='MIT',
    keywords='causal inference, instrumental variables, deep learning, partially linear models',
)