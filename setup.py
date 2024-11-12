from setuptools import setup, find_packages

setup(
    name='deeppliv',
    version='0.1.0',
    description='A neural network-based two-stages package for partially-linear IV settings',
    author='Tomer Weiss',
    author_email='Tomerweiss248@gmail.com',
    packages=find_packages(),
    install_requires=[
        'torch',  # or tensorflow, depending on your implementation
        'numpy',
        'scikit-learn',

        # add other dependencies
    ],
    python_requires='>=3.10',
)