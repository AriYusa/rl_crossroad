from setuptools import setup, find_packages

setup(
    name='jackal_crossroad_env',
    version='0.1.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'gym',
        'numpy',
        'rospkg',
    ],
)
