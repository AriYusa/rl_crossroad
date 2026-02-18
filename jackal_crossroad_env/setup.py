from setuptools import setup, find_packages

setup(
    name='jackal_crossroad_env',
    version='0.1.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    scripts=['scripts/train_main.py', 'scripts/visualize_metrics.py'],
    install_requires=[
        'gym',
        'numpy',
        'matplotlib',
        'rospkg',
    ],
)
