from setuptools import setup, find_packages

setup(
    name='spinlife',
    version='0.1.0',
    description='VASP PROCAR spin lifetime calculator',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20',
    ],
    extras_require={
        'plot': ['matplotlib>=3.0'],
    },
    entry_points={
        'console_scripts': [
            'spinlife = spinlife.main:main',
        ],
    },
)
