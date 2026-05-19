from setuptools import find_packages, setup


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="larq",
    version="0.13.3",
    python_requires=">=3.10",
    author="Plumerai",
    author_email="opensource@plumerai.com",
    description="An Open Source Machine Learning Library for Training Binarized Neural Networks",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://larq.dev/",
    packages=find_packages(exclude=["larq.snapshots"]),
    license="Apache 2.0",
    install_requires=[
        "numpy>=1.15.4,<2.0",
        "terminaltables>=3.1.0",
        "packaging>=19.2",
    ],
    extras_require={
        "tensorflow": ["tensorflow>=2.8.4"],
        "tensorflow_gpu": ["tensorflow-gpu>=2.8.4"],
        "test": [
            "pytest>=7.4",
            "pytest-cov>=4.0",
            "pytest-xdist>=3.4",
            "pytest-mock>=3.11",
            "snapshottest==0.6.*",
            "pytype==2024.10.11",
            "ruff==0.15.13",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
