from setuptools import find_packages, setup


def readme():
    with open("README.md", "r") as f:
        return f.read()


setup(
    name="larq",
    version="0.11.2",
    python_requires=">=3.6",
    author="Plumerai",
    author_email="opensource@plumerai.com",
    description="An Open Source Machine Learning Library for Training Binarized Neural Networks",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://larq.dev/",
    packages=find_packages(exclude=["larq.snapshots"]),
    license="Apache 2.0",
    install_requires=[
        "numpy >= 1.15.4, < 2.0",
        "terminaltables>=3.1.0",
        "dataclasses ; python_version<'3.7'",
        "importlib-metadata >= 2.0, < 4.0 ; python_version<'3.8'",
    ],
    extras_require={
        "tensorflow": ["tensorflow>=1.14.0"],
        "tensorflow_gpu": ["tensorflow-gpu>=1.14.0"],
        "test": [
            "black==20.8b1",
            "flake8>=3.7.9,<3.9.0",
            "isort==5.7.0",
            "packaging>=19.2,<21.0",
            "pytest>=5.2.4,<6.3.0",
            "pytest-cov>=2.8.1,<2.12.0",
            "pytest-xdist>=1.30,<2.3",
            "pytest-mock>=2.0,<3.6",
            "pytype==2020.10.8",
            "snapshottest>=0.5.1,<0.7.0",
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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
