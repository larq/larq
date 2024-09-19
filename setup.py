from setuptools import find_packages, setup


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="larq",
    version="0.13.3",
    python_requires=">=3.7",
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
        "importlib-metadata >= 2.0, < 4.0 ; python_version<'3.8'",
        "packaging>=19.2",
    ],
    extras_require={
        "tensorflow": ["tensorflow>=1.14.0"],
        "tensorflow_gpu": ["tensorflow-gpu>=1.14.0"],
        "test": [
            "pytest>=7.4,<8.2",
            "pytest-cov>=4.0,<4.2",
            "pytest-xdist>=3.4,<3.6",
            "pytest-mock>=3.11,<3.15",
            "snapshottest==0.6.*",
        ],
        "lint": [
            "black==24.4.2",
            "flake8==6.0.*",
            "isort==5.13.*",
            "pytype==2024.4.11",
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
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
