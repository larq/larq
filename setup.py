from setuptools import setup, find_packages


def readme():
    with open("README.md", "r") as f:
        return f.read()


setup(
    name="larq",
    version="0.5.0",
    python_requires=">=3.6",
    author="Plumerai",
    author_email="lukas@plumerai.co.uk",
    description="An Open Source Machine Learning Library for Training Binarized Neural Networks",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://larq.dev/",
    packages=find_packages(exclude=["larq.snapshots"]),
    license="Apache 2.0",
    install_requires=["numpy >= 1.15.4, < 2.0", "terminaltables>=3.1.0"],
    extras_require={
        "tensorflow": ["tensorflow>=1.13.1"],
        "tensorflow_gpu": ["tensorflow-gpu>=1.13.1"],
        "test": [
            "absl-py==0.7.1",
            "pytest==5.0.1",
            "pytest-cov==2.7.1",
            "snapshottest==0.5.0",
        ],
        "docs": [
            "mkdocs==1.0.4",
            "mkdocs-material==4.4.0",
            "pymdown-extensions==6.0",
            "mknotebooks==0.1.6",
            "matplotlib==3.1.1",
            "scour==0.37",
            "mkdocs-minify-plugin==0.2.1",
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
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
