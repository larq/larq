from setuptools import setup, find_packages


def readme():
    with open("README.md", "r") as f:
        return f.read()


setup(
    name="pl-xquant",
    version="0.0.0",
    author="Plumerai",
    author_email="lukas@plumerai.co.uk",
    description="An Open Source Machine Learning Framework for Training Extreme Quantized Neural Networks",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/lgeiger/xquant",
    packages=find_packages(),
    license="Apache 2.0",
    install_requires=["numpy >= 1.15.4, < 2.0"],
    extras_require={
        "tensorflow": ["tensorflow>=1.13.1"],
        "tensorflow_gpu": ["tensorflow-gpu>=1.13.1"],
        "test": ["absl-py>=0.7.0", "pytest>=4.3.1", "pytest-cov>=2.6.1"],
        "docs": [
            "pydoc-markdown@https://github.com/lgeiger/pydoc-markdown/archive/master.zip",
            "mkdocs-material>=4.1.0",
            "pymdown-extensions>=6.0",
            "mknotebooks>=0.1.5",
        ],
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
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
