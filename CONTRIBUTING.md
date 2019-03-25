# Contributing to XQuant

👍 🎉  First off, thanks for taking the time to contribute! 👍 🎉 

**Working on your first Pull Request?** You can learn how from this _free_ series
[How to Contribute to an Open Source Project on GitHub](https://egghead.io/courses/how-to-contribute-to-an-open-source-project-on-github).

## Project setup

To send a Pull Request it is required to fork XQuant on GitHub.
After that clone it to a desired directory:

```shell
git clone https://github.com/my-username/xquant.git
```

Install all required dependencies for local development by running:

```shell
cd xquant # go into the directory you just cloned
pip install -e .[tensorflow] # Installs Tensorflow for CPU
# pip install -e .[tensorflow_gpu] # Installs Tensorflow for GPU
pip install -e .[test] # Installs all development dependencies
```

## Run Unit tests

Inside the project directory run:

```shell
pytest .
```

## Build documentation

Inside the project directory run:

```shell
pydocmd serve
```

## Code style

We use [`black`](https://black.readthedocs.io/en/stable/) to format all of our code. We recommend installing it as a plugin for your favorite [code editor](https://black.readthedocs.io/en/stable/editor_integration.html).
