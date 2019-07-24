# Contributing to Larq

👍 🎉 First off, thanks for taking the time to contribute! 👍 🎉

**Working on your first Pull Request?** You can learn how from this _free_ series
[How to Contribute to an Open Source Project on GitHub](https://egghead.io/courses/how-to-contribute-to-an-open-source-project-on-github).

## Project setup

To send a Pull Request it is required to fork Larq on GitHub.
After that clone it to a desired directory:

```shell
git clone https://github.com/my-username/larq.git
```

Install all required dependencies for local development by running:

```shell
cd larq # go into the directory you just cloned
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

Installs dependencies for building the docs:

```shell
pip install nbconvert git+https://github.com/lgeiger/pydoc-markdown.git
pip install -e .[docs]
```

Inside the project directory run:

```shell
./larqdocs.sh serve
```

A new version of the documentation will be automatically published once merged into the master branch.

## Code style

We use [`black`](https://black.readthedocs.io/en/stable/) to format all of our code. We recommend installing it as a plugin for your favorite [code editor](https://black.readthedocs.io/en/stable/editor_integration.html).

## Publish release

1. Install dependencies

   ```shell
   python -m pip install --upgrade setuptools wheel twine
   ```

2. Increment the version number in `setup.py`

3. Push new tag

   ```shell
   git tag <version number>
   git push && git push --tags
   ```

4. Build wheels

   ```shell
   rm -r build/* dist/*
   python setup.py sdist bdist_wheel
   ```

5. Upload release

   ```shell
   python -m twine upload dist/*
   ```
