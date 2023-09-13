"""A setuptools based setup module."""
from os import path
from codecs import open

from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.test import test as TestCommand

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()


class PostDevelopCommand(develop):
    """Post-installation for development mode."""

    def run(self):
        print("TODO: PostDevelopCommand")
        develop.run(self)


class PostInstallCommand(install):
    """Post-installation for installation mode."""

    def run(self):
        print("TODO: PostInstallCommand")
        install.run(self)


class Tox(TestCommand):
    user_options = [("tox-args=", "a", "Arguments to pass to tox")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.tox_args = None

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import tox
        import shlex

        args = self.tox_args
        if args:
            args = shlex.split(self.tox_args)
        tox.cmdline(args=args)


setup(
    name="parsernaam",
    version="0.0.3",
    description=("Name parser"),
    long_description=long_description,
    # The project's main homepage.
    url="https://github.com/appeler/parsernaam",
    # Author details
    author="Rajashekar Chintalapati, Gaurav Sood",
    author_email="rajshekar.ch@gmail.com, gsood07@gmail.com",
    # Choose your license
    license="MIT",
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: MIT License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
    # What does your project relate to?
    keywords="parse names",
    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=["data", "docs", "tests", "scripts"]),
    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],
    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=["torch==2.0.1", "pandas==2.0.3", 
                      "joblib==1.3.2", "scikit-learn==1.3.0",],
    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={"dev": ["check-manifest"], "test": ["coverage"], },
    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={
        "parsernaam": [
            "notebooks/*.ipynb",
            "models/*.pt",
            "models/*.joblib",
        ],
    },
    entry_points={
        "console_scripts": [
            "parse_names = parsernaam.parse:main"
        ],
    },
    cmdclass={
        "develop": PostDevelopCommand,
        "install": PostInstallCommand,
        "test": Tox,
    },
    tests_require=["tox"],
)
