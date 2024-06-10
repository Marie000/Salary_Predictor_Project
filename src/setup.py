from setuptools import setup, find_packages
import os
import io
from pathlib import Path

NAME = "predictor_model"
URL = "https://github.com/Marie000/Salary_Predictor_Project"
EMAIL = "marie.pelletier@gmail.com"
AUTHOR = "Marie Pelletier"
REQUIRES_PYTHON = ">3.10"


pwd = os.path.abspath(os.path.dirname(__file__))


def list_requirements(fname="requirements.txt"):
    with io.open(os.path.join(pwd, fname), encoding="utf-8") as f:
        return f.read().splitlines()


about = {}
with open(Path(__file__).resolve().parent / "predictor_model" / "VERSION") as f:
    _version = f.read().strip()
    about["__version__"] = _version

setup(
    name=NAME,
    version=about["__version__"],
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests")),
    package_data={"predictor_model": ["VERSION"]},
    install_requires=list_requirements(),
    include_package_data=True,
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)
