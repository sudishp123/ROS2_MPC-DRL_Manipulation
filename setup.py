from setuptools import setup, find_packages

setup(
    name = "jetcobot_drl",
    version = "0.1.0",
    packages = find_packages(where="python"),
    package_dir = {"": "python"},
)