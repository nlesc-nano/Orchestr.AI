from setuptools import setup, find_packages

setup(
    name="orchestr_ai",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"orchestr_ai.qd": ["data/*.npz"]},
)
