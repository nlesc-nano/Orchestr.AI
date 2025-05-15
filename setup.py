from setuptools import setup, find_packages
  
setup(
    name="mlff_qd",
    version="0.1.0",
    packages=find_packages(),  # Look for packages in the root directory
    entry_points={
        "console_scripts": [
            "run-md-opt=postprocessing.run_md_opt:main",
        ]
    },
    install_requires=[
        # List your dependencies here, e.g., "numpy", "matplotlib", "ase", etc.
    ],
    python_requires=">=3.7",
)

