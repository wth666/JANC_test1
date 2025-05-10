from setuptools import setup, find_packages

setup(
    name="janc",
    version="0.1",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'cantera',
        'jaxamr @ git+https://github.com/JA4S/JAX-AMR.git'
    ],
    python_requires=">=3.7"
)
