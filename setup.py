"""
setup.py
========

Classic setuptools-based packaging script for POSITMJ.

This works alongside (and is kept in sync with) pyproject.toml, which is the
modern, PEP 517/518-compliant way most tools (pip, build, twine) now read
package metadata from. Both are provided so the project builds correctly
whether a tool reads pyproject.toml or falls back to setup.py.

Build & publish:
    pip install build twine
    python -m build
    twine upload dist/*
"""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="POSITMJ",
    version="0.1.0",
    author="Mihir and Jiyan",
    description=(
        "Train MLPs and export their weights in Posit number format "
        "for FPGA/ASIC hardware deployment"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/POSITMJ",
    packages=find_packages(),
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
    ],
    extras_require={
        "test": ["pytest", "softposit"],
    },
    keywords=[
        "posit", "mlp", "neural-network", "fpga", "asic",
        "verilog", "hardware", "quantization",
    ],
)
