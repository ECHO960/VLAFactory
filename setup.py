from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vlafactory",
    version="0.1.0",
    author="VLAFactory Team",
    description="A minimal framework for training Vision-Language-Action models based on LlamaFactory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ECHO960/VLAFactory",
    packages=find_packages("src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "accelerate>=0.20.0",
        "peft>=0.4.0",
        "trl>=0.4.7",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
        "vision": [
            "pillow>=9.0.0",
            "timm>=0.9.0",
        ],
        "llamafactory": [
            "llamafactory @ git+https://github.com/hiyouga/LlamaFactory.git",
        ],
    },
)
