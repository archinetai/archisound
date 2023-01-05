from setuptools import find_packages, setup

setup(
    name="archisound",
    packages=find_packages(exclude=[]),
    version="0.0.2",
    license="MIT",
    description="ArchiSound",
    long_description_content_type="text/markdown",
    author="Flavio Schneider",
    author_email="archinetai@protonmail.com",
    url="https://github.com/archinetai/archisound",
    keywords=["artificial intelligence", "deep learning"],
    install_requires=[
        "torch>=1.6",
        "data-science-types>=0.2",
        "transformers",
        "audio-diffusion-pytorch",
        "audio-encoders-pytorch",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)
