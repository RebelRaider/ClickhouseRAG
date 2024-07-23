from setuptools import find_packages, setup

setup(
    name="clickhouserag",
    version="0.1.0",
    description="A Python library for creating RAG with Clickhouse.",
    author="Leonid Chesnikov",
    author_email="leonid.chesnikov@gmail.com",
    url="https://github.com/RebelRaider/ClickhouseRAG",
    packages=find_packages(),
    install_requires=[
        "clickhouse-driver>=0.2.0",
        # add other dependencies here
    ],
    extras_require={
        "dev": [
            "pytest>=8.3.1",
            "ruff>=0.5.4",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
