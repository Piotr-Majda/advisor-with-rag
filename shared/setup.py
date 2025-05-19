from setuptools import setup, find_packages

setup(
    name="shared",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.109.2",
        "python-json-logger>=2.0.0",
        "redis>=5.0.0",
        "pydantic>=2.10.0",
    ],
    python_requires=">=3.11",
    author="Piotr Majda",
    description="Shared utilities for microservices",
    package_data={"shared": ["py.typed"]},
)
