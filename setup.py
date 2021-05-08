from setuptools import find_packages, setup

setup(
    name="ml_project",
    packages=find_packages(),
    version="0.1.0",
    description="First homework",
    author="Vakha Gaparkhoev",
    install_requires=[
        "python-dotenv>=0.5.1",
        "scikit-learn==0.24.2",
        "pyyaml>=3.11",
        "marshmallow-dataclass==8.3.0",
        "pandas==1.2.4",
        "Hydra==2.5",
        "hydra-core==1.1.0.dev6",
        "pytest==6.2.4",
        "Faker==8.1.2",
        "dacite==1.6.0",
        "pytest-order==0.11.0"
    ],
    license="MIT",
)
