from distutils.core import setup

setup(
    name="carbs",
    version="0.1.0",
    author="Untitled AI",
    author_email="abe@generallyintelligent.com",
    packages=["carbs"],
    license="LICENSE.txt",
    description="CARBS hyperparameter tuning algorithm",
    long_description=open("README.md").read(),
    install_requires=[
        "pyro-ppl>=1.6.0",
        "torch>=1.8.1",
        "loguru>=0.5.3",
        "cattrs>=1.3.0",
        "numpy",
        "wandb",
        "seaborn",
    ],
)
