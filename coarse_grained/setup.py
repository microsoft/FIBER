from setuptools import setup, find_packages

setup(
    name="fiber",
    packages=find_packages(exclude=[".dfc", ".vscode", "dataset", "notebooks", "result", "scripts"]),
    version="0.1.0",
    keywords=["vision and language pretraining"],
    install_requires=["torch", "pytorch_lightning"],
)
