import setuptools
import mpnn

with open("README.md", "r") as f:
    long_description = f.read()

if __name__ == "__main__":
    setuptools.setup(
        name=mpnn.__name__,
        version=mpnn.__version__,
        description="Message Passing Neural Network Implementation",
        long_description=long_description,
        packages=["mpnn"],
        install_requires=[
            "numpy",
            "torch",
        ],
        zip_safe=False,
    )
