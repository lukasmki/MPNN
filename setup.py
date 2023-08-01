import setuptools
import mpnn

if __name__=="__main__":
    setuptools.setup(
        name="MPNN",
        version=mpnn.__version__,
        description="Message Passing Neural Network Implementation",
        packages = setuptools.find_packages(),
    )