from setuptools import setup, find_packages

setup(
    name="rapid_twin",
    version="1.0.0",
    description="Retina-Anchored Physiological Digital Twin (RAPiD-Twin)",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "opencv-python>=4.7.0",
        "scipy>=1.10.0",
        "networkx>=3.0",
        "scikit-image>=0.21.0",
        "scikit-learn>=1.3.0",
    ],
)
