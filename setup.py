from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="Topsis-Yashvi-102203529",  # Replace with your package name
    version="1.0.0",
    author="Ayushi",
    author_email="ayushisaluja01@gmail.com",  # Replace with your email
    description="A Python package for TOPSIS method for multi-criteria decision making",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Topsis-Yashvi-102203529",  # Replace with your GitHub repository URL
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "pandas",
        "numpy",
    ],
    entry_points={
        "console_scripts": [
            "topsis=topsis.topsis:main",  # Allows users to run it via command line
        ],
    },
)