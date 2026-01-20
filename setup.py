"""
Cloudx Invoice AI
Training and API for invoice text extraction using Donut
"""
from setuptools import setup, find_packages

setup(
    name="cloudx-invoice-ai",
    version="0.1.0",
    description="Cloudx Invoice Document Understanding AI using Donut transformer",
    author="Cloudx",
    python_requires=">=3.8",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "transformers>=4.11.3",
        "timm",
        "datasets[vision]",
        "pytorch-lightning>=1.6.4",
        "nltk",
        "sentencepiece",
        "zss",
        "sconf>=0.2.3",
        "pdf2image>=1.16.0",
        "PyPDF2>=3.0.0",
        "pypdfium2>=4.0.0",
        "Pillow>=9.0.0",
        "opencv-python>=4.5.0",
        "fastapi>=0.95.0",
        "uvicorn[standard]>=0.21.0",
        "python-multipart>=0.0.6",
        "pydantic>=1.10.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "python-dotenv>=0.19.0",
        "pyyaml>=6.0",
        "jsonlines>=3.0.0",
        "scikit-learn>=1.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
