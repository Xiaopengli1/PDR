"""
PDR (Personalized Deep Research) - A User-Centric Framework for Knowledge Discovery.
SIGIR '26 Resource Track. https://github.com/Xiaopengli1/PDR
"""
from setuptools import find_packages, setup

setup(
    name="pdr",
    version="0.1.0",
    py_modules=["deepsearcher"],
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests", "examples"]),
    install_requires=[
        "argparse",
        "firecrawl-py",
        "langchain_text_splitters",
        "pdfplumber",
        "pymilvus[model]",
        "openai",
        "numpy",
        "tqdm",
        "termcolor",
        "fastapi",
        "uvicorn",
        "pydantic-settings",
        "rouge-score",
        "nltk",
        "pyyaml",
    ],
    entry_points={
        "console_scripts": ["deepsearcher=deepsearcher.cli:main"],
    },
    description="Personalized Deep Research (PDR): A User-Centric Framework for Knowledge Discovery",
    author="Xiaopeng Li et al.",
    author_email="",
    url="https://github.com/Xiaopengli1/PDR",
    keywords=["deep research", "personalization", "RAG", "information retrieval"],
)
