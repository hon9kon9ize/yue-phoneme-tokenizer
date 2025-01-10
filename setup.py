from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md")) as f:
    long_description = f.read()

with open(path.join(here, "yue_phoneme_tokenizer/version.py")) as f:
    exec(f.read())

setup(
    name="yue_phoneme_tokenizer",
    version=__version__,
    description="粵語拼音自動標註工具 Cantonese Pronunciation Automatic Labeling Tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hon9kon9ize/yue-phoneme-tokenizer",
    author="hon9kon9ize",
    author_email="joseph.cheng@hon9kon9ize.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="cantonese g2p phoneme tokenizer",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8, <4",
    entry_points={},
    project_urls={
        "Bug Reports": "https://github.com/hon9kon9ize/yue-phoneme-tokenizer/issues",
        "Source": "https://github.com/hon9kon9ize/yue-phoneme-tokenizer",
    },
    zip_safe=False,
)
