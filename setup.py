from setuptools import setup


def read(path):
    with open(path) as f:
        return f.read()


setup(
    name="overlapy",
    version="0.0.1",
    description="Library to help compute textual overlap.",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/nlx-group/overlapy",
    author="Ruben Branco, Luís Gomes",
    author_email="rmbranco@fc.ul.pt",
    license="MIT",
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Text Processing :: Filters",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
    ],
    install_requires=["stringology"],
    keywords="text tool",
    package_dir={"": "."},
    py_modules=["overlapy"],
)
