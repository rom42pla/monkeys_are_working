import setuptools

with open("README.md", "r", encoding="utf-8") as fp:
    long_description = fp.read()

with open("requirements.txt", "r", encoding="utf-8") as fp:
    requirements = [s for s in fp.read().split("\n") if s]

setuptools.setup(
    name="monkeys_are_working",
    version="0.0.2",
    author="Romeo Lanzino",
    author_email="romeo.lanzino@gmail.com",
    description='Testing installation of Package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rom42pla/monkeys_are_working",
    license="MIT",
    packages=["monkeys_are_working"],
    install_requires=requirements
)
