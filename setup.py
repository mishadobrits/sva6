import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sva6",
    version="0.1.2.dev1",
    author="Misha D.",
    description="A small package for accelerating video",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/pypa/sampleproject",  # todo
    # project_urls={
    #     "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    # },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    package_data={'': [
        'requirements.txt',
        'src/sva6/time_stretch/README.md',
        'src/sva6/time_stretch/requirements.txt',
    ]},
    include_package_data=True,
    packages=setuptools.find_packages(where="src"),
    install_requires=[
    ],
    python_requires=">=3.5",
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)

# import pip
# pip.main(["install", "-r", "requirements.txt"])
