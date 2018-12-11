from setuptools import setup, find_packages


with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

setup(
    name = 'drcme',
    version = '0.1.0',
    description = """dimensionality reduction and clustering for morphology and electrophysiology""",
    author = "Nathan Gouwens",
    author_email = "nathang@alleninstitute.org",
    url = '',
    packages = find_packages(),
    install_requires = required,
    include_package_data=True,
    setup_requires=['pytest-runner'],
)
