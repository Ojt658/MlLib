from setuptools import find_packages, setup

setup(
    name='mllib',
    packages=find_packages(include=['mllib']),
    version='0.1.0',
    description='A python library for generic machine learning algorithms',
    author='Ollie Thomas',
    license='MIT',
    install_requires=['numpy==1.19.2', 'pandas==1.1.3'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests'
)
