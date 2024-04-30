from setuptools import setup

setup(
    name='attack_time',
    version='1.0',
    author="Alberto Acquilino",
    author_email='alberto.acquilino@mail.mcgill.ca',
    description='Attack time detection for monophonic music sounds',
    # packages=['attack_time'],
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'librosa'
    ],
)
