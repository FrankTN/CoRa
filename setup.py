import setuptools as st

st.setup(
    name='cora',
    version='0.1',
    author='Frank te Nijenhuis',
    packages=st.find_packages(),
    description='CLI wrapper for pyradiomics implementing COVID analysis ',
    # py_modules=['corad'],
    install_requires=[
        'setuptools',
        'numpy',
        'pyradiomics',
        'click',
        'scipy',
        'SimpleITK',
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'cora = corad.__main__:main',
        ],
    },
)
