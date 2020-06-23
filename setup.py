import setuptools as st

st.setup(
    name='cora',
    version='0.1',
    author='Frank te Nijenhuis',
    description='CLI wrapper for pyradiomics implementing COVID analysis ',
    packages=st.find_packages(),
    install_requires=[
        'setuptools',
        'pywavelets>=1.1.1',
        'numpy',
        # 'pyradiomics',
        'click',
        'radiomics',
        'scipy',
        'SimpleITK',
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'cora = __main__:cora',
        ],
    },
)
