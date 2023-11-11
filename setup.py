from setuptools import setup

setup(
    name='SeriesSyndex',
    version='0.1.0',    
    description='Python package to evaluate the quality of synthetic time-series data.',
    url='https://github.com/vikram2000b',
    author='Vikram Singh Chundawat',
    author_email='vikram2000b@gmail.com',
    license='BSD 2-clause',
    packages=['SeriesSyndex'],
    install_requires=['numpy',                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)