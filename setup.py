descr = """A package to tune hyperparameter for scikit-learn models, then
           ensemble them
        """

DISTNAME = 'hyperemble'
DESCRIPTION = 'Tune hyperparameter, then do ensemble'
LONG_DESCRIPTION = descr
MAINTAINER = 'Hoang Duong'
URL = 'https://github.com/hduongtrong/hyperemble'
LICENSE = 'BSD License'
DOWNLOAD_URL = 'https://github.com/hduongtrong/hyperemble'
VERSION = '0.1dev'

INSTALL_REQUIRES = [
                    'numpy>=1.10.1',
                    'scipy>=0.17.0'
                    'pytest>=2.9.1',
                    'scikit_learn>=0.16.1',
                    'keras>=1.0.0'
                    ]

TESTS_REQUIRE = [
                'coverage>=4.0.3',
                'coveralls>=1.1',
                'pytest-cov>=2.2.1',
                'pytest-flakes>=1.0.1',
                'pytest-pep8>=1.0.6',
                ]


if __name__ == "__main__":
    from setuptools import setup
    setup(
          name=DISTNAME,
          version=VERSION,
          license=LICENSE,
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          maintainer=MAINTAINER,
          url=URL,
          download_url=DOWNLOAD_URL,

          classifiers=[
                      'Development Status :: 3 - Alpha',
                      'Environment :: Console',
                      'Intended Audience :: Developers',
                      'Intended Audience :: Science/Research',
                      'License :: OSI Approved :: BSD License',
                      'Programming Language :: Python',
                      'Programming Language :: Python :: 2.7',
                      'Programming Language :: Python :: 3.4',
                      'Topic :: Scientific/Engineering',
                      'Operating System :: Microsoft :: Windows',
                      'Operating System :: POSIX',
                      'Operating System :: Unix',
                      'Operating System :: MacOS',
                      ],

          install_requires=INSTALL_REQUIRES,
          tests_require=TESTS_REQUIRE,

          packages=["hyperemble"],
          )
