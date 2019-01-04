from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='mirror_symmetry',
      version='1.1.2',
      description='Tools to detect mirror symmetry in 3D image data.',
      long_description=readme(),
      classifiers=[
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Topic :: 3D image Processing :: Symmetry'],
      keywords='mirror symmetry',
      url='http://github.com/',
      author='David Drobny',
      author_email='DaDrobny@gmail.com',
      license='MIT',
      packages=['mirror_symmetry'],
      install_requires=['numpy>=1.15.0', 'nibabel'],
      entry_points={
        'console_scripts': ['mirror_symmetry_tool = '
                            'mirror_symmetry.command:process']},
      zip_safe=False)
