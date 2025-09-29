from setuptools import find_packages, setup

package_name = 'cvtest'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='intel',
    maintainer_email='pauaup@naver.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'cvtest=cvtest.cvtest:main',
            'labslider=cvtest.labslider:main',
						'tmp=cvtest.tmp:main
        ],
    },
)
