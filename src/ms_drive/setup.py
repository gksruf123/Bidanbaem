from setuptools import find_packages, setup

package_name = 'ms_drive'

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
            'nav=ms_drive.nav:main',
            'odom_bridge=ms_drive.odom_tf_bridge:main',
            'p2odom=ms_drive.p2odom:main',
            'driver=ms_drive.driver:main'
        ],
    },
)
