from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'autoware_forecast_mae_prediction'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(where='src'),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name), glob("launch/*.launch.xml"))
        
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ahsan Ahmed',
    maintainer_email='ahsan@scaledrive.ai',
    description='Machine learning based trajectory prediction module',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "autoware_forecast_mae = autoware_forecast_mae_prediction.autoware_forecast_mae:main"
        ],
    },
    package_dir={"": "src"},
)
