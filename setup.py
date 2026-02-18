from setuptools import find_packages, setup
import os

package_name = 'franka_mujoco'

data_files = [
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
]

# HELPER FUNCTION: Recursively find all files in the assets directory
def package_files(directory_list):
    paths_dict = {}
    for directory in directory_list:
        # Walk through all subdirectories
        for (path, directories, filenames) in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(path, filename)
                # Define where to install it (share/package_name/path)
                install_path = os.path.join('share', package_name, path)
                
                if install_path in paths_dict:
                    paths_dict[install_path].append(file_path)
                else:
                    paths_dict[install_path] = [file_path]
    
    # Add them to the data_files list
    for key in paths_dict:
        data_files.append((key, paths_dict[key]))

# Call the helper function for your assets folder
package_files([os.path.join(package_name, 'assets')])

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,  # We use the list we built above
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='faizaladin',
    maintainer_email='faizaladin@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'start_sim = franka_mujoco.simulation:main',
            'detect_box = franka_mujoco.detect_box:main',
            'forward_kin = franka_mujoco.forward_kinematics:main',
            'inverse_kin = franka_mujoco.inverse_kinematics:main',
            'cam_pick_place = franka_mujoco.camera_pick_and_place:main'
        ],
    },
)