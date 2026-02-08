from setuptools import setup
import os
from glob import glob

package_name = "tandem_girona"

setup(
    name=package_name,
    version="0.0.0",
    py_modules=["down_camera_localization"],
    package_dir={"": "src"},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        # si usas meshes en tandem_girona:
        (os.path.join("share", package_name, "meshes"), glob("meshes/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="cirtesu",
    maintainer_email="cirtesu@todo.todo",
    description="Down camera ArUco localization + EKF config",
    license="TODO",
    entry_points={
        "console_scripts": [
            "down_camera_localization = down_camera_localization:main",
        ],
    },
)
