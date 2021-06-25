from setuptools import setup

setup(
    name="urban_oculus",
    version="0.0.5",
    description="Misc utils for object detection and image compression.",
    packages=["evaldets", "jpeg", "uo"],
    entry_points=dict(
        console_scripts=[
            'view_detections=evaldets.visualization:_main',
            'view_gt=evaldets.visualization:_show_gt',
        ]
    )

)
