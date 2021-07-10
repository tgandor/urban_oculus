from setuptools import setup

setup(
    name="urban_oculus",
    version="0.0.8",
    description="Misc utils for object detection and image compression.",
    packages=["evaldets", "jpeg", "uo"],
    install_requires=[
        "opencv-python",
        "pycocotools @ git+https://github.com/tgandor/cocoapi.git@reformatted#egg=pycocotools&subdirectory=PythonAPI",  # noqa
        "detectron2 @ git+https://github.com/facebookresearch/detectron2.git@e1356b1ee79ad2e7f9739ad533250e24d4278c30",  # noqa
    ],
    dependency_links=[
        "git+https://github.com/tgandor/cocoapi.git@reformatted#egg=pycocotools&subdirectory=PythonAPI",  # noqa
        "git+https://github.com/facebookresearch/detectron2.git@e1356b1ee79ad2e7f9739ad533250e24d4278c30",  # noqa
    ],
    entry_points=dict(
        console_scripts=[
            "view_detections=evaldets.visualization:_main",
            "view_gt=evaldets.visualization:_show_gt",
            "symlink_q=evaldets.postprocess:_symlink_q",
            "summary_table=evaldets.postprocess:_main"
        ]
    ),
)
