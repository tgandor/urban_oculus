from setuptools import setup

setup(
    name="urban_oculus",
    version="0.1.0",
    description="Misc utils for object detection and image compression.",
    packages=["evaldets", "jpeg", "uo"],
    extras_require={
        "detectron": [
            "opencv-python",
            "pycocotools @ git+https://github.com/tgandor/cocoapi.git@reformatted#egg=pycocotools&subdirectory=PythonAPI",  # noqa
            "detectron2 @ git+https://github.com/facebookresearch/detectron2.git@bf358d61a8949d0cc815249f3f1d4b1d45cd11a0",  # noqa
        ]
    },
    dependency_links=[
        "git+https://github.com/tgandor/cocoapi.git@reformatted#egg=pycocotools&subdirectory=PythonAPI",  # noqa
        "git+https://github.com/facebookresearch/detectron2.git@bf358d61a8949d0cc815249f3f1d4b1d45cd11a0",  # noqa
    ],
    entry_points=dict(
        console_scripts=[
            "view_detections=evaldets.visualization:dt_main",
            "view_gt=evaldets.visualization:gt_main",
            "view_obj=evaldets.visualization:one_gt_main",
            "symlink_q=evaldets.postprocess:symlink_q_main",
            "summary_table=evaldets.postprocess:baseline_table_main",
        ]
    ),
)
