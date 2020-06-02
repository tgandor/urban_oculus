
"""
This is almost unnecessary, see:

urban_oculus]$ python ~/detectron2/demo/demo.py -h
usage: demo.py [-h] [--config-file FILE] [--webcam]
               [--video-input VIDEO_INPUT] [--input INPUT [INPUT ...]]
               [--output OUTPUT] [--confidence-threshold CONFIDENCE_THRESHOLD]
               [--opts ...]

Detectron2 demo for builtin models

optional arguments:
  -h, --help            show this help message and exit
  --config-file FILE    path to config file
  --webcam              Take inputs from webcam.
  --video-input VIDEO_INPUT
                        Path to video file.
  --input INPUT [INPUT ...]
                        A list of space separated input images; or a single
                        glob pattern such as 'directory/*.jpg'
  --output OUTPUT       A file or directory to save output visualizations. If
                        not given, will show output in an OpenCV window.
  --confidence-threshold CONFIDENCE_THRESHOLD
                        Minimum score for instance predictions to be shown
  --opts ...            Modify config options using the command-line 'KEY
                        VALUE' pairs

(well, the above message works at the time of writing: 2020-06-02 (wto) 23:34)
(it may change in the future - for the better or the worse...)

So, to process a single image just pass the right config to demo.py,
set some options (friendly names, not like cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST),
maybe pass --output and enjoy.

Still, some features are missing from the demo! The list is here:
* using model zoo, i.e. specifying the "ZOO name" and getting
  - the config file loaded
  - pretrained weights, and not transfer learning for training
* saving some "programmatic" form of results (JSON?) of the detections
  (this is understandable, because demo has different models with different output formats)
* ... (more ideas?)
"""

# TODO: maybe copy demo.py's functionality adding the extra stuff
