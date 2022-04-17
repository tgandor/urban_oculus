# urban\_oculus

Code used for conducting research (and possible reproduction of results) for the following publications:

* [First Gradually, Then Suddenly: Understanding the Impact of Image Compression on Object Detection Using Deep Learning](https://www.mdpi.com/1424-8220/22/3/1104/htm) (article)
* "Training deep convolutional object detectors for images affected by lossy compression" (chapter)
* "Detekcja obiektów w obrazach cyfrowych z użyciem uczenia głębokiego w warunkach stratnej kompresji obrazu" (thesis)

Extra data can be obtained from data repositories:

* <https://doi.org/10.7910/DVN/UPIKSF>
* <https://doi.org/10.7910/DVN/UHEP3C>

### Changelog of the included python library

Most of the code is in standalone scripts and Jupyter notebooks, but there is also a library of utilities, installable from sources via `python setup.py install` or `pip install .`.

* 0.1.0 - made the detectron dependencies optional, now run pip install -e .\[detectron\]
* 0.0.9 - view_obj <gt_id> command, update detectron2 dep
* 0.0.8 - summary_table; plots by model in evaldets.postprocess
* 0.0.7 - symlink_q from postprocess (and previous changes: Summary, GrandSummary)
* 0.0.5 - added view_gt command, with --scale and --verbose
* 0.0.1 - 0.0.4: jpeg, quantization, quality identification, evaldets, uo.utils
