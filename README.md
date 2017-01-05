### Traffic Sign Detection and Classification
This module is an extension of [Faster-RCNN](https://github.com/rbgirshick/py-faster-rcnn) to detect and classify [German Traffic Signs](http://benchmark.ini.rub.de/). The following animation shows the output of this module.

![OUTPUT](output.gif)

For installation, I modified the original Faster-RCNN [README.md](https://github.com/rbgirshick/py-faster-rcnn/blob/master/README.md) file to adapt changes for run this module. Please check below for license and citation information.

### Contents
1. [Requirements: software](#requirements-software)
2. [Requirements: hardware](#requirements-hardware)
3. [Basic installation](#installation-sufficient-for-the-demo)
4. [Demo](#demo)
5. [Beyond the demo: training and testing](#beyond-the-demo-installation-for-training-and-testing-models)
6. [Usage](#usage)

### Requirements: software

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

    **Note:** Caffe *must* be built with support for Python layers!

    ```make
    # In your Makefile.config, make sure to have this line uncommented
    WITH_PYTHON_LAYER := 1
    # Unrelatedly, it's also recommended that you use CUDNN
    USE_CUDNN := 1
    ```

    You can see the sample [Makefile.config](caffe-fast-rcnn/Makefile.config) avialable with this repository. It uses conda with GPU support. You need to modify this file to suit your hardware configuration.
2. Python packages you might not have: `cython`, `python-opencv`, `easydict`
3. [Optional] MATLAB is required for **official** PASCAL VOC evaluation only. The code now includes unofficial Python evaluation code.

### Requirements: hardware

1. For training smaller networks (ZF, VGG_CNN_M_1024) a good GPU (e.g., Titan, K20, K40, ...) with at least 3G of memory suffices
2. For training Fast R-CNN with VGG16, you'll need a K40 (~11G of memory)
3. For training the end-to-end version of Faster R-CNN with VGG16, 3G of GPU memory is sufficient (using CUDNN)

### Installation (sufficient for the demo)

1. Clone the Faster R-CNN repository
    ```Shell
    # Make sure to clone with --recursive
    git clone --recursive https://github.com/sridhar912/tsr-py-faster-rcnn.git
    ```

2. We'll call the directory that you cloned Faster R-CNN into `FRCN_ROOT`

   *Ignore notes 1 and 2 if you followed step 1 above.*

   **Note 1:** If you didn't clone Faster R-CNN with the `--recursive` flag, then you'll need to manually clone the `caffe-fast-rcnn` submodule:
    ```Shell
    git submodule update --init --recursive
    ```
    **Note 2:** The `caffe-fast-rcnn` submodule needs to be on the `faster-rcnn` branch (or equivalent detached state). This will happen automatically *if you followed step 1 instructions*.

3. Build the Cython modules
    ```Shell
    cd $FRCN_ROOT/lib
    make
    ```

4. Build Caffe and pycaffe
    ```Shell
    cd $FRCN_ROOT/caffe-fast-rcnn
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make -j8 && make pycaffe
    ```

5. Download pre-trained detector from this [link](https://drive.google.com/open?id=0B0CHhxRP_jmIRlVKR250d0pMNEE). This downloaded model need to be placed under the directory
	```Shell
	$FRCN_ROOT/data/GTSDB/TrainedModel
	```
### Demo

*After successfully completing [basic installation](#installation-sufficient-for-the-demo)*, you'll be ready to run the demo.

To run the demo
```Shell
cd $FRCN_ROOT
./tools/demo.py
```
The demo performs detection using a ZF network trained for detection on GTSDB. Few sample images from [test data set](http://benchmark.ini.rub.de/Dataset_GTSDB/TestIJCNN2013.zip) has been placed under folder
```Shell
cd $FRCN_ROOT/data/demo
```
For testing the complete test dataset, [test dataset](http://benchmark.ini.rub.de/Dataset_GTSDB/TestIJCNN2013.zip) has to be download and placed in the folder mentioned above

### Beyond the demo: installation for training and testing models

Before starting, you need to download the traffic sign datasets from [German Traffic Signs Datasets](http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset). In this implementation, the training and test datasets that were used for the competition ( [training data set](http://benchmark.ini.rub.de/Dataset_GTSDB/TrainIJCNN2013.zip) (1.1 GB), [test data set](http://benchmark.ini.rub.de/Dataset_GTSDB/TestIJCNN2013.zip) (500 MB) ) is used.

Here, the main goal is to enable Faster R-CNN to detect and classify traffic sign. So, model performance evaluation in test dataset was not carried out. The downloaded test dataset was only used for visual testing. After the dataset is downloaded, prepare the following directory structure. The training zip file contains the following files
- folders
- images (00000.ppm, 00001.ppm...., 00599.ppm)
- gt.txt

Copy all the images into Images directory as shown below. Rename gt.txt as train.txt and keep both gt.txt and train.txt as shown below. 

##### Format Your Dataset
At first, the dataset must be well organzied with the required format.
```
GTSDB
|-- Annotations
    |-- gt.txt (Annotation files)
|-- Images
    |-- *.ppm (Image files)
|-- ImageSets
    |-- train.txt
```

##### Download pre-trained ImageNet models

Pre-trained ImageNet models can be downloaded for the three networks described in the paper: ZF and VGG16.

```Shell
cd $FRCN_ROOT
./data/scripts/fetch_imagenet_models.sh
```
VGG16 comes from the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo), but is provided here for your convenience.
ZF was trained at MSRA.

### Usage
This implementation is tested only for approximate joint training.

To train and test a TSR Faster R-CNN detector using the **approximate joint training** method, use `experiments/scripts/faster_rcnn_end2end.sh`.
Output is written underneath `$FRCN_ROOT/output`.

```Shell
cd $FRCN_ROOT
./experiments/scripts/faster_rcnn_end2end.sh [GPU_ID] [NET] [--set ...] [DATASET]
# GPU_ID is the GPU you want to train on
# NET in {ZF, VGG_CNN_M_1024, VGG16} is the network arch to use
# --set ... allows you to specify fast_rcnn.config options, e.g.
# --set EXP_DIR seed_rng1701 RNG_SEED 1701
# DATASET to be used for training
```
Example script to train ZF model:
```Shell
cd $FRCN_ROOT
./experiments/scripts/faster_rcnn_end2end.sh 0 ZF gtsdb
```
Trained Fast R-CNN networks are saved under:
```
output/<experiment directory>/<dataset name>/
# Example: output/faster_rcnn_end2end/gtsdb_train
```
Test outputs are saved under:
```
output/<experiment directory>/<dataset name>/<network snapshot name>/
```

### Disclaimer

The official Faster R-CNN code (written in MATLAB) is available [here](https://github.com/ShaoqingRen/faster_rcnn).
If your goal is to reproduce the results in our NIPS 2015 paper, please use the [official code](https://github.com/ShaoqingRen/faster_rcnn).

This repository contains a Python *reimplementation* of the MATLAB code.
This Python implementation is built on a fork of [Fast R-CNN](https://github.com/rbgirshick/fast-rcnn).
There are slight differences between the two implementations.
In particular, this Python port
 - is ~10% slower at test-time, because some operations execute on the CPU in Python layers (e.g., 220ms / image vs. 200ms / image for VGG16)
 - gives similar, but not exactly the same, mAP as the MATLAB version
 - is *not compatible* with models trained using the MATLAB code due to the minor implementation differences
 - **includes approximate joint training** that is 1.5x faster than alternating optimization (for VGG16) -- see these [slides](https://www.dropbox.com/s/xtr4yd4i5e0vw8g/iccv15_tutorial_training_rbg.pdf?dl=0) for more information

# *Faster* R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

By Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun (Microsoft Research)

This Python implementation contains contributions from Sean Bell (Cornell) written during an MSR internship.

Please see the official [README.md](https://github.com/ShaoqingRen/faster_rcnn/blob/master/README.md) for more details.

Faster R-CNN was initially described in an [arXiv tech report](http://arxiv.org/abs/1506.01497) and was subsequently published in NIPS 2015.

### License

Faster R-CNN is released under the MIT License (refer to the LICENSE file for details).

### Citing Faster R-CNN

If you find Faster R-CNN useful in your research, please consider citing:

    @inproceedings{renNIPS15fasterrcnn,
        Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
        Title = {Faster {R-CNN}: Towards Real-Time Object Detection
                 with Region Proposal Networks},
        Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
        Year = {2015}
    }
