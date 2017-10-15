# gameSummarier
Automate the creation of sports game summaries. W210 capstone project MIDS@UCBerkeley

# Installation steps on Windows 10
Download and install Anaconda 64bit 4.3.1 from here: https://repo.continuum.io/archive/

Optional (if you have an NVidia GPU card): Download and install CUDA (http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)

    install Base CUDA installer

    install patch 2 (Released Jun 26, 2017)

    check and upgrade you display driver (CUDA may have downgraded your driver) http://www.nvidia.com/Download/Scan.aspx

If you have multiple Python environments switch to the newly installed Anaconda:

    create a new environment with python 3.5 and anaconda: 

        conda create -n capstone python=3.5 anaconda


    list environments:

        conda info --envs

    activate environment:

        activate capstone

        python -V   #(to show the version)    
    
    # To remove an environment:
    #    conda remove -n capstone -all

Run "conda install mingw libpython"

Install Tensorflow (with GPU and CUDA):

        Install cuDNN v6.0 for CUDA 8.0 from here https://developer.nvidia.com/rdp/cudnn-download#a-collapse7-8

            Extract the zip to %somewhere% and add %somewhere%/cuda/bin/ to the path

        # pip install --ignore-installed --upgrade tensorflow-gpu
        conda install -c conda-forge tensorflow-gpu

    or without GPU:

        pip install --ignore-installed --upgrade tensorflow

    Test the tensorflow installation:

        import tensorflow as tf

        hello = tf.constant('Hello, TensorFlow!')

        sess = tf.Session()

        print(sess.run(hello))

    output should be:

        "Hello, TensorFlow!"

Install keras:

    conda install -c conda-forge keras

Install opencv from here:
https://sourceforge.net/projects/opencvlibrary/files/latest/download?source=files
set environment variable to install root: e.g.: 
SETX -m OPENCV_DIR C:\Utils\cv3\opencv\build\x64\vc14
Add %OPENCV_DIR%/bin to the PATH

Download python for openCV:
http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv
# install using pip for example:
pip install opencv_python-3.3.0-cp36-cp36m-win_amd64.whl   

# Run test python program:
import cv2
print(cv2.__version__)

# Install leptonica library
# Install gImageReader  (includes tesseract)
https://github.com/manisandro/gImageReader/releases/download/v3.2.1/gImageReader_3.2.1_qt5_x86_64_tesseract4.0.0.git2f10be5.exe
# Install python wrapper for tesseract
pip install tesserocr


# Installing pyFASST
sudo pip install cython
download zip from here: https://pypi.python.org/pypi/pyFASST
unzip to a directory
cd to it (cd pyFASST-0.9.3)
run "python setup.py build"
