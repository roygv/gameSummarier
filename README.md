# gameSummarier
Automate the creation of sports game summaries. W210 capstone project MIDS@UCBerkeley

# Installation steps on Windows 10
Download and install Anaconda 64bit 3.5.1 from here: https://www.anaconda.com/download/
Optional (if you have an NVidia GPU card): Download and install CUDA (http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
    install Base CUDA installer
    install patch 2 (Released Jun 26, 2017)
    check and upgrade you display driver (CUDA may have downgraded your driver) http://www.nvidia.com/Download/Scan.aspx
If you have multiple Python environments switch to the newly installed Anaconda:
    create a new environment with python 3.6 and anaconda: 
        conda create -n gameSummarizer python=3.6 anaconda
    list environments:
        conda info --envs
    activate environment:
        activate gameSummarizer
        python -V   #(to show the version)    
Run "conda install mingw libpython"
Install Theano:
    pip install git+https://github.com/Theano/Theano.git
Install cuDNN v6.0 for CUDA 8.0 from here https://developer.nvidia.com/rdp/cudnn-download#a-collapse7-8



