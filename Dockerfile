FROM tensorflow/tensorflow:2.9.1-gpu
WORKDIR /root/Share/uw-madison
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get autoclean; apt-get update --allow-insecure-repositories; apt-get install ffmpeg libsm6 libxext6 git gfortran libopenblas-dev liblapack-dev -y

RUN python3 -m pip install --upgrade pip

RUN pip3 install \
    albumentations==1.1.0 \
    efficientnet==1.0.0 \
    image-classifiers==1.0.0 \
    keras==2.8.0 \
    matplotlib==3.3.4 \
    monai==0.8.1 \
    nibabel==3.1.1 \
    nilearn==0.7.0 \
    nltk==3.4.5 \
    numpy==1.19.5 \
    opencv-python==4.5.5.64 \
    pandas==1.1.5 \
    scipy==1.3.1 \
    scikit-learn \
    segmentation-models==1.0.1 \
    SimpleITK==2.1.1 \
    tqdm==4.62.3 \
    vtk==9.0.1 \
    jupyter \
    ipywidgets \
    tensorflow-addons

RUN pip3 install -U numpy

RUN python3 -m pip install git+https://github.com/huggingface/transformers

RUN apt-get install git-lfs -y


