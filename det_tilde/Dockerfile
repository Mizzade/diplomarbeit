FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04
# Main Features:
# OpenCV 3.4.1
# Python 3.6.6
# Update package manager.
RUN apt-get -y update
RUN apt-get -y upgrade

# Install essential packages.
RUN apt-get install -y build-essential libgtk2.0-dev libjpeg-dev libtiff5-dev libjasper-dev libopenexr-dev cmake python-dev python-numpy python-tk libtbb-dev libeigen3-dev yasm libfaac-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev libqt4-dev libqt4-opengl-dev sphinx-common texlive-latex-extra libv4l-dev libdc1394-22-dev libavcodec-dev libavformat-dev libswscale-dev default-jdk ant libvtk5-qt4-dev git cmake eog vim sqlite3 bzip2 python3 python3-pip
RUN pip3 install --upgrade pip
RUN apt-get install -y mlocate
RUN updatedb

# Install pyenv and python versions.
RUN git clone git://github.com/yyuu/pyenv.git .pyenv
RUN git clone https://github.com/yyuu/pyenv-virtualenv.git ~/.pyenv/plugins/pyenv-virtualenv
ENV HOME /
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN pyenv install 3.6.6
RUN pyenv global 3.6.6
# Upgrade Pip3
RUN pip3 install --upgrade pip
# Install tqdm
RUN pip3 install tqdm==4.28.1
# install opencv
RUN pip3 install opencv-python==3.4.2.17
# install numpy
RUN pip3 install numpy==1.15.1

# Create dir structure.
RUN ["mkdir", "-p", "/home/tilde"]
WORKDIR /home/tilde

# Download latest OpenCV, revert it to Version X and install it.
# 6ffc487 for v3.4.1
# df8e282 for v.2.4.9
RUN ["git",  "clone", "https://github.com/opencv/opencv.git"]
WORKDIR /home/tilde/opencv
RUN ["git", "reset", "--hard", "6ffc487"]
RUN ["mkdir", "build"]
WORKDIR /home/tilde/opencv/build
RUN cmake -D OPENCV_VERSION=3 -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF -D BUILD_EXAMPLES=OFF -D WITH_QT=ON -D WITH_OPENGL=ON -D WITH_VTK=ON -D CUDA_GENERATION=Auto -D WITH_CUDA=OFF ..
RUN make
RUN make install

# Download and install TILDE detector.
WORKDIR /home/tilde
RUN ["git", "clone", "https://github.com/kmyid/TILDE"]
WORKDIR /home/tilde/TILDE/c++
RUN ["mkdir", "build"]
WORKDIR /home/tilde/TILDE/c++/build
RUN ["cmake", ".."]
RUN ["make"]

# Configure opencv correctly to be found by g++
WORKDIR /etc/ld.so.conf.d
RUN touch opencv.conf
RUN echo /usr/local/lib >> opencv.conf
RUN ldconfig -v

WORKDIR /home/tilde/TILDE/c++/build
RUN mkdir output
# Kopiere use_tilde.cpp in das Image und kompiliere es.
COPY det_tilde/use_tilde.cpp /home/tilde/TILDE/c++/build/
RUN  g++ -std=c++11 -o use_tilde use_tilde.cpp /home/tilde/TILDE/c++/build/Lib/libTILDE_static.a -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_features2d

# Kopiere "Hello World" test shell Skript und führe es aus.
# TODO: Change with evaluation script to automatically start evaluation.
COPY det_tilde/hello_world.sh /home/tilde/TILDE/c++/build/
RUN chmod +x hello_world.sh

# Copy io_utils.py
COPY extern/io_utils.py /home/tilde/TILDE/c++/build/
COPY det_tilde/use_tilde.py /home/tilde/TILDE/c++/build/
RUN chmod +x use_tilde.py

CMD ["./hello_world.sh"]

