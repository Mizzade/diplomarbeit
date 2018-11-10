
### Befehle
Löscht den Container nach Beendigung.
Macht die Ordner data und outputs sichtbar im Container:

> docker run -it --rm --name TILDE --mount type=bind,source="/home/mizzade/Workspace/diplom/code/data",target="/home/mizzade/Workspace/diplom/code/data" --mount type=bind,source="/home/mizzade/Workspace/diplom/code/outputs",target="/home/mizzade/Workspace/diplom/code/outputs" d16ae1750e27 /bin/bash

#### Upgrade system
apt-get -y update
apt-get -y upgrade

#### Install essentials
apt-get install -y build-essential libgtk2.0-dev libjpeg-dev libtiff5-dev libjasper-dev libopenexr-dev cmake python-dev python-numpy python-tk libtbb-dev libeigen3-dev yasm libfaac-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev libqt4-dev libqt4-opengl-dev sphinx-common texlive-latex-extra libv4l-dev libdc1394-22-dev libavcodec-dev libavformat-dev libswscale-dev default-jdk ant libvtk5-qt4-dev eog

apt-get install -y git
apt-get install -y cmake

#### Folder structure
mkdir -p /home/tilde
cd home/tilde

git clone https://github.com/opencv/opencv.git
%git clone https://github.com/opencv/opencv_contrib.git
cd opencv


##### Set to Release 2.4.9
git reset --hard df8e282

mkdir build
cd build

Article for these commands: https://www.samontab.com/web/2014/06/installing-opencv-2-4-9-in-ubuntu-14-04-lts/

#### Generate makefile
cmake -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF -D BUILD_EXAMPLES=OFF -D WITH_QT=ON -D WITH_OPENGL=ON -D WITH_VTK=ON -D CUDA_GENERATION=Auto -D WITH_CUDA=OFF ..

### Generate and install OpenCV 2.4.9
make
sudo make install

cd ../..
git clone https://github.com/kmyid/TILDE
cd TILDE/c++
mkdir build
cd build
cmake ..
make

#### Test demo
cd Demo


#### Create from Dockerfile
alias nvidia-docker=nocker
nocker build -t tilde_test_app .

xhost +
nocker run --net=host --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" tilde_test_app

nocker run -it --name tilde_opencv32 --net=host --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" tilde_opencv-3.2 /bin/bash


NEU
nocker run -it \
  --name tilde_test_02 \
  --net=host \
  --env="DISPLAY" \
  --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
  --mount type=bind,source="/home/mizzade/Workspace/programming/cpp/read_filenames",target="/home/tilde/scripts" \
  tilde_opencv-2.4.9_python-3.6.6 \
  /bin/bash

# compile cpp file with static tilde lib
% in OpenCV 3.x muss -lopencv_imgcodecs anstelle von -lopencv_highgui eingetragen werden:
$ g++ -std=c++11 -o use_tilde use_tilde.cpp /home/tilde/TILDE/c++/build/Lib/libTILDE_static.a -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_features2d

# how to call use_tilde
./use_tilde --imageDir /home/tilde/TILDE/data --outputDir output --fileName testImage.png --filterPath /home/tilde/TILDE/c++/Lib/filters --filterName Mexico.txt

#### Copy compiled files from docker to host
nocker cp tilde_test_01:/home/tilde/TILDE/c++/libTILDE.so .

? TODO
apt-get install -y unetbootin
QT_X11_NO_MITSHM=1 unetbootin
