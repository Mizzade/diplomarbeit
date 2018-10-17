## Requirments
- docker
- nvidia-docker
- MATLAB mit Lizenz oder als Trial Version.
- wget
- tar
- mex
- g++ 6.3.x
- [Vlfeat](http://www.vlfeat.org/). Hier nutzen wir Version [0.9.21](http://www.vlfeat.org/download/vlfeat-0.9.21.tar.gz). Die entsprechende Version wird im Skript **download_vlfeat.sh** heruntergeladen. Manuell kann sie auf der obigen Seite heruntergeladen werden. Eine Anleitung dazu befindet sich [hier](http://www.vlfeat.org/install-matlab.html).
- [MatConvNet](http://www.vlfeat.org/matconvnet/) Hier nutzen wir Version [1.0-beta25](http://www.vlfeat.org/matconvnet/download/matconvnet-1.0-beta25.tar.gz). Die entsprechende Version wird im Skript **download_matconvnet.sh** heruntergeladen. Manuell kann sie auf der obigen Seite herunterladen werden. Eine Anleitung dazu befindet sich [hier](http://www.vlfeat.org/matconvnet/install/).
- Die Umgebungsvariable **MATLAB_HOME** muss gesetzt sein auf das */bin* Verzeichnis der Matlab Installation zeigen. Beispiel:
```
$ echo $MATLAB_HOME
~Workspace/MATLAB/R2018b/bin
```

### Installationsnotizen für VLFeat:
Auszug von [General instructions](http://www.vlfeat.org/compiling-unix.html):

To compile the library, it is usually sufficient to change to VLFeat root directory, denoted VLFEATROOT in the following, and type make:

    $ cd VLFEATROOT
    $ make
The make script attempts to automatically detect the host architecture and configure itself accordingly. If the architecture is not detected correctly, it can be specified manually. For instance

    $ make ARCH=glnx86

#### Compiling  MATLAB support
In order for MATLAB support to be compiled, the MATLAB mex script must be in the current path. If it is not, its location must be passed to make as follows. First, determine MATLAB's root directory by running a MATLAB session and issuing the matlabroot command. Let MATLABROOT denote the returned path (e.g. /Applications/MATLAB_R2009b.app/). The mex script is usually located in MALTABROOT/bin/mex. Then run the compilation with the command

$ make MEX=MATLABROOT/bin/mex

### Installationsnotizen für MatConvNet:
Auszug aus [Installing and compiling the library](http://www.vlfeat.org/matconvnet/install/):

Make sure that MATLAB is configured to use your compiler. In particular, before running vl_compilenn do not forget to setup mex (doing so nce is sufficient) as follows:

    $ mex -setup mex -setup C++

For Linux, make sure you have GCC 4.8 and LibJPEG are installed. To install LibJPEG in and Ubuntu/Debian-like distributions use: sudo apt-get install build-essential libjpeg-turbo8-dev For Fedora/Centos/RedHat-like distributions use instead: sudo yum install gcc gcc-c++ libjpeg-turbo-devel Older versions of GCC (e.g. 4.7) are not compatible with the C++ code in MatConvNet.
