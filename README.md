On 16.04: First make sure GPU is running use `nvidia-367` driver.

```
sudo apt-get purge nvidia-*
sudo apt-get autoremove
sudo reboot

sudo apt-get install gcc-4.9 g++-4.9
sudo ln -s  /usr/bin/gcc-4.9 /usr/bin/gcc -f
sudo ln -s  /usr/bin/g++-4.9 /usr/bin/g++ -f

sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-367
sudo reboot
```
(https://www.jayakumar.org/linux/gtx-1070-on-ubuntu-16-04-with-cuda-8-0-and-theano/)

Make sure by running nvidia-smi. This should give you a nice view of the GPU stats if all went well.

Download cuda 8.0 https://developer.nvidia.com/cuda-downloads (runfile 16.04)
DONT install the driver - ONLY install cuda

Make sure the binary and library path are setup correctly:
```
export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

To check nvcc is working fine:
```
nvcc -V
```

To install the code:

```
sudo apt-get install libusb-1.0-0-dev mercurial libgtest-dev cmake-qt-gui libglfw3-dev
```

Install librealsense (you might have to run `./scripts/patch-uvcvideo-16.04.simple.sh` twice to get the kernel patched properly)
```
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense/
sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && udevadm trigger
./scripts/patch-uvcvideo-16.04.simple.sh
sudo modprobe uvcvideo
make
sudo make install
```
Now verify that the camera works with the librealsense samples by running `./bin/cpp-tutorial-2-streams`. You should see depth, RGB, and IR streams. If not rerun the patch script.


From below you likely only need to compile this code (tdp) in your own
clone of the repo in your home since the other libraries should all
just be installed globally already.

```
cd /usr/src/gtest/
sudo su
mkdir build
cd build
cmake ..
make
ln libgtest* /usr/lib/
cd -
```


```
hg clone https://bitbucket.org/eigen/eigen
cd eigen
mkdir build
cd build
cmake ..
make
sudo make install
cd -
```

```
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
git checkout -b devel
git pull origin devel
sudo apt-get install libglew-dev
mkdir build
cd build
cmake ..
make -j
cd -
```

This code: git@github.mit.edu:jstraub/tdp.git
git fetch
git checkout lymp
```
mkdir build
cd build
cmake ..
make -j
```

To test the sensor, connect it to usb3.0 and run
cd
./tdp/build/experiements/simpleGui/simpleGui realsense:://

Calibu
------
Need to install libglog, libgflags(?), libCVars and libceres from scratch.

libglog
```
git clone https://github.com/google/glog.git
git checkout tags/v0.3.4
autoreconf -f -i
./configure
make
sudo make install
```

libCVars
```
git clone https://github.com/arpg/CVars.git
sudo apt-get install libtinyxml-dev libtinyxml2-dev
sudo ln -s /usr/lib/x86_64-linux-gnu/libtinyxml* /usr/lib/
mkdir build
cd build
cmake ..
make -j
sudo make install
```

libceres
```
git clone https://github.com/ceres-solver/ceres-solver.git
cd ceres-solver
cmake-gui ..
// select: BUILD_SHARED_LIBS
// select: CXX11
// select: MINILOG
make -j
sudo make install
```

Calibu needs to be built using [this branch](https://github.com/jstraub/Calibu/tree/fixesJstraub).

```
git clone https://github.com/jstraub/Calibu.git
git fetch
git checkout fixesJstraub
sudo apt-get install ffmpeg libopencv-dev libgtk-3-dev python-numpy python3-numpy libdc1394-22 libdc1394-22-dev libjpeg-dev libpng12-dev libtiff5-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libxine2-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libv4l-dev libtbb-dev qtbase5-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev x264 v4l-utils unzip
mkdir build
cd build
cmake-gui ..
// select: BUILD_CALIBGRID
make -j
sudo make install
```





