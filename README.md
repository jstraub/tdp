First make sure gpu is running use nvidia-340 driver.

Make sure by running nvidia-smi (if it complains the GPU is not
running)

Download cuda 6.5 https://developer.nvidia.com/cuda-downloads
(runfile 14.04)
DONT install the driver - ONLY install cuda 

To install the code:

```
sudo apt-get install lib-usb-1.0-0-dev mercurial libgtest-dev cmake-gt-gui
```

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
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense/
./scripts/install_glfw3.sh
make
sudo make install
cd ../
```

```
hg clone https://bitbucket.org/eigen/eigen
cd eigen
mkdir build
cd build
cmake ..
make
sudo make install
://developer.nvidia.com/cuda-downloads
cd -
```

```
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
git checkout -b devel
git pull origin devel
mkdir build
cd build
cmake ..
make -j
cd -
```

This code: git@github.mit.edu:jstraub/tdp.git
```
mkdir build
cd build 
cmake ..
make -j
```
