First make sure GPU is running use `nvidia-370` driver.  
Make sure by running nvidia-smi. This should give you a nice view of the GPU stats if all went well.

On 16.04: Download cuda 8.0 https://developer.nvidia.com/cuda-downloads (runfile 16.04)
DONT install the driver - ONLY install cuda 

To install the code:

```
sudo apt-get install libusb-1.0-0-dev mercurial libgtest-dev cmake-qt-gui
```

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
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense/
./scripts/install_glfw3.sh
sudo apt-get install libglfw3-dev
//Video4Linux backend
sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && udevadm trigger
./scripts/install_dependencies-4.4.sh
sudo reboot
cd librealsense
./scripts/patch-uvcvideo-16.04.simple.sh
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
