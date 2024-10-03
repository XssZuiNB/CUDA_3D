
## Requirments
### 1. GCC and G++

### 2. CMake 3.14 

### 3. CUDA 12.0
> CUDA 11.* has a big bug with GCC 11 or 12. The problem was fixed in CUDA version 12 or upper. 
> 
> Because of the version 12.0 of CUDA, the Nvidia driver version should be upper than **525.***! 

### 4. librealsense (SDK for realsense cameras by Intel)
**Notice: All the algorithms are not based on SDK. The usage of it is only to get the frames from realsense cameras**
> Unbuntu Linux should follow this guide to install the lib for realsense camera
https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md

### 5. Demo 
An easy dense slam implementation with color-ICP (https://openaccess.thecvf.com/content_ICCV_2017/papers/Park_Colored_Point_Cloud_ICCV_2017_paper.pdf) runs within 7ms per frame: https://youtu.be/5NcDEhguzGU.
