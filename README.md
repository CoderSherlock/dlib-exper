# dlib C++ library Exper [![Travis Status](https://travis-ci.org/CoderSherlock/dlib-exper.svg?branch=master)](https://travis-ci.org/CoderSherlock/dlib-exper)

We are working on distributed ml framework and optimazation for embedded systems. Dlib is an awesome framework which implemented using c++. Our work are doing some experiments and recraft based on this library. Credit to [Davis](https://github.com/davisking)'s great work.

## dlib C++ library 
Dlib is a modern C++ toolkit containing machine learning algorithms and tools for creating complex software in C++ to solve real world problems. See [http://dlib.net](http://dlib.net) for the main project documentation and API reference.



### Compiling dlib C++ example programs

Go into the examples folder and type:

```bash
mkdir build; cd build; cmake .. -DUSE_AVX_INSTRUCTIONS=1 -DDLIB_USE_CUDA=1; cmake --build .
```


### dlib sponsors

This research is based in part upon work supported by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA) under contract number 2014-14071600010. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of ODNI, IARPA, or the U.S. Government.

