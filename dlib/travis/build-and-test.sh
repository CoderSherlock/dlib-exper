#!/usr/bin/env bash
# Exit if anything fails.
set -eux

# execute the contents of MATRIX_EVAL if it's set
if [[ -v MATRIX_EVAL ]]; then
    eval "${MATRIX_EVAL}"
fi

# build dlib and tests
if [ "$VARIANT" = "test" ]; then
  mkdir build
  cd build
  cmake ../dlib/test 
  cmake --build . --target dtest -- -j 2
  ./dtest --runall
fi

if [ "$VARIANT" = "dlib_all_source_cpp" ]; then
  mkdir build
  cd build
  cmake ../dlib/test 
  cmake --build . --target dlib_all_source_cpp -- -j 2
fi

if [ "$VARIANT" = "tools" ]; then
  mkdir build
  cd build
  cmake ../dlib/test/tools 
  cmake --build .  -- -j 2
fi

# The point of this test is just to make sure the cmake scripts work with the
# oldest version of cmake we are supposed to support.
if [ "$VARIANT" = "old-cmake" ]; then
  mkdir build
  cd build
  CMAKEDIR=../cmake

  $CMAKEDIR/2.8/bin/cmake ../dlib/test/tools 
  $CMAKEDIR/2.8/bin/cmake --build .  -- -j 2

  rm -rf *
  $CMAKEDIR/3.1/bin/cmake ../dlib/test/tools 
  $CMAKEDIR/3.1/bin/cmake --build .  -- -j 2

  rm -rf *
  $CMAKEDIR/3.5/bin/cmake ../dlib/test/tools 
  $CMAKEDIR/3.5/bin/cmake --build .  -- -j 2


  # just to make sure there isn't anything funny about building standalone dlib
  rm -rf *
  $CMAKEDIR/2.8/bin/cmake ../dlib 
  $CMAKEDIR/2.8/bin/cmake --build .  -- -j 2
fi

if [ "$VARIANT" = "examples" ]; then
  mkdir build
  cd build
  cmake ../examples 
  cmake --build . -- -j 1
fi

if [ "$VARIANT" = "python-api" ]; then
  python setup.py test --clean
  pip uninstall numpy -y
  python setup.py test --clean
fi

# Expected environment of distributed deep learning.
if [ "$VARIANT" = "ddl" ]; then
  mkdir build
  cd build
  cmake ../examples -DCMAKE_BUILD_TYPE=Release -DDLIB_USE_CUDA=0 -DDLIB_JPEG_SUPPORT=0 -DDLIB_LINK_WITH_SQLITE3=0 -DDLIB_USE_BLAS=0 -DDLIB_USE_LAPACK=0 -DDLIB_PNG_SUPPORT=0 -DDLIB_GIF_SUPPORT=0 -DDLIB_USE_FFTW=0 -DDLIB_USE_MKL_FFT=0 -DDLIB_NO_GUI_SUPPORT=1
  cmake --build . -- -j 10
fi
