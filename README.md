# gpu-spectral-channel-DNS

A CUDA spectral DNS code for wall bounded channel flow or couette flow

## Introduction

This code uses a traditional spectral method to solve NS equation in a channel.
Periodic boundary conditions are imposed on streamwise and spanwise direction.
While non-slip boundary condition is used in wall-normal direction.
Fourier expansions are used in spanwsie direction. Chebyshev polynomials are used in wall-normal direction.

Currently, the computation of nonlinear part is mainly conducted on GPU, while the linear part is computated on CPU.
And only ONE gpu will be used for acceleration.

## Environment Requirements

CUDA Runtime : CUDA 9.0 or higher

CUDA Library used: cufft

Intel C++ compiler with mkl support is required.

A cuda supported nVidia graph card is required to run the code.

## Build & Run

You will need CMake 3.17.

It has been tested on {Visual Studio 2019 / Clang-8 / gcc5.4} + MKL 2020 + CUDA 10.2

For example, on Linux:

``` bash
cd gpu-spectral-channel-DNS
mkdir build
cd build

# change your build settings in build.sh
sh ../build.sh
make -j8

# you can modify parameters.txt for your need
cp ../parameters.txt ./
./gpu-spectral-channel-DNS
```

## Usage or Citation

You may contact riveryh[at]foxmail.com
