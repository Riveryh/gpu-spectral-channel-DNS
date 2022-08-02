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

A cuda supported NVIDIA graphic card is required to run the code.

## Build & Run

You will need CMake 3.17. and Eigen

Tested on {Visual Studio 2019 / Clang-8 / gcc5.4 / icc} + CUDA 10.2 or higher

On Linux, we recommend icc and clang-8 over gcc according to issues related to openmp.

For example, on Linux:

``` bash
mkdir build
cd build
cmake ..
make -j
./gpu-spectral-channel-DNS

# you can modify parameters.txt for your need
vim ../input/parameters.txt
```

## Usage or Citation

You may contact riveryh[at]foxmail.com

## Numerical method 
The method is breifly introduced in this papers,
https://link.springer.com/article/10.1007/s11433-018-9310-4 .
For time integration, it's basically a second-order Adams-Bashforth scheme for the non-linear term and a second-order Crank-Nicolson scheme fothe linear time. The spatial numerical scheme uses Fourier bases for spanwise and streamwsie along with periodical boundary conditions. For wall-normal direction, the chebyshev function is used for non-slip boundary condtion. The dialiasing method is 3/2 padding.
