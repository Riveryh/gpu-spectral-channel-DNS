cmake_minimum_required(VERSION 3.5)

project(gpu-spectral-channel-DNS)
enable_language(CUDA)
# set(CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMakeModules;${CMAKE_MODULE_PATH}")

# -------------------------------  Denpendencies -------------------------------
find_package(MKL REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
set(DEPENDENCIES ${MKL_LIBRARIES} CUDA::cudart_static CUDA::cufft CUDA::cublas Threads::Threads)

# -------------------------------  Eigen3  -------------------------------
find_package(Eigen3 3.2 NO_MODULE)
if(TARGET Eigen3::Eigen)
  set(DEPENDENCIES ${DEPENDENCIES} Eigen3::Eigen)
else()
  set(EIGEN3_DIR
      "installation/to/eigen"
      CACHE PATH "The directory containing eigen headers.")
  if(IS_DIRECTORY ${EIGEN3_DIR}/Eigen)
    include_directories(${EIGEN3_DIR})
    message(STATUS "Using user-specified Eigen3 directory.")
  else()
    message(
      FATAL_ERROR
        "Please specify EIGEN3_DIR as the directory of your Eigen3 installation."
    )
  endif()
endif(TARGET Eigen3::Eigen)

# -------------------------------  Define sources  -------------------------------
file(
  GLOB_RECURSE headers
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  include/*.h)
file(
  GLOB_RECURSE cu_headers
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  include/*.cuh)
file(
  GLOB_RECURSE src
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  source/*.cpp)
file(
  GLOB_RECURSE cu_src
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  source/*.cu)

# -------------------------------  Define output -------------------------------
include_directories(${MKL_INCLUDE_DIR} include)
add_executable(gpu-spectral-channel-DNS ${src} ${cu_src} ${headers}
                                        ${cu_headers})

set_target_properties(gpu-spectral-channel-DNS PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(gpu-spectral-channel-DNS ${DEPENDENCIES})
