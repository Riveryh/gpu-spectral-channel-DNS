cmake_minimum_required(VERSION 3.17)

project(gpu-spectral-channel-DNS)
enable_language(CUDA)
# set(CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMakeModules;${CMAKE_MODULE_PATH}")

# -------------------------------  Denpendencies -------------------------------
find_package(MKL REQUIRED)
find_package(CUDAToolkit REQUIRED)
set(DEPENDENCIES ${MKL_LIBRARIES} CUDA::cudart_static CUDA::cufft CUDA::cublas)

# -------------------------------  Dealing with pthreads
# -------------------------------
if(WIN32)
  set(PTHREADS_PATH
      path/to/pthreads
      CACHE PATH "The path to a win32 version pthreads.")

  if(EXISTS ${PTHREADS_PATH}/include/pthread.h)
    list(APPEND CMAKE_PREFIX_PATH ${PTHREADS_PATH})
    find_package(PTHREADW32)
  else()
    message(
      FATAL_ERROR
        "Please specify PTHREADS_PATH as the directory of your win32 pthreads installation."
    )
  endif()

  if(NOT PTHREADW32_FOUND)
    message(FATAL_ERROR "PTHREADS_PATH invalid, please check again.")
  endif()
  set(DEPENDENCIES ${DEPENDENCIES} pthreadw32)
else()
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(Threads REQUIRED)
  set(DEPENDENCIES ${DEPENDENCIES} Threads::Threads)
endif()

# -------------------------------  Dealing with Eigen3
# -------------------------------
find_package(Eigen3 3.3 NO_MODULE)
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

# -------------------------------  Define source files
# -------------------------------
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
