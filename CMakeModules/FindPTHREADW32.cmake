# FindPTHREADW32
# --------
#
# Find Pthread for win32
#
# Find the native pthread for win32 includes and library This module defines
#
# ::
#
#   PTHREADW32_INCLUDE_DIR, where to find pthread.h, etc.
#   PTHREADW32_LIBRARY, where to find the pthread for win32 library.
#   PTHREADW32_FOUND, If false, do not try to use pthread for win32.
#   HAVE_PTHREAD, if true, pthread supported by compiler
#
#   IMPORTED TARGET: pthreadw32
#=============================================================================

if(NOT WIN32)
	return()
endif()
# 检查编译器是否支持pthread如果支持就返回, 
# POSIX版本的MinGW原生支持pthread,不需要额外的pthread for win32库
include(CheckLibraryExists)
CHECK_LIBRARY_EXISTS (pthread pthread_rwlock_init "" HAVE_PTHREAD)
if(HAVE_PTHREAD)
	message(STATUS "pthread supported")
	return()
endif()
# 查找pthread.h 头文件位置
find_path(PTHREADW32_INCLUDE_DIR pthread.h)
set(PTHREADW32_NAME pthread)
if(MSVC)
	set(PTHREADW32_NAME pthread.lib pthreadVC3)
elseif(MINGW)
	set(PTHREADW32_NAME libpthread.a pthreadGC2)	
endif()

# 查找库文件
find_library(PTHREADW32_LIBRARY NAMES ${PTHREADW32_NAME} pthread PATH_SUFFIXES lib )

# handle the QUIETLY and REQUIRED arguments and set PTHREADW32_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(PTHREADW32 DEFAULT_MSG PTHREADW32_LIBRARY PTHREADW32_INCLUDE_DIR)

mark_as_advanced(PTHREADW32_LIBRARY PTHREADW32_INCLUDE_DIR )
#message(STATUS PTHREADW32_INCLUDE_DIR=${PTHREADW32_INCLUDE_DIR})

if(PTHREADW32_FOUND) 
	if(MSVC)
		set(_dll_name pthreadVC3.dll)
	elseif(MINGW)
		set(_dll_name pthreadGC2.dll)	
	endif()
	find_file(PTHREADW32_DLL ${_dll_name} PATH_SUFFIXES bin)
	#message(STATUS PTHREADW32_DLL=${PTHREADW32_DLL})
	# 创建imported target
	add_library(pthreadw32 UNKNOWN IMPORTED)
	set_target_properties(pthreadw32 PROPERTIES
	  INTERFACE_INCLUDE_DIRECTORIES "${PTHREADW32_INCLUDE_DIR}"
	  IMPORTED_LINK_INTERFACE_LANGUAGES "C"
	  IMPORTED_LOCATION "${PTHREADW32_LIBRARY}"
	  )
	
	# 解决 Visual Studio 2015下编译struct timespec重定义问题
	if(MSVC)
		# 检查是否定义了 struct timespec
		include(CheckStructHasMember)
		CHECK_STRUCT_HAS_MEMBER("struct timespec" tv_sec time.h HAVE_STRUCT_TIMESPEC LANGUAGE C )  
		if(HAVE_STRUCT_TIMESPEC)
			set_target_properties(pthreadw32 PROPERTIES INTERFACE_COMPILE_DEFINITIONS HAVE_STRUCT_TIMESPEC )
		endif()
	endif()
endif()

unset(_arch)
