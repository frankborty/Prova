set(MODULE src)

set(SRCS
  CA/Cell.cxx
  CA/Cluster.cxx
  CA/Event.cxx
  CA/IOUtils.cxx
  CA/Label.cxx
  CA/Layer.cxx
  CA/PrimaryVertexContext.cxx
  CA/Road.cxx
  CA/Tracker.cxx
  CA/TrackingUtils.cxx
  CA/Tracklet.cxx
)

include_directories(${TRACKING-ITSU_SOURCE_DIR}/include)

if(TRACKINGITSU_TARGET_DEVICE STREQUAL GPU_CUDA)
	find_package(CUDA QUIET REQUIRED)
	include(FindCUDA)

	set(CUDA_ARCH -gencode arch=compute_52,code=sm_52 CACHE STRING "CUDA target architecture.")
	set(CUDA_PTXAS_OPTIONS -dlcm=ca)
	#set(CUDA_PTXAS_OPTIONS ${CUDA_PTXAS_OPTIONS},-v)
	set(CUDA_HOST_COMPILER_FLAGS -fPIC CACHE STRING "CUDA -Xcompiler flags.")

	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -Xcompiler ${CUDA_HOST_COMPILER_FLAGS}; ${CUDA_ARCH}; -std=c++11; --ptxas-options=${CUDA_PTXAS_OPTIONS};)

	# Cannot use CUDA_NVCC_FLAGS_<CONFIG> due to issue 16411
	if(CMAKE_BUILD_TYPE STREQUAL "Debug")
	    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -G; -g; -O2; -DDEBUG;)
	endif(CMAKE_BUILD_TYPE STREQUAL "Debug")

	if(CMAKE_BUILD_TYPE STREQUAL "Release")
	    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3; --use_fast_math;)
	endif(CMAKE_BUILD_TYPE STREQUAL "Release")

	if(CMAKE_BUILD_TYPE STREQUAL "Profile")
	    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -lineinfo)
	endif(CMAKE_BUILD_TYPE STREQUAL "Profile")

	set(CUDA_SEPARABLE_COMPILATION ON)
	set(CUDA_PROPAGATE_HOST_FLAGS OFF)
	#set(CUDA_VERBOSE_BUILD ON)

	# -lcudadevrt option
	#find_library_local_first(CUDA_CUDADEVRT_LIBRARY cudadevrt "\"cudadevrt\" library")
	#set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUDADEVRT_LIBRARY})

	message(STATUS "Compiling with NVCC flags: ${CUDA_NVCC_FLAGS}")

	include_directories(
		#FIXME with aliBuild dependencies
	    ${TRACKING-ITSU_SOURCE_DIR}/include/ITSReconstruction/CA/external/cub
	)

	set(CUDA_SRCS
	    CA/gpu/cuda/Context.cu
	    CA/gpu/cuda/PrimaryVertexContext.cu
	    CA/gpu/cuda/Stream.cu
	    CA/gpu/cuda/Utils.cu
	    CA/gpu/cuda/Tracker.cu
	)

	foreach(HOST_SRC_FILE ${SRCS})

	    get_filename_component(HOST_SRC_FILENAME ${HOST_SRC_FILE} NAME_WE)

	    add_custom_command(
	        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${HOST_SRC_FILENAME}.cu
	        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${HOST_SRC_FILE} ${CMAKE_CURRENT_BINARY_DIR}/${HOST_SRC_FILENAME}.cu
	    )

	    set(CUDA_SRCS ${CUDA_SRCS} ${CMAKE_CURRENT_BINARY_DIR}/${HOST_SRC_FILENAME}.cu)

	endforeach(HOST_SRC_FILE)

	cuda_add_library(${MODULE} ${CUDA_SRCS})
else(TRACKINGITSU_TARGET_DEVICE STREQUAL GPU_CUDA)

if(TRACKINGITSU_TARGET_DEVICE STREQUAL OPEN_CL)
  LIST(APPEND SRCS
    CA/opencl/Context.cpp
    CA/opencl/PrimaryVertexContext.cpp
    CA/opencl/Stream.cpp
    CA/opencl/Utils.cpp
    CA/opencl/Tracker.cpp
  )

configure_file(CA/opencl/kernel/computeLayerTracklets.cl ${CMAKE_CURRENT_BINARY_DIR}/kernel/computeLayerTracklets.cl COPYONLY)
configure_file(CA/opencl/kernel/SortKernels.cl ${CMAKE_CURRENT_BINARY_DIR}/kernel/SortKernels.cl COPYONLY)
 include_directories( ${OPENCL_INCLUDE_DIRS} ${TRACKING-ITSU_SOURCE_DIR}/src/CA/opencl/kernel)

endif(TRACKINGITSU_TARGET_DEVICE STREQUAL OPEN_CL)

	add_library(${MODULE} ${SRCS})
endif(TRACKINGITSU_TARGET_DEVICE STREQUAL GPU_CUDA)
