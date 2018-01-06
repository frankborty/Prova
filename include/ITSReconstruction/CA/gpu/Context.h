// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file Context.h
/// \brief
///

#ifndef TRAKINGITSU_INCLUDE_GPU_CONTEXT_H_
#define TRAKINGITSU_INCLUDE_GPU_CONTEXT_H_

#include <string>
#include <vector>
#include "ITSReconstruction/CA/Definitions.h"
#include "ITSReconstruction/CA/gpu/StructGPUPrimaryVertex.h"
#include "ITSReconstruction/CA/PrimaryVertexContext.h"

#if TRACKINGITSU_OCL_MODE
	#include <stdio.h>
	#include <sstream>
	#include <stdexcept>
	#include <iomanip>
	#include <iostream>
	#include <limits>
	#include <fstream>
	#include <vector>
	#include<CL/cl.hpp>
#endif

namespace o2
{
namespace ITS
{
namespace CA
{
namespace GPU
{

struct DeviceProperties final
{
    std::string name;
    long globalMemorySize;
    int warpSize;

#if TRACKINGITSU_CUDA_MODE
    int gpuProcessors;
    int cudaCores;
    long constantMemorySize;
    long sharedMemorySize;
    long maxClockRate;
    int busWidth;
    long l2CacheSize;
    long registersPerBlock;
    int maxThreadsPerBlock;
    int maxBlocksPerSM;
    dim3 maxThreadsDim;
    dim3 maxGridDim;

#elif TRACKINGITSU_OCL_MODE
    std::string vendor;
    std::size_t maxComputeUnits;
    std::size_t maxWorkGroupSize;
    std::size_t maxWorkItemDimension;
	dim3 maxWorkItemSize;
    cl::Context oclContext;
    cl::Device oclDevice;
    cl::CommandQueue oclQueue;

#endif
};

class Context final
{
  public:
    static Context& getInstance();

    Context(const Context&);
    Context& operator=(const Context&);

    const DeviceProperties& getDeviceProperties();
    const DeviceProperties& getDeviceProperties(const int);


  private:
    Context();
    ~Context() = default;

#if TRACKINGITSU_OCL_MODE
    int iCurrentDevice;
#endif
    int mDevicesNum;
    std::vector<DeviceProperties> mDeviceProperties;
};

}
}
}
}

#endif /* TRAKINGITSU_INCLUDE_GPU_CONTEXT_H_ */
