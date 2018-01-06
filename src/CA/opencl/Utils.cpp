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
/// \file CAGPUtils.cu
/// \brief
///

#include "ITSReconstruction/CA/gpu/Utils.h"

#include <sstream>
#include <stdexcept>
#include <string>
#include <iostream>
#include<fstream>

#define __CL_ENABLE_EXCEPTIONS //abilita le eccezioni
#include <CL/cl.hpp>
#include "ITSReconstruction/CA/gpu/Context.h"
#include <unistd.h>
#include "myThresholds.h"


namespace {

//void checkCUDAError(const cudaError_t error, const char *file, const int line)
//{
//  if (error != cudaSuccess) {
//
//    std::ostringstream errorString { };
//
//    errorString << file << ":" << line << " CUDA API returned error [" << cudaGetErrorString(error) << "] (code "
//        << error << ")" << std::endl;
//
//    throw std::runtime_error { errorString.str() };
//  }
//}

int roundUp(const int numToRound, const int multiple)
{
	if (multiple == 0) {
		return numToRound;
	}
	int remainder { numToRound % multiple };
	if (remainder == 0) {
		return numToRound;
	}
	return numToRound + multiple - remainder;
}

int findNearestDivisor(const int numToRound, const int divisor)
{
	if (numToRound > divisor) {
		return divisor;
	}
	int result = numToRound;

	while (divisor % result != 0) {
		++result;
	}
	return result;
}

}

namespace o2
{
namespace ITS
{
namespace CA
{
namespace GPU
{


int Utils::getOclWorkGroup(int workItemsNumber,int warpSize){
	int maxWorkGroupSize=128;
	while(true){
		if(maxWorkGroupSize<=0)
			return maxWorkGroupSize;
		if(/*maxWorkGroupSize%warpSize==0 && */workItemsNumber%maxWorkGroupSize==0)
			return maxWorkGroupSize;
		maxWorkGroupSize--;
	}
}

cl::Kernel Utils::CreateKernelFromFile(cl::Context oclContext, cl::Device oclDevice, const char* fileName,const char* kernelName){
	//std::cout << "CreateKernelFromFile: "<<fileName <<"... ";
	cl::Kernel kernel;
	std::ifstream kernelFile(fileName, std::ios::in);
	if (!kernelFile.is_open())
	{
		std::cerr << "\nFailed to open file for reading: " << fileName << std::endl;
		return cl::Kernel();
	}

	std::ostringstream oss;
	oss << kernelFile.rdbuf();

	std::string srcStdStr = oss.str();
	//std::cerr<<srcStr<< std::endl;

	cl::Program::Sources sources;
	sources.push_back({srcStdStr.c_str(),srcStdStr.length()});

	cl::Program program(oclContext,sources);
	try{
		std::vector<cl::Device> oclDeviceList;
		oclDeviceList.push_back(oclDevice);
		char buildOption[100];
		sprintf(buildOption,"-cl-std=CL2.0 -cl-unsafe-math-optimizations -cl-fast-relaxed-math");
		//std::cout<<"kernel option "<<buildOption<<std::endl;
		program.build(oclDeviceList,buildOption);
		kernel=cl::Kernel(program,kernelName);
	}
	catch(const cl::Error &err){

		std::string errString=Utils::OCLErr_code(err.err());
		std::cout<< errString << std::endl;

		std::cerr
				<< "OpenCL compilation error" << std::endl
				<< program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(oclDevice)
				<< std::endl;

		throw std::runtime_error { errString };
	}
	return kernel;
}




char* Utils::OCLErr_code (int err_in){
	switch (err_in) {

	case CL_SUCCESS :
		return (char*)" CL_SUCCESS ";
	case CL_DEVICE_NOT_FOUND :
		return (char*)" CL_DEVICE_NOT_FOUND ";
	case CL_DEVICE_NOT_AVAILABLE :
		return (char*)" CL_DEVICE_NOT_AVAILABLE ";
	case CL_COMPILER_NOT_AVAILABLE :
		return (char*)" CL_COMPILER_NOT_AVAILABLE ";
	case CL_MEM_OBJECT_ALLOCATION_FAILURE :
		return (char*)" CL_MEM_OBJECT_ALLOCATION_FAILURE ";
	case CL_OUT_OF_RESOURCES :
		return (char*)" CL_OUT_OF_RESOURCES ";
	case CL_OUT_OF_HOST_MEMORY :
		return (char*)" CL_OUT_OF_HOST_MEMORY ";
	case CL_PROFILING_INFO_NOT_AVAILABLE :
		return (char*)" CL_PROFILING_INFO_NOT_AVAILABLE ";
	case CL_MEM_COPY_OVERLAP :
		return (char*)" CL_MEM_COPY_OVERLAP ";
	case CL_IMAGE_FORMAT_MISMATCH :
		return (char*)" CL_IMAGE_FORMAT_MISMATCH ";
	case CL_IMAGE_FORMAT_NOT_SUPPORTED :
		return (char*)" CL_IMAGE_FORMAT_NOT_SUPPORTED ";
	case CL_BUILD_PROGRAM_FAILURE :
		return (char*)" CL_BUILD_PROGRAM_FAILURE ";
	case CL_MAP_FAILURE :
		return (char*)" CL_MAP_FAILURE ";
	case CL_MISALIGNED_SUB_BUFFER_OFFSET :
		return (char*)" CL_MISALIGNED_SUB_BUFFER_OFFSET ";
	case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST :
		return (char*)" CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST ";
	case CL_INVALID_VALUE :
		return (char*)" CL_INVALID_VALUE ";
	case CL_INVALID_DEVICE_TYPE :
		return (char*)" CL_INVALID_DEVICE_TYPE ";
	case CL_INVALID_PLATFORM :
		return (char*)" CL_INVALID_PLATFORM ";
	case CL_INVALID_DEVICE :
		return (char*)" CL_INVALID_DEVICE ";
	case CL_INVALID_CONTEXT :
		return (char*)" CL_INVALID_CONTEXT ";
	case CL_INVALID_QUEUE_PROPERTIES :
		return (char*)" CL_INVALID_QUEUE_PROPERTIES ";
	case CL_INVALID_COMMAND_QUEUE :
		return (char*)" CL_INVALID_COMMAND_QUEUE ";
	case CL_INVALID_HOST_PTR :
		return (char*)" CL_INVALID_HOST_PTR ";
	case CL_INVALID_MEM_OBJECT :
		return (char*)" CL_INVALID_MEM_OBJECT ";
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR :
		return (char*)" CL_INVALID_IMAGE_FORMAT_DESCRIPTOR ";
	case CL_INVALID_IMAGE_SIZE :
		return (char*)" CL_INVALID_IMAGE_SIZE ";
	case CL_INVALID_SAMPLER :
		return (char*)" CL_INVALID_SAMPLER ";
	case CL_INVALID_BINARY :
		return (char*)" CL_INVALID_BINARY ";
	case CL_INVALID_BUILD_OPTIONS :
		return (char*)" CL_INVALID_BUILD_OPTIONS ";
	case CL_INVALID_PROGRAM :
		return (char*)" CL_INVALID_PROGRAM ";
	case CL_INVALID_PROGRAM_EXECUTABLE :
		return (char*)" CL_INVALID_PROGRAM_EXECUTABLE ";
	case CL_INVALID_KERNEL_NAME :
		return (char*)" CL_INVALID_KERNEL_NAME ";
	case CL_INVALID_KERNEL_DEFINITION :
		return (char*)" CL_INVALID_KERNEL_DEFINITION ";
	case CL_INVALID_KERNEL :
		return (char*)" CL_INVALID_KERNEL ";
	case CL_INVALID_ARG_INDEX :
		return (char*)" CL_INVALID_ARG_INDEX ";
	case CL_INVALID_ARG_VALUE :
		return (char*)" CL_INVALID_ARG_VALUE ";
	case CL_INVALID_ARG_SIZE :
		return (char*)" CL_INVALID_ARG_SIZE ";
	case CL_INVALID_KERNEL_ARGS :
		return (char*)" CL_INVALID_KERNEL_ARGS ";
	case CL_INVALID_WORK_DIMENSION :
		return (char*)" CL_INVALID_WORK_DIMENSION ";
	case CL_INVALID_WORK_GROUP_SIZE :
		return (char*)" CL_INVALID_WORK_GROUP_SIZE ";
	case CL_INVALID_WORK_ITEM_SIZE :
		return (char*)" CL_INVALID_WORK_ITEM_SIZE ";
	case CL_INVALID_GLOBAL_OFFSET :
		return (char*)" CL_INVALID_GLOBAL_OFFSET ";
	case CL_INVALID_EVENT_WAIT_LIST :
		return (char*)" CL_INVALID_EVENT_WAIT_LIST ";
	case CL_INVALID_EVENT :
		return (char*)" CL_INVALID_EVENT ";
	case CL_INVALID_OPERATION :
		return (char*)" CL_INVALID_OPERATION ";
	case CL_INVALID_GL_OBJECT :
		return (char*)" CL_INVALID_GL_OBJECT ";
	case CL_INVALID_BUFFER_SIZE :
		return (char*)" CL_INVALID_BUFFER_SIZE ";
	case CL_INVALID_MIP_LEVEL :
		return (char*)" CL_INVALID_MIP_LEVEL ";
	case CL_INVALID_GLOBAL_WORK_SIZE :
		return (char*)" CL_INVALID_GLOBAL_WORK_SIZE ";
	case CL_INVALID_PROPERTY :
		return (char*)" CL_INVALID_PROPERTY ";
	default:
		return (char*)"UNKNOWN ERROR";

	}
}


dim3 Utils::Host::getBlockSize(const int colsNum)
{
	return getBlockSize(colsNum, 1);
}

dim3 Utils::Host::getBlockSize(const int colsNum, const int rowsNum)
{
  const DeviceProperties& deviceProperties = Context::getInstance().getDeviceProperties();
  //il terzo parameto indica il max numero di threa per blocco (cuda) quindi in openCl è il max numero di work item per work group, valore che è salvato nelle properties
  //return getBlockSize(colsNum, rowsNum, deviceProperties.cudaCores / deviceProperties.maxBlocksPerSM);
  return getBlockSize(colsNum, rowsNum, deviceProperties.maxWorkGroupSize);
}

dim3 Utils::Host::getBlockSize(const int colsNum, const int rowsNum, const int maxThreadsPerBlock)
{
	const DeviceProperties& deviceProperties = Context::getInstance().getDeviceProperties();
	//la funzione min non è riconosciuta quindi ho scritto std::min e aggiunto il cast
	int xThreads = std::min(colsNum, (int)deviceProperties.maxWorkItemSize.x);
	int yThreads = std::min(rowsNum, (int)deviceProperties.maxWorkItemSize.y);
	const int totalThreads = roundUp(std::min(xThreads * yThreads,(int)maxThreadsPerBlock),deviceProperties.warpSize);

	if (xThreads > yThreads) {
		xThreads = findNearestDivisor(xThreads, totalThreads);
		yThreads = totalThreads / xThreads;
	} else {
		yThreads = findNearestDivisor(yThreads, totalThreads);
		xThreads = totalThreads / yThreads;
	}
	return dim3 { static_cast<unsigned int>(xThreads), static_cast<unsigned int>(yThreads) };
}

dim3 Utils::Host::getBlocksGrid(const dim3 &threadsPerBlock, const int rowsNum)
{
	return getBlocksGrid(threadsPerBlock, rowsNum, 1);
}

dim3 Utils::Host::getBlocksGrid(const dim3 &threadsPerBlock, const int rowsNum, const int colsNum)
{
	return dim3 { 1 + (rowsNum - 1) / threadsPerBlock.x, 1 + (colsNum - 1) / threadsPerBlock.y };
}

void Utils::Host::gpuMalloc(void **p, const int size)
{
	//tmp:alloco sull'host e dopo sposto sul device
	//std::cout<<"gpuMalloc"<< std::endl;
	*p=malloc(size);
}

void Utils::Host::gpuFree(void *p)
{
	//tmp:free sull'host e dopo sposto sul device
	free(p);
}

void Utils::Host::gpuMemset(void *p, int value, int size)
{
//  checkCUDAError(cudaMemset(p, value, size), __FILE__, __LINE__);
	p=memset(p,value,size);
}

void Utils::Host::gpuMemcpyHostToDevice(void *dst, const void *src, int size)
{
	/*cl::Context oclContext=Context::getInstance().getDeviceProperties().oclContext;
	cl::Buffer buf(oclContext,CL_MEM_READ_WRITE,size);
	cl::CommandQueue Q= Context::getInstance().getDeviceProperties().oclQueue;
	dst = (void*)Q.enqueueMapBuffer(buf, CL_TRUE, CL_MAP_WRITE, 0, size);
	*/
	//std::cout<<"gpuMemcpyHostToDevice"<< std::endl;
	memcpy(dst,src,size);
}

void Utils::Host::gpuMemcpyHostToDeviceAsync(void *dst, const void *src, int size, Stream &stream)
{
//  checkCUDAError(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream.get()), __FILE__, __LINE__);
	memcpy(dst,src,size);
}

void Utils::Host::gpuMemcpyDeviceToHost(void *dst, const void *src, int size)
{
//  checkCUDAError(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
	memcpy(dst,src,size);
}

void Utils::Host::gpuStartProfiler()
{
//  checkCUDAError(cudaProfilerStart(), __FILE__, __LINE__);
}

void Utils::Host::gpuStopProfiler()
{
//  checkCUDAError(cudaProfilerStop(), __FILE__, __LINE__);
}

GPU_DEVICE int Utils::Device::getLaneIndex()
{
//  uint32_t laneIndex;
//  asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneIndex));
//  return static_cast<int>(laneIndex);
	return 1;
}

GPU_DEVICE int Utils::Device::shareToWarp(const int value, const int laneIndex)
{
//  return __shfl(value, laneIndex);
	return 1;
}

GPU_DEVICE int Utils::Device::gpuAtomicAdd(int *p, const int incrementSize)
{
//  return atomicAdd(p, incrementSize);
	return 1;
}



}
}
}
}
