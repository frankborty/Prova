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
/// \file Context.cu
/// \brief
///


#include "ITSReconstruction/CA/gpu/Context.h"
#include "ITSReconstruction/CA/gpu/Utils.h"


//#define __CL_ENABLE_EXCEPTIONS //abilita le eccezioni


#define AMD_WAVEFRONT 		0x4043
#define NVIDIA_WAVEFRONT 	0x4003

//#include <cuda_runtime.h>

namespace {




//inline int getMaxThreadsPerSM(const int major, const int minor)
//{
//  return 8;
//}

}

namespace o2
{
namespace ITS
{
namespace CA
{
namespace GPU
{

Context::Context()
{
	std::vector<cl::Platform> platformList;
	std::vector<cl::Device> deviceList;
	std::vector<std::size_t> sizeDim;
	std::string info;
	std::size_t iPlatformList;
	std::size_t iTotalDevice=0;

	try{

		// Get the list of platform
		cl::Platform::get(&platformList);
		iPlatformList=platformList.size();
		// Pick first platform

		//std::cout << "There are " << iPlatformList << " platform" << std::endl;
		//std::cout << std::endl;
		for(int iPlatForm=0;iPlatForm<(int)iPlatformList;iPlatForm++){
			//std::cout << "Platform #" << iPlatForm+1 << std::endl;
			cl::Context context;
			try{
				cl_context_properties cprops[] = {
				CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList[iPlatForm])(), 0};
				context=cl::Context(CL_DEVICE_TYPE_ALL, cprops);
			}
			catch(const cl::Error &err){
					std::string errString=Utils::OCLErr_code(err.err());
					std::cout<<"No device found for platform #"<<iPlatForm<< std::endl;
					continue;
			}


			//print platform information
			platformList[iPlatForm].getInfo(CL_PLATFORM_NAME,&info);
			//std::cout << "Name:" 	<< info << std::endl;
			platformList[iPlatForm].getInfo(CL_PLATFORM_VENDOR,&info);
			//std::cout << "Vendor:"	<< info << std::endl;
			platformList[iPlatForm].getInfo(CL_PLATFORM_VERSION,&info);
			//std::cout << "Version: "<< info << std::endl;


			// Get devices associated with the first platform
			platformList[iPlatForm].getDevices(CL_DEVICE_TYPE_ALL,&deviceList);
			mDevicesNum=deviceList.size();
			mDeviceProperties.resize(iTotalDevice+mDevicesNum, DeviceProperties { });

			//std::cout << "There are " << mDevicesNum << " devices" << std::endl;

			for(int iDevice=0;iDevice<mDevicesNum;iDevice++){

				std::string name;
				deviceList[iDevice].getInfo(CL_DEVICE_NAME,&(mDeviceProperties[iTotalDevice].name));
				//std::acout << "	>> Device: " << mDeviceProperties[iTotalDevice].name << std::endl;

				//compute number of compute units (cores)
				deviceList[iDevice].getInfo(CL_DEVICE_MAX_COMPUTE_UNITS,&(mDeviceProperties[iTotalDevice].maxComputeUnits));
				//std::cout << "		Compute units: " << mDeviceProperties[iTotalDevice].maxComputeUnits << std::endl;

				//compute max device alloc size
				deviceList[iDevice].getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE,&(mDeviceProperties[iTotalDevice].globalMemorySize));
				//std::cout << "		Device Max Device Alloc Size: " << mDeviceProperties[iTotalDevice].globalMemorySize << std::endl;

				//compute device global memory size
				deviceList[iDevice].getInfo(CL_DEVICE_GLOBAL_MEM_SIZE,&(mDeviceProperties[iTotalDevice].globalMemorySize));
				//std::cout << "		Device Global Memory: " << mDeviceProperties[iTotalDevice].globalMemorySize << std::endl;

				//compute the max number of work-item in a work group executing a kernel (refer to clEnqueueNDRangeKernel)
				deviceList[iDevice].getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE,&(mDeviceProperties[iTotalDevice].maxWorkGroupSize));
				//std::cout << "		Max work-group size: " << mDeviceProperties[iTotalDevice].maxWorkGroupSize << std::endl;

				//compute the max work-item dimension
				deviceList[iDevice].getInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,&(mDeviceProperties[iTotalDevice].maxWorkItemDimension));
				//std::cout << "		Max work-item dimension: " << mDeviceProperties[iTotalDevice].maxWorkItemDimension << std::endl;

				//compute the max number of work-item that can be specified in each dimension of the work-group to clEnqueueNDRangeKernel
				deviceList[iDevice].getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES,&(sizeDim));
				mDeviceProperties[iTotalDevice].maxWorkItemSize.x=sizeDim[0];
				mDeviceProperties[iTotalDevice].maxWorkItemSize.y=sizeDim[1];
				mDeviceProperties[iTotalDevice].maxWorkItemSize.z=sizeDim[2];
				//std::cout << "		Max work-item Sizes: [" << mDeviceProperties[iTotalDevice].maxWorkItemSize.x << "," << mDeviceProperties[iTotalDevice].maxWorkItemSize.y << ","<< mDeviceProperties[iTotalDevice].maxWorkItemSize.z << "]"<< std::endl;

				//get vendor name to obtain the warps size
				deviceList[iDevice].getInfo(CL_DEVICE_VENDOR,&(mDeviceProperties[iTotalDevice].vendor));
				//std::cout << "		Device vendor: " << mDeviceProperties[iTotalDevice].vendor << std::endl;
				if(mDeviceProperties[iTotalDevice].vendor.find("NVIDIA")!=std::string::npos){
					//std::cout<<">> NVIDIA" << std::endl;
					deviceList[iDevice].getInfo(NVIDIA_WAVEFRONT,&(mDeviceProperties[iTotalDevice].warpSize));
				}
				else if(mDeviceProperties[iTotalDevice].vendor.find("AMD")!=std::string::npos){
					//std::cout<<">> AMD" << std::endl;
					deviceList[iDevice].getInfo(AMD_WAVEFRONT,&(mDeviceProperties[iTotalDevice].warpSize));
				}
				else{
					//std::cout<<">> NOT NVIDIA/AMD" << std::endl;
					mDeviceProperties[iTotalDevice].warpSize=16;
				}



				//store the context
				mDeviceProperties[iTotalDevice].oclContext=context;

				//store the device
				mDeviceProperties[iTotalDevice].oclDevice=deviceList[iDevice];

				iTotalDevice++;
			}

			//std::cout << std::endl;
		}


	}
	catch(const cl::Error &err){
		std::string errString=Utils::OCLErr_code(err.err());
		throw std::runtime_error { errString };
	}

	iCurrentDevice=1;
	//std::cout << std::endl<< ">> First device is selected" << std::endl;
	//create command queue associated to selected device
	try{
		mDeviceProperties[iCurrentDevice].oclQueue=cl::CommandQueue(mDeviceProperties[iCurrentDevice].oclContext, mDeviceProperties[iCurrentDevice].oclDevice, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE|CL_QUEUE_PROFILING_ENABLE );
	}catch(const cl::Error &err){
			std::string errString=Utils::OCLErr_code(err.err());
			//std::cout<< errString << std::endl;
			throw std::runtime_error { errString };
		}


}


Context& Context::getInstance()
{
  //std::cout << "Context" << std::endl;
  static Context gpuContext;
  return gpuContext;
}

const DeviceProperties& Context::getDeviceProperties()
{
  return getDeviceProperties(iCurrentDevice);
}

const DeviceProperties& Context::getDeviceProperties(const int deviceIndex)
{
	return mDeviceProperties[deviceIndex];

}

}
}
}
}
