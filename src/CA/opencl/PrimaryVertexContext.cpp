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
/// \file PrimaryVertexContext.cxx
/// \brief
///

#ifndef _TRAKINGITSU_INCLUDE_PRIMARY_VERTEX_CONTEXT_H_
#define _TRAKINGITSU_INCLUDE_PRIMARY_VERTEX_CONTEXT_H_

#include "ITSReconstruction/CA/gpu/PrimaryVertexContext.h"

#include <sstream>
#include <iostream>
#include "ITSReconstruction/CA/gpu/Stream.h"
#include "ITSReconstruction/CA/gpu/Context.h"
#include "ITSReconstruction/CA/gpu/Utils.h"

namespace {


using namespace o2::ITS::CA;

/*
__device__ void fillIndexTables(GPU::PrimaryVertexContext &primaryVertexContext, const int layerIndex)
{

  const int currentClusterIndex { static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x) };
  const int nextLayerClustersNum { static_cast<int>(primaryVertexContext.getClusters()[layerIndex + 1].size()) };

  if (currentClusterIndex < nextLayerClustersNum) {

    const int currentBinIndex {
        primaryVertexContext.getClusters()[layerIndex + 1][currentClusterIndex].indexTableBinIndex };
    int previousBinIndex;

    if (currentClusterIndex == 0) {

      primaryVertexContext.getIndexTables()[layerIndex][0] = 0;
      previousBinIndex = 0;

    } else {

      previousBinIndex = primaryVertexContext.getClusters()[layerIndex + 1][currentClusterIndex - 1].indexTableBinIndex;
    }

    if (currentBinIndex > previousBinIndex) {

      for (int iBin { previousBinIndex + 1 }; iBin <= currentBinIndex; ++iBin) {

        primaryVertexContext.getIndexTables()[layerIndex][iBin] = currentClusterIndex;
      }

      previousBinIndex = currentBinIndex;
    }

    if (currentClusterIndex == nextLayerClustersNum - 1) {

      for (int iBin { currentBinIndex + 1 }; iBin <= Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins;
          iBin++) {

        primaryVertexContext.getIndexTables()[layerIndex][iBin] = nextLayerClustersNum;
      }
    }
  }
}

__device__ void fillTrackletsPerClusterTables(GPU::PrimaryVertexContext &primaryVertexContext, const int layerIndex)
{
  const int currentClusterIndex { static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x) };
  const int clustersSize { static_cast<int>(primaryVertexContext.getClusters()[layerIndex + 1].size()) };

  if (currentClusterIndex < clustersSize) {

    primaryVertexContext.getTrackletsPerClusterTable()[layerIndex][currentClusterIndex] = 0;
  }
}

__device__ void fillCellsPerClusterTables(GPU::PrimaryVertexContext &primaryVertexContext, const int layerIndex)
{
  const int totalThreadNum { static_cast<int>(primaryVertexContext.getClusters()[layerIndex + 1].size()) };
  const int trackletsSize { static_cast<int>(primaryVertexContext.getTracklets()[layerIndex + 1].capacity()) };
  const int trackletsPerThread { 1 + (trackletsSize - 1) / totalThreadNum };
  const int firstTrackletIndex { static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x) * trackletsPerThread };

  if (firstTrackletIndex < trackletsSize) {

    const int trackletsToSet { min(trackletsSize, firstTrackletIndex + trackletsPerThread) - firstTrackletIndex };
    memset(&primaryVertexContext.getCellsPerTrackletTable()[layerIndex][firstTrackletIndex], 0,
        trackletsToSet * sizeof(int));
  }
}
*/
//void fillDeviceStructures(GPU::PrimaryVertexContext &primaryVertexContext, const int layerIndex)
//{
//	std::cout<< "fillDeviceStructures()" << std::endl;

  /*fillIndexTables(primaryVertexContext, layerIndex);

  if (layerIndex < Constants::ITS::CellsPerRoad) {

    fillTrackletsPerClusterTables(primaryVertexContext, layerIndex);
  }

  if (layerIndex < Constants::ITS::CellsPerRoad - 1) {

    fillCellsPerClusterTables(primaryVertexContext, layerIndex);
  }
  */
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

bool myClusterComparator(const ClusterStruct &lhs, const ClusterStruct &rhs)
{
	return lhs.indexTableBinIndex < rhs.indexTableBinIndex;
}



PrimaryVertexContext::PrimaryVertexContext()
{
  // Nothing to do
}

void  PrimaryVertexContext::sortClusters(int iLayer)
{
	std::sort(this->mClusters[iLayer],this->mClusters[iLayer]+this->iClusterSize[iLayer],myClusterComparator);
}


void PrimaryVertexContext::initialize(cl::Context oclContext)
{
	for(int i=0;i<Constants::ITS::LayersNumber;i++){
		this->bLayerIndex[i]=cl::Buffer(oclContext,
		(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof(int),
		(void *) &(i));
	}
	int iTrackletsFoundForLayer[]={0,0,0,0,0,0,0};
	this->bTrackletsFoundForLayer=cl::Buffer(
		oclContext,
		(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		o2::ITS::CA::Constants::ITS::LayersNumber*sizeof(int),
		(void *) iTrackletsFoundForLayer);
}

}
}
}
}

#endif /* _TRAKINGITSU_INCLUDE_PRIMARY_VERTEX_CONTEXT_H_ */
