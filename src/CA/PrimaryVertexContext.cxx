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

#include "ITSReconstruction/CA/PrimaryVertexContext.h"

#include "ITSReconstruction/CA/Event.h"

#ifdef TRACKINGITSU_OCL_MODE
#include "ITSReconstruction/CA/gpu/Context.h"
#include "ITSReconstruction/CA/gpu/Utils.h"
#include "ITSReconstruction/CA/gpu/Utils.h"
#endif

namespace o2
{
namespace ITS
{
namespace CA
{

PrimaryVertexContext::PrimaryVertexContext()
{
  // Nothing to do

}



void PrimaryVertexContext::initialize(const Event& event, const int primaryVertexIndex) {
#ifdef TRACKINGITSU_OCL_MODE
	time_t t1,t2;

	float diff;
	t1=clock();
	cl::Context oclContext=GPU::Context::getInstance().getDeviceProperties().oclContext;
	t2=clock();
	diff = ((float) t2 - (float) t1) / (CLOCKS_PER_SEC / 1000);
	std::cout << "OClContext = "<<diff << std::endl;
	openClPrimaryVertexContext.initialize(oclContext);

	float3 mPrimaryVertex=event.getPrimaryVertex(primaryVertexIndex);
	openClPrimaryVertexContext.mPrimaryVertex.x=mPrimaryVertex.x;
	openClPrimaryVertexContext.mPrimaryVertex.y=mPrimaryVertex.y;
	openClPrimaryVertexContext.mPrimaryVertex.z=mPrimaryVertex.z;

	openClPrimaryVertexContext.bPrimaryVertex=cl::Buffer(
			oclContext,
			(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			3*sizeof(float),
			(void *) &(openClPrimaryVertexContext.mPrimaryVertex));

	//clusters
	t1=clock();
	for (int iLayer { 0 }; iLayer < Constants::ITS::LayersNumber; ++iLayer) {

		const Layer& currentLayer { event.getLayer(iLayer) };
		const int clustersNum { currentLayer.getClustersSize() };

		if(openClPrimaryVertexContext.mClusters[iLayer]!=NULL)
			free(openClPrimaryVertexContext.mClusters[iLayer]);

		//check if size of clusters is multiple of page size, otherwise extend clusters with 0 value
		int factor;
		int clusterSize=clustersNum*sizeof(ClusterStruct);
	/*	factor=clusterSize%64;
		if(factor!=0){
			factor++;
			clusterSize+=clusterSize;
		}
		//openClPrimaryVertexContext.mClusters[iLayer]=(ClusterStruct*)aligned_alloc(4096,clusterSize);
/*
		int res=posix_memalign((void**)&openClPrimaryVertexContext.mClusters[iLayer],4096,clusterSize);
		if(res!=0){
			std::cout<<"layer = "<<iLayer<<"\t clusters = "<<res<<std::endl;
		}
*/
		openClPrimaryVertexContext.mClusters[iLayer]=(ClusterStruct*)malloc(clustersNum*sizeof(ClusterStruct));
		openClPrimaryVertexContext.iClusterSize[iLayer]=clustersNum*sizeof(ClusterStruct);
		openClPrimaryVertexContext.iClusterAllocatedSize[iLayer]=clusterSize;

		for (int iCluster { 0 }; iCluster < clustersNum; ++iCluster) {
			const Cluster& currentCluster { currentLayer.getCluster(iCluster) };
			openClPrimaryVertexContext.addClusters(mPrimaryVertex,currentCluster,iLayer,iCluster);

		}
		openClPrimaryVertexContext.iClusterSize[iLayer]=clustersNum;

		openClPrimaryVertexContext.sortClusters(iLayer);

		openClPrimaryVertexContext.bClusters[iLayer]=cl::Buffer(
				oclContext,
				(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
				clusterSize,
				(void *) &(openClPrimaryVertexContext.mClusters[iLayer][0]));

	}
	for (int iLayer { 0 }; iLayer < Constants::ITS::LayersNumber; ++iLayer) {
		const int clustersNum = static_cast<int>(openClPrimaryVertexContext.iClusterSize[iLayer]);

	    //index table
	    if(iLayer > 0) {

			int previousBinIndex { 0 };
			openClPrimaryVertexContext.mIndexTables[iLayer - 1][0] = 0;

			for (int iCluster { 0 }; iCluster < clustersNum; ++iCluster) {
				const int currentBinIndex { openClPrimaryVertexContext.mClusters[iLayer][iCluster].indexTableBinIndex };
				if (currentBinIndex > previousBinIndex) {
					for (int iBin { previousBinIndex + 1 }; iBin <= currentBinIndex; ++iBin) {
						openClPrimaryVertexContext.mIndexTables[iLayer - 1][iBin] = iCluster;
					}
					previousBinIndex = currentBinIndex;
				}
			}

			for (int iBin { previousBinIndex + 1 }; iBin <= Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins;iBin++) {
				openClPrimaryVertexContext.mIndexTables[iLayer - 1][iBin] = clustersNum;
			}

			openClPrimaryVertexContext.bIndexTables[iLayer-1]=cl::Buffer(
				oclContext,
				(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
				openClPrimaryVertexContext.iIndexTableSize*sizeof(int),
				(void *) &(openClPrimaryVertexContext.mIndexTables[iLayer-1][0]));


	    }
	    //tracklets
	    if(iLayer < Constants::ITS::TrackletsPerRoad) {
	    	if(openClPrimaryVertexContext.mTracklets[iLayer]!=NULL)
	    		free(openClPrimaryVertexContext.mTracklets[iLayer]);

	      float trackletsMemorySize = std::ceil((Constants::Memory::TrackletsMemoryCoefficients[iLayer] * event.getLayer(iLayer).getClustersSize())
	         * event.getLayer(iLayer + 1).getClustersSize());
	      int trackletSize=trackletsMemorySize*sizeof(TrackletStruct);
	    /*  int factor=trackletSize%64;
	      if(factor!=0){
			factor++;
			trackletSize=factor*trackletSize;
	      }
/*
	      int res=posix_memalign((void**)&openClPrimaryVertexContext.mTracklets[iLayer],4096,trackletSize);
	      if(res!=0){
	    	  std::cout<<"layer = "<<iLayer<<"\t trackletSize result = "<<res<<std::endl;
	      }
*/
	      openClPrimaryVertexContext.mTracklets[iLayer]=(TrackletStruct*)malloc(trackletSize);//delete
	      openClPrimaryVertexContext.bTracklets[iLayer]=cl::Buffer(
				oclContext,
				(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
				trackletSize,
				(void *) &(openClPrimaryVertexContext.mTracklets[iLayer][0]));

	    }

	    //tracklets lookup
	    if(iLayer < Constants::ITS::CellsPerRoad) {
	    	if(openClPrimaryVertexContext.mTrackletsLookupTable[iLayer]!=NULL)
	    		free(openClPrimaryVertexContext.mTrackletsLookupTable[iLayer]);
	    	int size=event.getLayer(iLayer + 1).getClustersSize()*sizeof(int);
	      	int lookUpSize=size;
			int factor=lookUpSize%64;
			if(factor!=0){
				factor++;
				lookUpSize=factor*lookUpSize;
			}
			//openClPrimaryVertexContext.mTrackletsLookupTable[iLayer]=(int*)aligned_alloc(4096,lookUpSize);
			int res=posix_memalign((void**)&openClPrimaryVertexContext.mTrackletsLookupTable[iLayer],4096,lookUpSize);
			if(res!=0){
				std::cout<<"layer = "<<iLayer<<"\t tracklets lookup = "<<res<<std::endl;
			}
			openClPrimaryVertexContext.iTrackletsLookupTableAllocatedSize[iLayer]=lookUpSize;
	    	openClPrimaryVertexContext.iTrackletsLookupTableSize[iLayer]=size;
			memset(openClPrimaryVertexContext.mTrackletsLookupTable[iLayer],-1,lookUpSize);

			openClPrimaryVertexContext.bTrackletsLookupTable[iLayer]=cl::Buffer(
				oclContext,
				(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
				lookUpSize,
				(void *) &(openClPrimaryVertexContext.mTrackletsLookupTable[iLayer][0]));
	    }
	}
	openClPrimaryVertexContext.bClustersSize=cl::Buffer(
					oclContext,
					(cl_mem_flags)CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
					7*sizeof(int),
					(void *) &(openClPrimaryVertexContext.iClusterSize[0]));

#else
  mPrimaryVertex = event.getPrimaryVertex(primaryVertexIndex);
  time_t t1,t2;
  float diff;
  for (int iLayer { 0 }; iLayer < Constants::ITS::LayersNumber; ++iLayer) {

    const Layer& currentLayer { event.getLayer(iLayer) };
    const int clustersNum { currentLayer.getClustersSize() };

    mClusters[iLayer].clear();

    if(clustersNum > (int)mClusters[iLayer].capacity()) {

      mClusters[iLayer].reserve(clustersNum);
    }

    for (int iCluster { 0 }; iCluster < clustersNum; ++iCluster) {
      const Cluster& currentCluster { currentLayer.getCluster(iCluster) };
      mClusters[iLayer].emplace_back(iLayer, event.getPrimaryVertex(primaryVertexIndex), currentCluster);
    }
//    t1=clock();
    std::sort(mClusters[iLayer].begin(), mClusters[iLayer].end(), [](Cluster& cluster1, Cluster& cluster2) {
      return cluster1.indexTableBinIndex < cluster2.indexTableBinIndex;
    });
/*    t2=clock();
    diff = ((float) t2 - (float) t1) / (CLOCKS_PER_SEC / 1000);
    	std::cout <<"Layer = "<<iLayer<<"\tSort ="<< diff << std::endl;
    	*/
    if(iLayer < Constants::ITS::CellsPerRoad) {

      mCells[iLayer].clear();
      float cellsMemorySize = std::ceil(((Constants::Memory::CellsMemoryCoefficients[iLayer] * event.getLayer(iLayer).getClustersSize())
         * event.getLayer(iLayer + 1).getClustersSize()) * event.getLayer(iLayer + 2).getClustersSize());

      if(cellsMemorySize > mCells[iLayer].capacity()) {

        mCells[iLayer].reserve(cellsMemorySize);
      }
    }

    if(iLayer < Constants::ITS::CellsPerRoad - 1) {

      mCellsLookupTable[iLayer].clear();
      mCellsLookupTable[iLayer].resize(std::ceil(
        (Constants::Memory::TrackletsMemoryCoefficients[iLayer + 1] * event.getLayer(iLayer + 1).getClustersSize())
          * event.getLayer(iLayer + 2).getClustersSize()), Constants::ITS::UnusedIndex);


      mCellsNeighbours[iLayer].clear();
    }
  }

  for (int iLayer { 0 }; iLayer < Constants::ITS::CellsPerRoad - 1; ++iLayer) {

    mCellsNeighbours[iLayer].clear();
  }

  mRoads.clear();
/*#if TRACKINGITSU_GPU_MODE
  std::cerr<< "GPU primaryVertex"<< std::endl;
  mGPUContextDevicePointer = mGPUContext.initialize(mPrimaryVertex, mClusters, mCells, mCellsLookupTable);
  #else
*/
  for (int iLayer { 0 }; iLayer < Constants::ITS::LayersNumber; ++iLayer) {

    const int clustersNum = static_cast<int>(mClusters[iLayer].size());

    if(iLayer > 0) {

      int previousBinIndex { 0 };
      mIndexTables[iLayer - 1][0] = 0;

      for (int iCluster { 0 }; iCluster < clustersNum; ++iCluster) {

        const int currentBinIndex { mClusters[iLayer][iCluster].indexTableBinIndex };

        if (currentBinIndex > previousBinIndex) {

          for (int iBin { previousBinIndex + 1 }; iBin <= currentBinIndex; ++iBin) {

            mIndexTables[iLayer - 1][iBin] = iCluster;
          }

          previousBinIndex = currentBinIndex;
        }
      }

      for (int iBin { previousBinIndex + 1 }; iBin <= Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins;
          iBin++) {

        mIndexTables[iLayer - 1][iBin] = clustersNum;
      }
    }

    if(iLayer < Constants::ITS::TrackletsPerRoad) {

      mTracklets[iLayer].clear();

      float trackletsMemorySize = std::ceil((Constants::Memory::TrackletsMemoryCoefficients[iLayer] * event.getLayer(iLayer).getClustersSize())
         * event.getLayer(iLayer + 1).getClustersSize());

      if(trackletsMemorySize > mTracklets[iLayer].capacity()) {

        mTracklets[iLayer].reserve(trackletsMemorySize);
      }
    }

    if(iLayer < Constants::ITS::CellsPerRoad) {

      mTrackletsLookupTable[iLayer].clear();
      mTrackletsLookupTable[iLayer].resize(
         event.getLayer(iLayer + 1).getClustersSize(), Constants::ITS::UnusedIndex);
    }
  }


#endif



}

}
}
}
