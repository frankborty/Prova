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
/// \file PrimaryVertexContext.h
/// \brief
///
#ifndef _TRAKINGITSU_INCLUDE_GPU_PRIMARY_VERTEX_CONTEXT_H_
#define _TRAKINGITSU_INCLUDE_GPU_PRIMARY_VERTEX_CONTEXT_H_

#include "ITSReconstruction/CA/Cell.h"
#include "ITSReconstruction/CA/Cluster.h"
#include "ITSReconstruction/CA/Constants.h"
#include "ITSReconstruction/CA/Definitions.h"
#include "ITSReconstruction/CA/Tracklet.h"
#include "ITSReconstruction/CA/gpu/Array.h"
#include "ITSReconstruction/CA/gpu/UniquePointer.h"
#include "ITSReconstruction/CA/gpu/Vector.h"
#include "ITSReconstruction/CA/IndexTableUtils.h"
#ifdef TRACKINGITSU_OCL_MODE
#include <CL/cl.hpp>
#include "ITSReconstruction/CA/gpu/StructGPUPrimaryVertex.h"
#endif

namespace o2
{
namespace ITS
{
namespace CA
{
namespace GPU
{

class PrimaryVertexContext
  final
  {
    public:
      PrimaryVertexContext();

      void initialize(cl::Context oclContext);
      void sortClusters(int iLayer);


      GPU_DEVICE const Float3Struct* getPrimaryVertex();
      GPU_HOST_DEVICE ClusterStruct** getClusters();
      GPU_HOST_DEVICE inline void addClusters(const float3 &primaryVertex, const Cluster& other, int iLayer,int iCluster);
      GPU_HOST_DEVICE TrackletStruct** getTracklets();
      GPU_HOST_DEVICE int** getTrackletsLookupTable();
      GPU_HOST_DEVICE int** getTrackletsPerClusterTable();


    public:
     Float3Struct mPrimaryVertex;
     cl::Buffer bPrimaryVertex;

     cl::Buffer bLayerIndex[Constants::ITS::LayersNumber];

     ClusterStruct* mClusters[Constants::ITS::LayersNumber];
     cl::Buffer bClusters[Constants::ITS::LayersNumber];
     cl::Buffer bClustersSize;
     int iClusterSize[Constants::ITS::LayersNumber];
     int iClusterAllocatedSize[Constants::ITS::LayersNumber];

     int iIndexTableSize=Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins + 1;
     int mIndexTables[Constants::ITS::TrackletsPerRoad][Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins + 1];
     cl::Buffer bIndexTables[Constants::ITS::TrackletsPerRoad];


     TrackletStruct* mTracklets[Constants::ITS::TrackletsPerRoad];
     cl::Buffer bTracklets[Constants::ITS::TrackletsPerRoad];
     int iTrackletSize[Constants::ITS::TrackletsPerRoad];
     int iTrackletAllocatedSize[Constants::ITS::TrackletsPerRoad];
     cl::Buffer bTrackletsSize;

     int* mTrackletsLookupTable[Constants::ITS::CellsPerRoad];
     int iTrackletsLookupTableSize[Constants::ITS::CellsPerRoad];
     int iTrackletsLookupTableAllocatedSize[Constants::ITS::CellsPerRoad];
     cl::Buffer bTrackletsLookupTable[Constants::ITS::CellsPerRoad];


     int* mTrackletsPerClusterTable[Constants::ITS::CellsPerRoad];

     cl::Buffer bTrackletsFoundForLayer;

  };


  inline const Float3Struct* PrimaryVertexContext::getPrimaryVertex()
  {
    return &mPrimaryVertex;
  }

  GPU_HOST_DEVICE inline ClusterStruct** PrimaryVertexContext::getClusters()
  {
    return mClusters;
  }

  GPU_HOST_DEVICE inline void PrimaryVertexContext::addClusters(const float3 &primaryVertex, const Cluster& other, int iLayer,int iCluster)
  {
	  	mClusters[iLayer][iCluster].xCoordinate=other.xCoordinate;
	  	mClusters[iLayer][iCluster].yCoordinate=other.yCoordinate;
	  	mClusters[iLayer][iCluster].zCoordinate=other.zCoordinate;
	  	mClusters[iLayer][iCluster].clusterId=other.clusterId;
	  	mClusters[iLayer][iCluster].monteCarloId=other.monteCarloId;
	  	mClusters[iLayer][iCluster].alphaAngle=other.alphaAngle;
		mClusters[iLayer][iCluster].phiCoordinate=MathUtils::getNormalizedPhiCoordinate(MathUtils::calculatePhiCoordinate(other.xCoordinate - primaryVertex.x, other.yCoordinate - primaryVertex.y));
		mClusters[iLayer][iCluster].rCoordinate=MathUtils::calculateRCoordinate(other.xCoordinate - primaryVertex.x, other.yCoordinate - primaryVertex.y);
		mClusters[iLayer][iCluster].indexTableBinIndex=IndexTableUtils::getBinIndex(IndexTableUtils::getZBinIndex(iLayer, other.zCoordinate),IndexTableUtils::getPhiBinIndex(mClusters[iLayer][iCluster].phiCoordinate)) ;
		/*if(iLayer==0 && (other.clusterId== 57 || other.clusterId==36))
				std::cout<<"pippo"<<std::endl;*/
  }
/*
  GPU_DEVICE inline int** PrimaryVertexContext::getIndexTables()
  {
    return mIndexTables;
  }
*/
  GPU_DEVICE inline TrackletStruct** PrimaryVertexContext::getTracklets()
  {
    return mTracklets;
  }

  GPU_DEVICE inline int** PrimaryVertexContext::getTrackletsLookupTable()
  {
    return mTrackletsLookupTable;
  }

  GPU_DEVICE inline int** PrimaryVertexContext::getTrackletsPerClusterTable()
  {
    return mTrackletsPerClusterTable;
  }


}
}
}
}

#endif /* _TRAKINGITSU_INCLUDE_GPU_PRIMARY_VERTEX_CONTEXT_H_ */
