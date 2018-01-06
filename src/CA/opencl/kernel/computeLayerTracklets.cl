//============================================================================
// Name        : KernelTest.cpp
// Author      : frank
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

/*
#define __kernel
#define __constant const
#define __global
*/
__constant float TrackletMaxDeltaZThreshold[6]= { 0.1f, 0.1f, 0.3f, 0.3f, 0.3f, 0.3f }; //default
//__constant float TrackletMaxDeltaZThreshold[6]= { 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f };
__constant int ZBins=20;
__constant int PhiBins=20;
__constant float Pi=3.14159265359f;
__constant float TwoPi=2.0f * 3.14159265359f ;
__constant int UnusedIndex=-1 ;
__constant float CellMaxDeltaPhiThreshold=0.14f;
__constant float PhiCoordinateCut=0.3f;	//default
//__constant float PhiCoordinateCut=PHICUT;	//default

__constant float ZCoordinateCut=0.5f;	//default
__constant float InversePhiBinSize=20 / (2.0f * 3.14159265359f) ;
__constant float LayersZCoordinate[7]={16.333f, 16.333f, 16.333f, 42.140f, 42.140f, 73.745f, 73.745f};
__constant float LayersRCoordinate[7]={2.33959f, 3.14076f, 3.91924f, 19.6213f, 24.5597f, 34.388f, 39.3329f};
__constant float InverseZBinSize[7]=  {0.5f * 20 / 16.333f, 0.5f * 20 / 16.333f, 0.5f * 20 / 16.333f,0.5f * 20 / 42.140f, 0.5f * 20 / 42.140f, 0.5f * 20 / 73.745f, 0.5f * 20 / 73.745f };


	typedef struct{
		int x;
		int y;
		int z;
		int w;
	}Int4Struct;

	typedef struct{
		float x;
		float y;
		float z;
	}Float3Struct;

	typedef struct{
		float xCoordinate;
		float yCoordinate;
		float zCoordinate;
		float phiCoordinate;
		float rCoordinate;
		int clusterId;
		float alphaAngle;
		int monteCarloId;
		int indexTableBinIndex;
	}ClusterStruct;

	typedef struct{
		int mFirstClusterIndex;
		int mSecondClusterIndex;
		int mThirdClusterIndex;
		int mFirstTrackletIndex;
		int mSecondTrackletIndex;
		Float3Struct mNormalVectorCoordinates;
		float mCurvature;
		int mLevel;
	}CellStruct;

	typedef struct{
		int firstClusterIndex;
		int secondClusterIndex;
		float tanLambda;
		float phiCoordinate;
	}TrackletStruct;

	typedef struct{
		int firstClusterIndex;
		int secondClusterIndex;
		float tanLambda;
		float phiCoordinate;
	}RoadsStruct;

	typedef struct{
		void * srPunt;
		int size;
	}VectStruct;


int myMin(int a,int b){
	if(a<b)
		return a;
	return b;
}

int myMax(int a,int b){
	if(a>b)
		return a;
	return b;
}

double myAbs(double iNum){
	if(iNum<0)
		return -1*iNum;
	return iNum;
}



int getZBinIndex(int layerIndex, float zCoordinate){
	return (zCoordinate + LayersZCoordinate[layerIndex])* InverseZBinSize[layerIndex];
}

int getPhiBinIndex(float currentPhi)
{
  return (currentPhi * InversePhiBinSize);
}

float getNormalizedPhiCoordinate(float phiCoordinate)
{
  return (phiCoordinate < 0) ? phiCoordinate + TwoPi :
         (phiCoordinate > TwoPi) ? phiCoordinate - TwoPi : phiCoordinate;
}


Int4Struct getBinsRect(__global ClusterStruct* currentCluster, int layerIndex,float directionZIntersection)
{
	const float zRangeMin = directionZIntersection - 2 * ZCoordinateCut;
	const float phiRangeMin = currentCluster->phiCoordinate - PhiCoordinateCut;
	const float zRangeMax = directionZIntersection + 2 * ZCoordinateCut;
	const float phiRangeMax = currentCluster->phiCoordinate + PhiCoordinateCut;
	Int4Struct binRect;
	binRect.x=0;
	binRect.y=0;
	binRect.z=0;
	binRect.w=0;

	if (zRangeMax < -LayersZCoordinate[layerIndex + 1]|| zRangeMin > LayersZCoordinate[layerIndex + 1] || zRangeMin > zRangeMax) {
		return binRect;
	}

	binRect.x=myMax(0, getZBinIndex(layerIndex + 1, zRangeMin));
	binRect.y=getPhiBinIndex(getNormalizedPhiCoordinate(phiRangeMin));
	binRect.z=myMin(ZBins - 1, getZBinIndex(layerIndex + 1, zRangeMax));
	binRect.w=getPhiBinIndex(getNormalizedPhiCoordinate(phiRangeMax));
	return binRect;

}


int getBinIndex(int zIndex,int phiIndex)
{
	return myMin(phiIndex * PhiBins + zIndex,ZBins * PhiBins);
}

__kernel void openClScan(__global int *in, __global int *out)
{
	int in_data;
	int i = get_global_id(0);
	const int numberOfClusterForCurrentLayer=get_global_size(0);
	
	in_data = in[i];
	out[i] = work_group_scan_exclusive_add(in_data);

}

__kernel void countLayerTracklets(
				__global Float3Struct* primaryVertex,	//0
				__global ClusterStruct* currentLayerClusters, //1
				__global ClusterStruct* nextLayerClusters, //2
				__global int * currentLayerIndexTable, //3
				__global int * iCurrentLayer, //4
				__global int * iCurrentTrackletsPosition, //5
				__global int * iLayerClusterSize, //6
				__global int * iTrackletsPerClusterTablePreviousLayer //7
		)
					
{
	const int currentClusterIndex=get_global_id(0);
	int clusterTrackletsNum=0;
	int iLayer=*iCurrentLayer;
	
		//printf("pippo=%d\n",currentClusterIndex);
	int currentLayerClusterVectorSize=iLayerClusterSize[iLayer];
	int nextLayerClusterVectorSize=iLayerClusterSize[iLayer+1];

	//if(currentClusterIndex==0)
		//printf("%d %d %d\n",iLayer,currentLayerClusterVectorSize,nextLayerClusterVectorSize);
	if(currentClusterIndex<currentLayerClusterVectorSize){
		__global ClusterStruct *currentCluster=&currentLayerClusters[currentClusterIndex];
		
		float tanLambda=(currentCluster->zCoordinate-primaryVertex->z)/currentCluster->rCoordinate;

		float directionZIntersection= tanLambda*(LayersRCoordinate[iLayer+1]-currentCluster->rCoordinate)+currentCluster->zCoordinate;

		const Int4Struct selectedBinsRect=getBinsRect(currentCluster,iLayer,directionZIntersection);
		if (selectedBinsRect.x != 0 || selectedBinsRect.y != 0 || selectedBinsRect.z != 0 || selectedBinsRect.w != 0) {
	      	const int nextLayerClustersNum=nextLayerClusterVectorSize;
	      	int phiBinsNum=selectedBinsRect.w - selectedBinsRect.y + 1;
			if(phiBinsNum<0){
		    	  phiBinsNum+=PhiBins;
		    }
			
		    for(int iPhiBin=selectedBinsRect.y,iPhiCount=0;iPhiCount < phiBinsNum;iPhiBin = ++iPhiBin == PhiBins ? 0 : iPhiBin, iPhiCount++){
		    	   
		    	  const int firstBinIndex=getBinIndex(selectedBinsRect.x,iPhiBin);
		    	  const int firstRowClusterIndex = currentLayerIndexTable[firstBinIndex];
		    	  const int maxRowClusterIndex = currentLayerIndexTable[firstBinIndex+ selectedBinsRect.z - selectedBinsRect.x + 1 ];
				  //if(iLayer==0)
					  //printf("%d %d\n",firstRowClusterIndex,nextLayerClustersNum);
				  for (int iNextLayerCluster=firstRowClusterIndex;iNextLayerCluster <= maxRowClusterIndex && iNextLayerCluster < nextLayerClustersNum; ++iNextLayerCluster) {
		    		
		    		  __global ClusterStruct *nextCluster=&nextLayerClusters[iNextLayerCluster];
		    		 
		    		  const float deltaZ=myAbs(tanLambda * (nextCluster->rCoordinate - currentCluster->rCoordinate) + currentCluster->zCoordinate - nextCluster->zCoordinate);
		    		  const float deltaPhi=myAbs(currentCluster->phiCoordinate - nextCluster->phiCoordinate);

		    		  if (deltaZ < TrackletMaxDeltaZThreshold[iLayer] && (deltaPhi<PhiCoordinateCut || myAbs(deltaPhi-TwoPi)<PhiCoordinateCut)){
		    			  int iTrackletPosition=atom_inc(&iCurrentTrackletsPosition[iLayer])+1;
		    			  /*__global TrackletStruct* tracklet=&currentLayerTracklets[iTrackletPosition];
		    			  tracklet->firstClusterIndex=currentClusterIndex;
		    			  tracklet->secondClusterIndex=iNextLayerCluster;
		    			  tracklet->tanLambda=(currentCluster->zCoordinate - nextCluster->zCoordinate) / (currentCluster->rCoordinate - nextCluster->rCoordinate);
		    			  tracklet->phiCoordinate= atan2(currentCluster->yCoordinate - nextCluster->yCoordinate, currentCluster->xCoordinate - nextCluster->xCoordinate);
						  *///if(iLayer==0)
							//  printf("%d %d\n",tracklet->firstClusterIndex,tracklet->secondClusterIndex);
		    			  ++clusterTrackletsNum;
		    			  
		    		  }
		    	  }
		      }
		}
	}	
	if(clusterTrackletsNum>0) {
	// 	printf("iLayer = %d,\tclusterTrackletsNum = %d,\tcurrentClusterIndex = %d\n",iLayer,clusterTrackletsNum,currentClusterIndex);
		iTrackletsPerClusterTablePreviousLayer[currentClusterIndex] = clusterTrackletsNum;
  	}
  	else{
    	iTrackletsPerClusterTablePreviousLayer[currentClusterIndex] = 0;
  	}	
     
     
}







__kernel void computeLayerTracklets(
				__global Float3Struct* primaryVertex,	//0
				__global ClusterStruct* currentLayerClusters, //1
				__global ClusterStruct* nextLayerClusters, //2
				__global int * currentLayerIndexTable, //3
				__global TrackletStruct* currentLayerTracklets, //4
				__global int * iCurrentLayer, //5
				__global int * iCurrentTrackletsPosition, //6
				__global int * iLayerClusterSize, //7
				__global int * iTrackletsPerClusterTablePreviousLayer //8
		)
					
{
	const int currentClusterIndex=get_global_id(0);
	const int numberOfClusterForCurrentLayer=get_global_size(0);
	if(currentClusterIndex==numberOfClusterForCurrentLayer)
		return;	//to fix
	
	int clusterTrackletsNum=0;
	int iLayer=*iCurrentLayer;
	
	int currentLayerClusterVectorSize=iLayerClusterSize[iLayer];
	int nextLayerClusterVectorSize=iLayerClusterSize[iLayer+1];
	
	int currentLookUpValue=iTrackletsPerClusterTablePreviousLayer[currentClusterIndex];
	int nextLookUpValue=iTrackletsPerClusterTablePreviousLayer[currentClusterIndex+1];
	int numberOftrackletToFind=nextLookUpValue-currentLookUpValue;
	
	if(currentLookUpValue==nextLookUpValue)
		return;

	if(currentClusterIndex<currentLayerClusterVectorSize){
		__global ClusterStruct *currentCluster=&currentLayerClusters[currentClusterIndex];
		
		float tanLambda=(currentCluster->zCoordinate-primaryVertex->z)/currentCluster->rCoordinate;

		float directionZIntersection= tanLambda*(LayersRCoordinate[iLayer+1]-currentCluster->rCoordinate)+currentCluster->zCoordinate;

		const Int4Struct selectedBinsRect=getBinsRect(currentCluster,iLayer,directionZIntersection);
		if (selectedBinsRect.x != 0 || selectedBinsRect.y != 0 || selectedBinsRect.z != 0 || selectedBinsRect.w != 0) {
	      	const int nextLayerClustersNum=nextLayerClusterVectorSize;
	      	int phiBinsNum=selectedBinsRect.w - selectedBinsRect.y + 1;
			if(phiBinsNum<0){
		    	  phiBinsNum+=PhiBins;
		    }
			
		    for(int iPhiBin=selectedBinsRect.y,iPhiCount=0;iPhiCount < phiBinsNum;iPhiBin = ++iPhiBin == PhiBins ? 0 : iPhiBin, iPhiCount++){
		    	   
		    	  const int firstBinIndex=getBinIndex(selectedBinsRect.x,iPhiBin);
		    	  const int firstRowClusterIndex = currentLayerIndexTable[firstBinIndex];
		    	  const int maxRowClusterIndex = currentLayerIndexTable[firstBinIndex+ selectedBinsRect.z - selectedBinsRect.x + 1 ];
				  //if(iLayer==0)
					  //printf("%d %d\n",firstRowClusterIndex,nextLayerClustersNum);
				  for (int iNextLayerCluster=firstRowClusterIndex;iNextLayerCluster <= maxRowClusterIndex && iNextLayerCluster < nextLayerClustersNum; ++iNextLayerCluster) {
		    		
		    		  __global ClusterStruct *nextCluster=&nextLayerClusters[iNextLayerCluster];
		    		 
		    		  const float deltaZ=myAbs(tanLambda * (nextCluster->rCoordinate - currentCluster->rCoordinate) + currentCluster->zCoordinate - nextCluster->zCoordinate);
		    		  const float deltaPhi=myAbs(currentCluster->phiCoordinate - nextCluster->phiCoordinate);

		    		  if (deltaZ < TrackletMaxDeltaZThreshold[iLayer] && (deltaPhi<PhiCoordinateCut || myAbs(deltaPhi-TwoPi)<PhiCoordinateCut)){
		    			  __global TrackletStruct* tracklet=&currentLayerTracklets[currentLookUpValue];
		    			  
		    			  
		    			  tracklet->firstClusterIndex=currentClusterIndex;
		    			  tracklet->secondClusterIndex=iNextLayerCluster;
		    			  tracklet->tanLambda=(currentCluster->zCoordinate - nextCluster->zCoordinate) / (currentCluster->rCoordinate - nextCluster->rCoordinate);
		    			  tracklet->phiCoordinate= atan2(currentCluster->yCoordinate - nextCluster->yCoordinate, currentCluster->xCoordinate - nextCluster->xCoordinate);
						  currentLookUpValue++;
						  if(currentLookUpValue==nextLookUpValue)
						  	return;
		    					  
		    		  }
		    	  }
		      }
		}
	}	
		

}


