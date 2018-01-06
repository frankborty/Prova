typedef struct{
	int firstClusterIndex;
	int secondClusterIndex;
	float tanLambda;
	float phiCoordinate;
}TrackletStruct;



// One thread per record
__kernel void ParallelSelection(__global TrackletStruct * in,__global TrackletStruct * out)
{
  int i = get_global_id(0); // current thread
  int n = get_global_size(0); // input size
  
  
  TrackletStruct iData = in[i];
  printf("%d\t%d\t%d\n",i,in[i].firstClusterIndex,in[i].secondClusterIndex);
  
  uint iKey = iData.firstClusterIndex;
  uint iSecondKey = iData.secondClusterIndex;
  // Compute position of in[i] in output
  
  int pos = 0;
  for (int j=0;j<n;j++)
  {
    uint jKey = in[j].firstClusterIndex; // broadcasted
    uint jSecondKey = in[j].secondClusterIndex; // broadcasted
    //printf("iKey=%d jKey=%d\n",iKey,jKey);
    bool smaller = (jKey < iKey) || (jKey == iKey && jSecondKey < iSecondKey); // in[j] < in[i] ?
    pos += (smaller)?1:0;
  }
  
  	out[pos].firstClusterIndex=iData.firstClusterIndex;
	out[pos].secondClusterIndex=iData.secondClusterIndex;
	out[pos].tanLambda=iData.tanLambda;
	out[pos].phiCoordinate=iData.phiCoordinate;
  
}