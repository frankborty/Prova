BEGIN {
  meanTrackletsTime = .0;
  meanCellsTime = .0;
  meanTotalTime = .0;
  rowsCount = 0;

  minTrackletsTime = 2^53 - 1;
  minCellsTime = 2^53 - 1;
  minTotalTime = 2^53 - 1;
  
  maxTrackletsTime = .0;
  maxCellsTime = .0;
  maxTotalTime = .0;
}
{
  meanTrackletsTime += $1;
  meanCellsTime += $2;
  meanTotalTime += $6
  rowsCount++;

  if(minTrackletsTime > $1) {

    minTrackletsTime = $1;
  }

  if(minCellsTime > $2) {
    
    minCellsTime = $2;
  }

  if(minTotalTime > $6) {
    
    minTotalTime = $6;
  }

  if(maxTrackletsTime < $1) {
  
    maxTrackletsTime = $1;
  }

  if(maxCellsTime < $2) {

    maxCellsTime = $2;
  }

  if(maxTotalTime < $6) {

    maxTotalTime = $6;
  }
}
END {
  print meanTrackletsTime / rowsCount, minTrackletsTime, maxTrackletsTime
  print meanCellsTime / rowsCount, minCellsTime, maxCellsTime
  print meanTotalTime / rowsCount, minTotalTime, maxTotalTime
}
