#include <iomanip>
#include <iostream>
#include <limits>
#include <fstream>
#include <vector>

#include "ITSReconstruction/CA/Definitions.h"
#include "ITSReconstruction/CA/IOUtils.h"
#include "ITSReconstruction/CA/Tracker.h"

#if defined HAVE_VALGRIND
# include <valgrind/callgrind.h>
#endif

#if TRACKINGITSU_GPU_MODE
# include "ITSReconstruction/CA/gpu/Utils.h"
#include "ITSReconstruction/CA/gpu/myThresholds.h"
float myPhiThreshold;
float myZThreshold;
int myWorkGroupSize;
#endif

using namespace o2::ITS::CA;

std::string getDirectory(const std::string& fname)
{
  size_t pos = fname.find_last_of("\\/");
  return (std::string::npos == pos) ? "" : fname.substr(0, pos + 1);
}

int main(int argc, char** argv)
{

#if TRACKINGITSU_CUDA_MODE
	std::cout << ">> CUDA MODE"<< std::endl;
#elif TRACKINGITSU_OCL_MODE
	std::cout << ">> OCL MODE"<< std::endl;
#else
	std::cout << ">> CPU MODE"<< std::endl;
#endif


  if (argv[1] == NULL) {
    std::cerr << "Please, provide a data file." << std::endl;
    exit(EXIT_FAILURE);
  }
#if TRACKINGITSU_GPU_MODE
  if (argc>= 3) {
	  myWorkGroupSize=atoi(argv[2]);
  }
  else
	  myWorkGroupSize=16;
myPhiThreshold=Constants::Thresholds::PhiCoordinateCut;
myZThreshold=Constants::Thresholds::TrackletMaxDeltaZThreshold()[0];
#endif
  std::string eventsFileName(argv[1]);
  std::string benchmarkFolderName = getDirectory(eventsFileName);
  std::vector<Event> events = IOUtils::loadEventData(eventsFileName);
  const int eventsNum = events.size();
  if(eventsNum < 1){
	  std::cerr << "Input file is empty" << std::endl;
	  exit(EXIT_FAILURE);
  }

  std::vector<std::unordered_map<int, Label>> labelsMap;
  bool createBenchmarkData = false;
  std::ofstream correctRoadsOutputStream;
  std::ofstream duplicateRoadsOutputStream;
  std::ofstream fakeRoadsOutputStream;
  int verticesNum=0;
  for (int iEvent = 0; iEvent < eventsNum; ++iEvent) {
    verticesNum += events[iEvent].getPrimaryVerticesNum();
  }


  if (argv[2] != NULL) {

    std::string labelsFileName(argv[2]);

    createBenchmarkData = false;
    labelsMap = IOUtils::loadLabels(eventsNum, labelsFileName);

    correctRoadsOutputStream.open(benchmarkFolderName + "CorrectRoads.txt");
    duplicateRoadsOutputStream.open(benchmarkFolderName + "DuplicateRoads.txt");
    fakeRoadsOutputStream.open(benchmarkFolderName + "FakeRoads.txt");
  }


  clock_t t1, t2;
  float totalTime = 0.f, minTime = std::numeric_limits<float>::max(), maxTime = -1;
#if defined MEMORY_BENCHMARK
  std::ofstream memoryBenchmarkOutputStream;
  memoryBenchmarkOutputStream.open(benchmarkFolderName + "MemoryOccupancy.txt");
#elif defined TIME_BENCHMARK
  std::ofstream timeBenchmarkOutputStream;
  timeBenchmarkOutputStream.open("../TimeOccupancy.txt");
#endif

  // Prevent cold cache benchmark noise
  Tracker<TRACKINGITSU_GPU_MODE> tracker{};
  tracker.clustersToTracks(events[0]);
  //tracker.clustersToTracksTimeBenchmark(events[0],timeBenchmarkOutputStream);

#if defined GPU_PROFILING_MODE
  Utils::Host::gpuStartProfiler();
#endif

  for (size_t iEvent = 0; iEvent < events.size(); ++iEvent) {

    Event& currentEvent = events[iEvent];
    std::cout << "Processing event " << iEvent + 1 << std::endl;

    t1 = clock();

#if defined HAVE_VALGRIND
    // Run callgrind with --collect-atstart=no
    CALLGRIND_TOGGLE_COLLECT;
#endif

    try {
#if defined(MEMORY_BENCHMARK)
      std::vector<std::vector<Road>> roads = tracker.clustersToTracksMemoryBenchmark(currentEvent, memoryBenchmarkOutputStream);
#elif defined(DEBUG)
      std::vector<std::vector<Road>> roads = tracker.clustersToTracksVerbose(currentEvent);
#elif defined TIME_BENCHMARK
      std::vector<std::vector<Road>> roads = tracker.clustersToTracksTimeBenchmark(currentEvent, timeBenchmarkOutputStream);
#else
      std::vector<std::vector<Road>> roads = tracker.clustersToTracks(currentEvent);
#endif

#if defined HAVE_VALGRIND
      CALLGRIND_TOGGLE_COLLECT;
#endif

      t2 = clock();
      const float diff = ((float) t2 - (float) t1) / (CLOCKS_PER_SEC / 1000);

      totalTime += diff;

      if (minTime > diff)
        minTime = diff;
      if (maxTime < diff)
        maxTime = diff;
/*
      for(int iVertex = 0; iVertex < currentEvent.getPrimaryVerticesNum(); ++iVertex) {

        std::cout << "Found " << roads[iVertex].size() << " roads for vertex " << iVertex + 1 << std::endl;
      }

      std::cout << "Event " << iEvent + 1 << " processed in: " << diff << "ms" << std::endl;

      if(currentEvent.getPrimaryVerticesNum() > 1) {

        std::cout << "Vertex processing mean time: " << diff / currentEvent.getPrimaryVerticesNum() << "ms" << std::endl;
      }

      std::cout << std::endl;

      if (createBenchmarkData) {

        IOUtils::writeRoadsReport(correctRoadsOutputStream, duplicateRoadsOutputStream, fakeRoadsOutputStream, roads,
            labelsMap[iEvent]);
      }
*/
    } catch (std::exception& e) {

      std::cout << e.what() << std::endl;
    }

  }

#if defined GPU_PROFILING_MODE
  Utils::Host::gpuStopProfiler();
#endif

  std::cout << std::endl;
  std::cout << "Avg time: " << totalTime / verticesNum << "ms" << std::endl;
  std::cout << "Min time: " << minTime << "ms" << std::endl;
  std::cout << "Max time: " << maxTime << "ms" << std::endl;

  return 0;
}

