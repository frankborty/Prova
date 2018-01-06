#include <array>
#include <fstream>
#include <sstream>
#include <string>
#include <map>

#include "TMultiGraph.h"
#include "TGraph.h"

namespace {
constexpr int SamplesNumber { 4 };
constexpr std::array<float, SamplesNumber> VerticesNum { { 1, 2, 4, 5 } };
}

void compileData(const std::string &inputFolder, std::map<std::string, std::array<float, SamplesNumber>> &inputDataMap)
{
  std::string line;
  std::istringstream inputStringStream;

  for (int iSample = 0; iSample < SamplesNumber; ++iSample) {

    std::ostringstream verticesNumStr;
    verticesNumStr << std::setprecision(0) << VerticesNum[iSample];
    std::ifstream inputStream;
    inputStream.open(inputFolder + "TimeOccupancyReport" + verticesNumStr.str() + ".txt");
    float meanValue, minValue, maxValue;

    std::getline(inputStream, line);
    inputStringStream = std::istringstream { line };
    inputStringStream >> meanValue >> minValue >> maxValue;
    inputDataMap["TrackletsMean"][iSample] = meanValue;
    inputDataMap["TrackletsMin"][iSample] = minValue;
    inputDataMap["TrackletsMax"][iSample] = maxValue;

    std::getline(inputStream, line);
    inputStringStream = std::istringstream { line };
    inputStringStream >> meanValue >> minValue >> maxValue;
    inputDataMap["CellsMean"][iSample] = meanValue;
    inputDataMap["CellsMin"][iSample] = minValue;
    inputDataMap["CellsMax"][iSample] = maxValue;

    std::getline(inputStream, line);
    inputStringStream = std::istringstream { line };
    inputStringStream >> meanValue >> minValue >> maxValue;
    inputDataMap["TotalMean"][iSample] = meanValue;
    inputDataMap["TotalMin"][iSample] = minValue;
    inputDataMap["TotalMax"][iSample] = maxValue;
  }
}

void plotMultiGraph(const std::string& outputFileName, const std::string& graphTitle,
    const std::array<float, SamplesNumber>& meanData, const std::array<float, SamplesNumber>& minData,
    const std::array<float, SamplesNumber>& maxData)
{
  TCanvas graphCanvas { };
  graphCanvas.SetGrid();
  graphCanvas.SetLeftMargin(0.15);
  graphCanvas.SetBottomMargin(0.15);

  TMultiGraph multiGraph { };
  multiGraph.SetTitle(graphTitle.c_str());

  TGraph meanGraph { SamplesNumber, VerticesNum.data(), meanData.data() };
  meanGraph.SetTitle("Mean Time");
  meanGraph.SetLineColor(kRed);
  meanGraph.SetMarkerStyle(20);
  meanGraph.SetMarkerColor(kRed);
  multiGraph.Add(&meanGraph);

  TGraph minGraph { SamplesNumber, VerticesNum.data(), minData.data() };
  minGraph.SetTitle("Min Time");
  minGraph.SetLineColor(kGreen);
  minGraph.SetMarkerStyle(20);
  minGraph.SetMarkerColor(kGreen);
  multiGraph.Add(&minGraph);

  TGraph maxGraph { SamplesNumber, VerticesNum.data(), maxData.data() };
  maxGraph.SetTitle("Max Time");
  maxGraph.SetLineColor(kBlue);
  maxGraph.SetMarkerStyle(20);
  maxGraph.SetMarkerColor(kBlue);
  multiGraph.Add(&maxGraph);

  multiGraph.Draw("ACP");
  multiGraph.GetXaxis()->SetTitle("#Vertices");
  multiGraph.GetXaxis()->SetTitleSize(0.05);
  multiGraph.GetYaxis()->SetTitle("Computing time (ms)");
  multiGraph.GetYaxis()->SetTitleSize(0.05);
  multiGraph.GetYaxis()->SetTitleOffset(1.5);

  graphCanvas.BuildLegend(0.1, 0.7, 0.3, 0.9);
  graphCanvas.Print(outputFileName.c_str());
}

void plotComputingTimeBenchmark(const std::string& inputFolder, const std::string& outputFolder)
{
  std::map<std::string, std::array<float, SamplesNumber>> inputDataMap;
  inputDataMap.emplace(std::make_pair(std::string("TrackletsMean"), std::array<float, SamplesNumber> { }));
  inputDataMap.emplace(std::make_pair(std::string("TrackletsMin"), std::array<float, SamplesNumber> { }));
  inputDataMap.emplace(std::make_pair(std::string("TrackletsMax"), std::array<float, SamplesNumber> { }));

  inputDataMap.emplace(std::make_pair(std::string("CellsMean"), std::array<float, SamplesNumber> { }));
  inputDataMap.emplace(std::make_pair(std::string("CellsMin"), std::array<float, SamplesNumber> { }));
  inputDataMap.emplace(std::make_pair(std::string("CellsMax"), std::array<float, SamplesNumber> { }));

  inputDataMap.emplace(std::make_pair(std::string("TotalMean"), std::array<float, SamplesNumber> { }));
  inputDataMap.emplace(std::make_pair(std::string("TotalMin"), std::array<float, SamplesNumber> { }));
  inputDataMap.emplace(std::make_pair(std::string("TotalMax"), std::array<float, SamplesNumber> { }));

  compileData(inputFolder, inputDataMap);

  plotMultiGraph(outputFolder + "TrackletsComputingTime.pdf", "Tracklets Computing Time", inputDataMap["TrackletsMean"],
      inputDataMap["TrackletsMin"], inputDataMap["TrackletsMax"]);
  plotMultiGraph(outputFolder + "CellsComputingTime.pdf", "Cells Computing Time", inputDataMap["CellsMean"],
      inputDataMap["CellsMin"], inputDataMap["CellsMax"]);
  plotMultiGraph(outputFolder + "TotalComputingTime.pdf", "Total Computing Time", inputDataMap["TotalMean"],
      inputDataMap["TotalMin"], inputDataMap["TotalMax"]);
}
