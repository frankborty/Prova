/// \file plotMemoryOccupancyBenchmark.cxx
/// \brief 
///
/// \author Iacopo Colonnelli, Politecnico di Torino

/***************************************************************************
 *  Copyright (C) 2017  Iacopo Colonnelli
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ***************************************************************************/

#include <array>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "TCanvas.h"
#include "TGraph.h"
#include "TROOT.h"

namespace {
constexpr int BinNumber { 100 };
constexpr int LayersNumber { 7 };
constexpr int TrackletsNumber { 6 };
constexpr int CellsNumber { 5 };
constexpr int RoadsNumber { 1 };
constexpr int DataTypes { 6 };
constexpr int ClustersDataType { 0 };
constexpr int PredictiveTrackletAllocationDataType { 1 };
constexpr int TrackletsDataType { 2 };
constexpr int PredictiveCellsAllocationDataType { 3 };
constexpr int CellsDataType { 4 };
constexpr int RoadsDataType { 5 };
constexpr std::array<int, DataTypes> LineDataNum { LayersNumber, TrackletsNumber, TrackletsNumber, CellsNumber,
    CellsNumber, RoadsNumber };
}

std::vector<std::array<std::vector<int>, DataTypes>> loadData(const std::string& fileName)
{
  std::ifstream inputStream;
  std::string line;
  std::vector<std::array<std::vector<int>, DataTypes>> dataReport;
  std::array<std::vector<int>, DataTypes> eventReport;
  int lineType = 0, currentDataValue;

  inputStream.open(fileName);

  while (std::getline(inputStream, line)) {

    std::istringstream inputStringStream(line);
    std::vector<int> currentLineData;
    const int currentLineDataNum = LineDataNum[lineType];
    currentLineData.reserve(currentLineDataNum);

    for (int iValue = 0; iValue < currentLineDataNum; ++iValue) {

      inputStringStream >> currentDataValue;
      currentLineData.emplace_back(currentDataValue);
    }

    eventReport[lineType] = currentLineData;
    ++lineType;

    if (lineType == DataTypes) {

      lineType = 0;
      dataReport.emplace_back(eventReport);
    }
  }

  return dataReport;
}

void plotHistogram(TH1F& histogram, const std::string& outputFileName)
{
  TCanvas graphCanvas { };
  graphCanvas.SetGrid();
  graphCanvas.SetBottomMargin(0.22);

  histogram.GetXaxis()->SetTitleSize(0.05);
  histogram.GetYaxis()->SetTitleSize(0.05);

  gStyle->SetOptStat(1100);

  histogram.Draw();
  graphCanvas.Update();

  double maximum { gPad->GetUymax() };
  double mean = { histogram.GetMean() };
  TLine meanLine{ mean, .0, mean, maximum };
  meanLine.SetLineWidth(2);
  meanLine.SetLineColor(kRed);
  meanLine.Draw();

  graphCanvas.Print(outputFileName.c_str());
}

void plotHistogramWithoutMean(TH1F& histogram, const std::string& outputFileName)
{
  TCanvas graphCanvas { };
  graphCanvas.SetGrid();
  graphCanvas.SetLeftMargin(0.15);
  graphCanvas.SetBottomMargin(0.22);

  histogram.GetXaxis()->SetTitleSize(0.05);
  histogram.GetYaxis()->SetTitleSize(0.05);

  gStyle->SetOptStat(1100);


  histogram.Draw();

  graphCanvas.Print(outputFileName.c_str());
}

void fillMemoryOccupancyHistogram(TH1F& histogram, std::vector<std::array<std::vector<int>, DataTypes>>& dataReport,
    const int dataType, const int layerIndex)
{

  const int numEvents = dataReport.size();

  for (int iEvent = 0; iEvent < numEvents; ++iEvent) {

    const int currentValue = dataReport[iEvent][dataType][layerIndex];
    const int complexity = LayersNumber + 1 - LineDataNum[dataType];
    unsigned long long clustersProduct = 1;

    for (int iClustersLayer = layerIndex; iClustersLayer < layerIndex + complexity; ++iClustersLayer) {

      clustersProduct *= dataReport[iEvent][ClustersDataType][iClustersLayer];
    }

    histogram.Fill(static_cast<double>(currentValue) / clustersProduct);
  }
}

void fillFillFactorHistogram(TH1F& histogram, std::vector<std::array<std::vector<int>, DataTypes>>& dataReport,
    const int dataType)
{

  const int numEvents = dataReport.size();

  for (int iEvent = 0; iEvent < numEvents; ++iEvent) {

    for (int iLayer = 0; iLayer < LineDataNum[dataType]; ++iLayer) {

      const int preallocatedSizeValue = dataReport[iEvent][dataType - 1][iLayer];
      const int actualSizeValue = dataReport[iEvent][dataType][iLayer];

      histogram.Fill(100.0 * actualSizeValue / preallocatedSizeValue);
    }
  }
}

void plotMemoryOccupancyBenchmark(const std::string& inputFolder, const std::string& outputFolder)
{
  std::string graphPrefix = "plot-memory-occupancy-benchmark";
  std::vector<std::array<std::vector<int>, DataTypes>> dataReport = loadData(inputFolder + "MemoryOccupancy.txt");

  double binSize;
  std::array<double, BinNumber + 1> binsEdges;
  std::string histogramId, histogramTitle, outputFileName;

  /// Tracklets Histograms
  binSize = 1.3e-03 / BinNumber;
  for (int iBin = 0; iBin <= BinNumber; ++iBin) {

    binsEdges[iBin] = iBin * binSize;
  }

  for (int iLayer = 0; iLayer < TrackletsNumber; ++iLayer) {

    histogramId = std::string(graphPrefix + ".tracklets-graph-" + std::to_string(iLayer));
    histogramTitle = std::string("Layer " + std::to_string(iLayer + 1) + " Tracklets Memory Occupancy");
    outputFileName = std::string(outputFolder + "Layer" + std::to_string(iLayer + 1) + "TrackletsHistogram.pdf");

    TH1F trackletsMemoryOccupancyHistogram(histogramId.c_str(), histogramTitle.c_str(), BinNumber, binsEdges.data());
    fillMemoryOccupancyHistogram(trackletsMemoryOccupancyHistogram, dataReport, TrackletsDataType, iLayer);

    trackletsMemoryOccupancyHistogram.GetXaxis()->SetTitle("#frac{N_{T}_{1}}{(#alpha_{i} + #alpha_{i + 1}) #upoint N^{2}}");
    trackletsMemoryOccupancyHistogram.GetXaxis()->SetTitleOffset(1.6);
    trackletsMemoryOccupancyHistogram.GetYaxis()->SetTitle("#samples");
    trackletsMemoryOccupancyHistogram.SetFillColorAlpha(kBlue, .2);

    plotHistogram(trackletsMemoryOccupancyHistogram, outputFileName);
  }

  /// Cells Histograms
  binSize = 2.5e-08 / BinNumber;
  for (int iBin = 0; iBin <= BinNumber; ++iBin) {

    binsEdges[iBin] = iBin * binSize;
  }

  for (int iLayer = 0; iLayer < CellsNumber; ++iLayer) {

    histogramId = std::string(graphPrefix + ".cells-graph-" + std::to_string(iLayer));
    histogramTitle = std::string("Layer " + std::to_string(iLayer + 1) + " Cells Memory Occupancy");
    outputFileName = std::string(outputFolder + "Layer" + std::to_string(iLayer + 1) + "CellsHistogram.pdf");

    TH1F cellsMemoryOccupancyHistogram(histogramId.c_str(), histogramTitle.c_str(), BinNumber, binsEdges.data());
    fillMemoryOccupancyHistogram(cellsMemoryOccupancyHistogram, dataReport, CellsDataType, iLayer);

    cellsMemoryOccupancyHistogram.GetXaxis()->SetTitle("#frac{N_{C}_{1}}{(#alpha_{i} + #alpha_{i + 1} + #alpha_{i + 2}) #upoint N^{3}}");
    cellsMemoryOccupancyHistogram.GetXaxis()->SetTitleOffset(1.6);
    cellsMemoryOccupancyHistogram.GetYaxis()->SetTitle("#samples");
    cellsMemoryOccupancyHistogram.SetFillColorAlpha(kBlue, .2);

    plotHistogram(cellsMemoryOccupancyHistogram, outputFileName);
  }

  /// Roads Histogram
  binSize = 5e-14 / BinNumber;
  for (int iBin = 0; iBin <= BinNumber; ++iBin) {

    binsEdges[iBin] = iBin * binSize;
  }

  histogramId = std::string(graphPrefix + ".roads-graph");
  histogramTitle = std::string("Roads Memory Occupancy");
  outputFileName = std::string(outputFolder + "RoadsHistogram.pdf");

  TH1F roadsMemoryOccupancyHistogram(histogramId.c_str(), histogramTitle.c_str(), BinNumber, binsEdges.data());

  roadsMemoryOccupancyHistogram.GetXaxis()->SetTitle("Actual/theoretical roads occupancy");
  roadsMemoryOccupancyHistogram.GetXaxis()->SetTitleOffset(1.4);
  roadsMemoryOccupancyHistogram.GetYaxis()->SetTitle("#samples");
  roadsMemoryOccupancyHistogram.SetFillColorAlpha(kBlue, .2);

  fillMemoryOccupancyHistogram(roadsMemoryOccupancyHistogram, dataReport, RoadsDataType, 0);
  plotHistogramWithoutMean(roadsMemoryOccupancyHistogram, outputFileName);

  // Tracklets Fill Factor Histograms
  binSize = 100.f / BinNumber;
  for (int iBin = 0; iBin <= BinNumber; ++iBin) {

    binsEdges[iBin] = iBin * binSize;
  }

  histogramId = std::string(graphPrefix + ".tracklets-fill-factor-graph");
  histogramTitle = std::string("Tracklets Fill Factor Distribution");
  outputFileName = std::string(outputFolder + "TrackletsFillFactor.pdf");

  TH1F trackletsFillFactorHistogram(histogramId.c_str(), histogramTitle.c_str(), BinNumber, binsEdges.data());
  fillFillFactorHistogram(trackletsFillFactorHistogram, dataReport, TrackletsDataType);

  trackletsFillFactorHistogram.GetXaxis()->SetTitle("Used / reserved memory (%)");
  trackletsFillFactorHistogram.GetXaxis()->SetTitleOffset(1.4);
  trackletsFillFactorHistogram.GetYaxis()->SetTitle("# samples");
  trackletsFillFactorHistogram.SetFillColorAlpha(kBlue, .2);

  plotHistogramWithoutMean(trackletsFillFactorHistogram, outputFileName);

  // Cells Fill Factor Histograms
  binSize = 100.f / BinNumber;
  for (int iBin = 0; iBin <= BinNumber; ++iBin) {

    binsEdges[iBin] = iBin * binSize;
  }

  histogramId = std::string(graphPrefix + ".cells-fill-factor-graph");
  histogramTitle = std::string("Cells Fill Factor Distribution");
  outputFileName = std::string(outputFolder + "CellsFillFactor.pdf");

  TH1F cellsFillFactorHistogram(histogramId.c_str(), histogramTitle.c_str(), BinNumber, binsEdges.data());
  fillFillFactorHistogram(cellsFillFactorHistogram, dataReport, CellsDataType);

  cellsFillFactorHistogram.GetXaxis()->SetTitle("Used / reserved memory (%)");
  cellsFillFactorHistogram.GetXaxis()->SetTitleOffset(1.4);
  cellsFillFactorHistogram.GetYaxis()->SetTitle("# samples");
  cellsFillFactorHistogram.SetFillColorAlpha(kBlue, .2);

  plotHistogramWithoutMean(cellsFillFactorHistogram, outputFileName);
}

int main(int argc, char** argv)
{
  std::string inputFolder { "" };
  std::string outputFolder { "" };

  if (argv[1] != NULL) {

    inputFolder = std::string(argv[1]);
  }

  if (argv[2] != NULL) {

    outputFolder = std::string(argv[2]);
  }

  plotMemoryOccupancyBenchmark(inputFolder, outputFolder);
}

