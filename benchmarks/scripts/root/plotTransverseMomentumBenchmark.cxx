/// \file plotTransverseMomentumBenchmark.cxx
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
#include <cmath>
#include <fstream>
#include <string>

#include "TCanvas.h"
#include "TGraph.h"
#include "TROOT.h"

namespace {
constexpr int BinNumber { 100 };
constexpr float MaxTransverseMomentum { 20 };
constexpr float MinTransverseMomentum { 0.01 };
constexpr int EventLabelsSeparator { -1 };
constexpr int PionCode { 211 };
}

void loadData(const std::string& fileName, TH1F& histogram)
{
  std::ifstream inputStream;
  std::string line;
  inputStream.open(fileName);
  int monteCarloId, pdgCode, numberOfClusters;
  float transverseMomentum, phiCoordinate, pseudorapidity;

  while (std::getline(inputStream, line)) {

    std::istringstream inputStringStream(line);

    if (inputStringStream >> monteCarloId) {

      if (monteCarloId == EventLabelsSeparator) {

        continue;

      } else {

        if (inputStringStream >> transverseMomentum >> phiCoordinate >> pseudorapidity >> pdgCode >> numberOfClusters) {

          if (std::abs(pdgCode) == PionCode && numberOfClusters == 7) {

            histogram.Fill(transverseMomentum);
          }
        }
      }
    }
  }
}

void plotHistogramsRatio(TH1F& numeratorHistogram, TH1F& denominatorHistogram,
    std::array<float, BinNumber + 1>& binsEdges, const std::string& outputFileName, const char *histogramTitle,
    const char *yAxisTitle)
{
  TCanvas graphCanvas { };
  graphCanvas.SetGrid();
  graphCanvas.SetLogx();
  graphCanvas.SetLeftMargin(0.15);
  graphCanvas.SetBottomMargin(0.15);

  gStyle->SetOptStat(1100);

  gStyle->SetOptStat(1100);

  TH1F histogramsRatio("plot-transverse-momentum-benchmark.histograms-ratio", histogramTitle, BinNumber,
      binsEdges.data());

  histogramsRatio.GetXaxis()->SetTitle("Transverse momentum #rho_{T} (Gev/c)");
  histogramsRatio.GetXaxis()->SetTitleSize(0.05);
  histogramsRatio.GetXaxis()->SetTitleOffset(1.4);
  histogramsRatio.GetYaxis()->SetTitle(yAxisTitle);
  histogramsRatio.GetYaxis()->SetTitleSize(0.05);
  histogramsRatio.SetFillColorAlpha(kBlue, .2);

  histogramsRatio.Divide(&numeratorHistogram, &denominatorHistogram);
  histogramsRatio.Draw();

  graphCanvas.Print(outputFileName.c_str());
}

void plotTransverseMomentumBenchmark(const std::string& inputFolder, const std::string& outputFolder)
{
  std::string histogramPrefix = "plot-transverse-momentum-benchmark";

  float binSize = std::log(MaxTransverseMomentum / MinTransverseMomentum) / BinNumber;
  std::array<float, BinNumber + 1> binsEdges;

  for (int iBin = 0; iBin <= BinNumber; ++iBin) {

    binsEdges[iBin] = MinTransverseMomentum * std::exp(iBin * binSize);
  }

  TH1F generatedHistogram((histogramPrefix + ".generated-histogram").c_str(), "Generated Histogram", BinNumber,
      binsEdges.data());
  loadData(inputFolder + "merged_labels.txt", generatedHistogram);

  TH1F correctHistogram((histogramPrefix + ".correct-histogram").c_str(), "Correct Histogram", BinNumber,
      binsEdges.data());
  loadData(inputFolder + "CorrectRoads.txt", correctHistogram);

  plotHistogramsRatio(correctHistogram, generatedHistogram, binsEdges, outputFolder + "CorrectRoadsBenchmark.pdf",
      "Correct Roads Histogram", "Efficiency");

  TH1F duplicatedHistogram((histogramPrefix + ".duplicate-histogram").c_str(), "Duplicate Histogram", BinNumber,
      binsEdges.data());
  loadData(inputFolder + "DuplicateRoads.txt", duplicatedHistogram);

  plotHistogramsRatio(duplicatedHistogram, generatedHistogram, binsEdges, outputFolder + "DuplicatedRoadsBenchmark.pdf",
      "Duplicated Roads Histogram", "Duplicated to generated ratio");

  TH1F fakeHistogram((histogramPrefix + ".fake-histogram").c_str(), "Fake Histogram", BinNumber, binsEdges.data());
  loadData(inputFolder + "FakeRoads.txt", fakeHistogram);

  plotHistogramsRatio(fakeHistogram, generatedHistogram, binsEdges, outputFolder + "FakeRoadsBenchmark.pdf",
      "Fake Roads Histogram", "Fake to generated ratio");

  TH1F duplicatedAndCorrectHistogram((histogramPrefix + ".duplicate-and-correct-histogram").c_str(),
      "Duplicate And Correct Histogram", BinNumber, binsEdges.data());
  duplicatedAndCorrectHistogram.Add(&correctHistogram, &duplicatedHistogram);

  plotHistogramsRatio(duplicatedHistogram, duplicatedAndCorrectHistogram, binsEdges,
      outputFolder + "DuplicatedToNotFakeRoadsBenchmark.pdf", "Duplicated To Not Fake Roads Histogram", "Duplicated to not fake ratio");
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

  plotTransverseMomentumBenchmark(inputFolder, outputFolder);
}
