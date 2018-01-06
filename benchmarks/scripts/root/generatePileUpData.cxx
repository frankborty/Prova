/// \file generatePileUpData.cxx
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
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {
constexpr int MonteCarloOffset { 100000 };
constexpr int PrimaryVertexId { -1 };
constexpr int EventsFileColumnsNum { 10 };
constexpr int EventsFileMonteCarloIndex { 8 };
constexpr int LabelsFileColumnsNum { 6 };
}

int roundUp(const int numToRound, const int multiple);
void mergeEvents(const int pileUp, const std::string& inputFolder, const std::string& outputFolder);
void mergeLabels(const int pileUp, const std::string& inputFolder, const std::string& outputFolder);
void generatePileUpData(const int pileUp, const std::string& inputFolder);
void generatePileUpData(const int pileUp, const std::string& inputFolder, const std::string& outputFolder);

void mergeEvents(const int pileUp, const std::string& inputFolder, const std::string& outputFolder)
{

  std::ifstream inputStream;
  std::ofstream outputStream;
  std::string inputLine;
  int layerId, monteCarloId, verticesNum = 0;
  std::vector<std::string> clusterLines;

  inputStream.open(inputFolder + "data.txt");
  outputStream.open(outputFolder + "merged_data.txt");

  while (std::getline(inputStream, inputLine)) {

    std::istringstream inputStringStream(inputLine);

    if (inputStringStream >> layerId) {

      if (layerId == PrimaryVertexId) {

        if (verticesNum == pileUp) {

          const int linesNum = clusterLines.size();
          for (int iLine = 0; iLine < linesNum; ++iLine) {

            outputStream << clusterLines[iLine] << std::endl;
          }

          verticesNum = 0;
          clusterLines.clear();
        }

        outputStream << inputLine << std::endl;
        ++verticesNum;

      } else {

        std::array<std::string, EventsFileColumnsNum> lineData;

        lineData[0] = std::to_string(layerId);

        for (int iValue = 1; iValue < EventsFileColumnsNum; ++iValue) {

          inputStringStream >> lineData[iValue];
        }

        monteCarloId = std::stoi(lineData[EventsFileMonteCarloIndex]);

        if (verticesNum > 1) {

          lineData[EventsFileMonteCarloIndex] = std::to_string(monteCarloId + ((verticesNum - 1) * MonteCarloOffset));

          std::ostringstream outputStringStream;
          for (int iValue = 0; iValue < EventsFileColumnsNum; ++iValue) {

            if (iValue != 0) {

              outputStringStream << "\t";
            }

            outputStringStream << lineData[iValue];
          }

          clusterLines.push_back(outputStringStream.str());

        } else {

          clusterLines.push_back(inputLine);
        }

      }
    }
  }

  const int linesNum = clusterLines.size();
  for (int iLine = 0; iLine < linesNum; ++iLine) {

    outputStream << clusterLines[iLine] << std::endl;
  }

  std::cout << "Generated " << outputFolder << "merged_data.txt with pile-up " << pileUp << std::endl;
}

void mergeLabels(const int pileUp, const std::string& inputFolder, const std::string& outputFolder)
{

  std::ifstream inputStream;
  std::ofstream outputStream;
  std::string inputLine;
  int monteCarloId, verticesNum = 0;
  std::vector<std::string> labelLines;

  inputStream.open(inputFolder + "labels.txt");
  outputStream.open(outputFolder + "merged_labels.txt");

  while (std::getline(inputStream, inputLine)) {

    std::istringstream inputStringStream(inputLine);

    if (inputStringStream >> monteCarloId) {

      if (monteCarloId == PrimaryVertexId) {

        if (verticesNum == pileUp) {

          outputStream << inputLine << std::endl;

          const int linesNum = labelLines.size();
          for (int iLine = 0; iLine < linesNum; ++iLine) {

            outputStream << labelLines[iLine] << std::endl;
          }

          verticesNum = 0;
          labelLines.clear();
        }

        ++verticesNum;

      } else {

        std::array<std::string, EventsFileColumnsNum> lineData;

        lineData[0] = std::to_string(monteCarloId);

        for (int iValue = 1; iValue < LabelsFileColumnsNum; ++iValue) {

          inputStringStream >> lineData[iValue];
        }

        if (verticesNum > 1) {

          lineData[0] = std::to_string(monteCarloId + ((verticesNum - 1) * MonteCarloOffset));

          std::ostringstream outputStringStream;
          for (int iValue = 0; iValue < LabelsFileColumnsNum; ++iValue) {

            if (iValue != 0) {

              outputStringStream << "\t";
            }

            outputStringStream << lineData[iValue];
          }

          labelLines.push_back(outputStringStream.str());

        } else {

          labelLines.push_back(inputLine);
        }
      }
    }
  }

  outputStream << PrimaryVertexId << std::endl;
  const int linesNum = labelLines.size();
  for (int iLine = 0; iLine < linesNum; ++iLine) {

    outputStream << labelLines[iLine] << std::endl;
  }

  std::cout << "Generated " << outputFolder << "merged_labels.txt with pile-up " << pileUp << std::endl;
}

void generatePileUpData(const int pileUp, const std::string& inputFolder)
{

  generatePileUpData(pileUp, inputFolder, inputFolder);
}

void generatePileUpData(const int pileUp, const std::string& inputFolder, const std::string& outputFolder)
{

  mergeEvents(pileUp, inputFolder, outputFolder);
  mergeLabels(pileUp, inputFolder, outputFolder);
}

int main(int argc, char** argv)
{

  int pileUp;
  std::string inputFolder { "" };
  std::string outputFolder { "" };

  if (argv[1] != NULL) {

    pileUp = std::stoi(argv[1]);

  } else {

    std::cout << "generatePileUpData.cxx pileUp inputFolder [outputFolder=inputFolder]" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (argv[2] != NULL) {

    inputFolder = std::string(argv[2]);
  }

  if (argv[3] != NULL) {

    outputFolder = std::string(argv[3]);

  } else if (argv[2] != NULL) {

    outputFolder = inputFolder;
  }

  generatePileUpData(pileUp, inputFolder, outputFolder);
  return 0;
}
