//
// Created by Pharrell_WANG on 11/10/2017.
//

#include "AphaTfPha.h"

namespace DMM1TimeCost {
  double g_dmm1TimeCost(0);
}

namespace TFGlobalVars {
  std::vector<Tensor> OutputsOfFirstBatchSize08;
  std::vector<Tensor> OutputsOfSeconBatchSize08;
  std::map<int, std::map<int, int> > MapPositionToIndicesSize08;
  Tensor FirstBatchOfIndicesSize08;
  Tensor SeconBatchOfIndicesSize08;
  Tensor FirstBatchOfScoresSize08;
  Tensor SeconBatchOfScoresSize08;

  std::vector<Tensor> OutputsOfFirstBatchSize16;
  std::vector<Tensor> OutputsOfSeconBatchSize16;
  std::map<int, std::map<int, int> > MapPositionToIndicesSize16;
  Tensor FirstBatchOfIndicesSize16;
  Tensor SeconBatchOfIndicesSize16;
  Tensor FirstBatchOfScoresSize16;
  Tensor SeconBatchOfScoresSize16;

  std::vector<Tensor> OutputsOfFirstBatchSize32;
  std::vector<Tensor> OutputsOfSeconBatchSize32;
  std::map<int, std::map<int, int> > MapPositionToIndicesSize32;
  Tensor FirstBatchOfIndicesSize32;
  Tensor SeconBatchOfIndicesSize32;
  Tensor FirstBatchOfScoresSize32;
  Tensor SeconBatchOfScoresSize32;
}