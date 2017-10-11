//
// Created by Pharrell_WANG on 11/10/2017.
//

#ifndef HTM162_TAPPENCODER_APHATFPHA_H
#define HTM162_TAPPENCODER_APHATFPHA_H

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

namespace DMM1TimeCost {
  extern double g_dmm1TimeCost;
}

namespace TFGlobalVars {
  extern std::vector<Tensor> OutputsOfFirstBatchSize08;
  extern std::vector<Tensor> OutputsOfSeconBatchSize08;
  extern std::map<int, std::map<int, int> > MapPositionToIndicesSize08;
  extern Tensor FirstBatchOfIndicesSize08;
  extern Tensor SeconBatchOfIndicesSize08;
  extern Tensor FirstBatchOfScoresSize08;
  extern Tensor SeconBatchOfScoresSize08;

  extern std::vector<Tensor> OutputsOfFirstBatchSize16;
  extern std::vector<Tensor> OutputsOfSeconBatchSize16;
  extern std::map<int, std::map<int, int> > MapPositionToIndicesSize16;
  extern Tensor FirstBatchOfIndicesSize16;
  extern Tensor SeconBatchOfIndicesSize16;
  extern Tensor FirstBatchOfScoresSize16;
  extern Tensor SeconBatchOfScoresSize16;

  extern std::vector<Tensor> OutputsOfFirstBatchSize32;
  extern std::vector<Tensor> OutputsOfSeconBatchSize32;
  extern std::map<int, std::map<int, int> > MapPositionToIndicesSize32;
  extern Tensor FirstBatchOfIndicesSize32;
  extern Tensor SeconBatchOfIndicesSize32;
  extern Tensor FirstBatchOfScoresSize32;
  extern Tensor SeconBatchOfScoresSize32;
}


#endif //HTM162_TAPPENCODER_APHATFPHA_H
