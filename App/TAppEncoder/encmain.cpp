/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2016, ITU/ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
 *    be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

/** \file     encmain.cpp
    \brief    Encoder application main
*/

#include <time.h>
#include <iostream>
#include "TAppEncTop.h"
#include "../../Lib/TLibCommon/program_options_lite.h"
#if ENABLE_RESNET

#define GetCurrentDir getcwd

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

// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
Status ReadLabelsFile(string file_name, std::vector<string> *result,
                      size_t *found_label_count) {
  std::ifstream file(file_name);
  if (!file) {
    return tensorflow::errors::NotFound("Labels file ", file_name,
                                        " not found.");
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  *found_label_count = result->size();
  const int padding = 16;
  while (result->size() % padding) {
    result->emplace_back();
  }
  return Status::OK();
}

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(string graph_file_name,
                 std::unique_ptr<tensorflow::Session> *session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
    ReadBinaryProto(tensorflow::Env::Default(), graph_file_name,
                    &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);

  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

// Analyzes the output of the Inception graph to retrieve the highest scores and
// their positions in the tensor, which correspond to categories.
Status GetTopLabels(const std::vector<Tensor> &outputs, int how_many_labels,
                    Tensor *indices, Tensor *scores) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string output_name = "top_k";
  TopK(root.WithOpName(output_name), outputs[0], how_many_labels);
  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensors.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
    tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  // The TopK node returns two outputs, the scores and their original indices,
  // so we have to append :0 and :1 to specify them both.
  std::vector<Tensor> out_tensors;
  TF_RETURN_IF_ERROR(
    session->Run({}, {output_name + ":0", output_name + ":1"},
                 {}, &out_tensors));
  *scores = out_tensors[0];
  *indices = out_tensors[1];
  return Status::OK();
}

// Given the output of a model run, and the name of a file containing the labels
// this prints out the top five highest-scoring values.
Status PrintTopLabels(const std::vector<Tensor> &outputs,
                      string labels_file_name) {

  std::vector<string> labels;
  size_t label_count;

  Status read_labels_status =
    ReadLabelsFile(labels_file_name, &labels, &label_count);
  if (!read_labels_status.ok()) {
    return read_labels_status;
  }

  const int how_many_labels = std::min(16, static_cast<int>(label_count));
  Tensor indices;
  Tensor scores;

  TF_RETURN_IF_ERROR(
    GetTopLabels(outputs, how_many_labels, &indices, &scores));
  tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();
  tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
  for (int pos = 0; pos < how_many_labels; ++pos) {
    const int label_index = indices_flat(pos);
    const float score = scores_flat(pos);
    LOG(INFO) << labels[label_index] << " : "
              << score;
  }
  LOG(INFO) << "";
  return Status::OK();
}

#endif

//! \ingroup TAppEncoder
//! \{

#include "../../Lib/TLibCommon/Debug.h"

// ====================================================================================================================
// Main function
// ====================================================================================================================

int main(int argc, char* argv[])
{
#if ENABLE_RESNET

  // path for the first graph
  string homeDir= getenv("HOME");
  string secPart = "/frozen_graphs/frozen_resnet_for_fdc_blk08x08_133049.pb";
  string nameOfGraphOne = homeDir + secPart;
  string graph = nameOfGraphOne;
  // end first graph

//  string graph_2 =
//    "/Users/Pharrell_WANG/resnet_logs_bak/size_16_log/resnet/graphs/frozen_resnet_for_fdc_blk16x16_304857.pb";

  // path for label text file
  string secondPartLabelFile = "/labels/labels_for_fdc_32_classes.txt";
  string labels = homeDir + secondPartLabelFile;
  // path for label file end

//    "/Users/Pharrell_WANG/labels/labels_for_fdc_32_classes.txt";
  string input_layer = "input";
  string output_layer = "logits/fdc_output_node";
  string root_dir = "";

  // First we load and initialize the model.
  std::unique_ptr<tensorflow::Session> session;
  string graph_path = tensorflow::io::JoinPath(root_dir, graph);
  Status load_graph_status = LoadGraph(graph_path, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }

  // First we load and initialize the model.
  std::unique_ptr<tensorflow::Session> session_2;
//  string graph_path_2 = tensorflow::io::JoinPath(root_dir, graph_2);
//  Status load_graph_status_2 = LoadGraph(graph_path_2, &session_2);
//  if (!load_graph_status_2.ok()) {
//        LOG(ERROR) << load_graph_status;
//    return -1;
//  }

  tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT,
                                  tensorflow::TensorShape({1, 8, 8, 1}));
  // input_tensor_mapped is
  // 1. an interface to the data of ``input_tensor``
  // 1. It is used to copy data into the ``input_tensor``
  auto input_tensor_mapped = input_tensor.tensor<float, 4>();

  // Assign block width
  int BLOCK_WIDTH = 8;

  // set values and copy to ``input_tensor`` using for loop

  for (int row = 0; row < BLOCK_WIDTH; ++row)
    for (int col = 0; col < BLOCK_WIDTH; ++col)
      input_tensor_mapped(0, row, col,
                          0) = 3.0; // this is where we get the pixels

//  tensorflow::Tensor input_tensor_2(tensorflow::DT_FLOAT,
//                                    tensorflow::TensorShape({1, 16, 16, 1}));
//  auto input_tensor_mapped_2 = input_tensor_2.tensor<float, 4>();

  // Assign block width
//  int BLOCK_WIDTH_2 = 16;
  //************************ a few testing data start
  // this data should be mode 25
//  std::string data = "138,138,133,128,122,122,122,117,117,117,117,117,117,117,117,117,138,138,133,128,122,122,122,117,117,117,117,117,117,117,117,117,138,138,133,122,122,122,117,117,117,117,117,117,117,117,117,117,138,138,133,122,122,122,117,117,117,117,117,117,117,117,117,117,138,138,133,122,122,122,117,117,117,117,117,117,117,117,117,117,138,138,133,122,122,122,117,117,117,117,117,117,117,117,117,117,138,138,133,122,122,122,117,117,117,117,117,117,117,117,117,117,138,133,133,128,122,117,117,117,117,117,117,117,117,117,117,117,138,133,133,128,122,117,117,117,117,117,117,117,117,117,117,117,133,133,133,133,128,117,117,117,117,117,117,117,117,117,117,117,133,133,133,133,128,117,117,117,117,117,117,117,117,117,117,117,133,133,133,133,128,117,117,117,117,117,117,117,117,117,117,117,133,133,133,133,128,117,117,117,117,117,117,117,117,117,117,117,133,133,133,133,128,117,117,117,117,117,117,117,117,117,117,117,133,133,133,133,128,117,117,117,117,117,117,117,117,117,117,117,133,133,133,133,128,117,117,117,117,117,117,117,117,117,117,117\n";

  // should be mode 0
//  std::string data = "90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,85,85,85,85,90,90,90,90,90,90,90,90,90,90,90,85,85,85,85,85,90,90,90,90,90,90,90,90,90,90,90,85,85,85,85,85,90,90,90,90,90,90,90,90,90,90,90,85,85,85,85,85,90,90,90,90,90,90,90,90,90,90,90,85,85,85,85,85,90,90,90,90,90,90,90,90,90,90,85,85,85,85,85,85,90,90,90,90,90,90,90,90,90,85,85,85,85,85,85,85,90,90,90,90,90,90,90,90,90,85,85,85,85,85,85,85,90,90,90,90,90,90,90,90,85,85,85,85,85,85,85,85,90,90,90,90,90,90,90,90,85,85,85,85,85,85,85,85,90,90,90,90,90,90,90,90,85,85,85,85,85,85,85,85,90,90,90,90,90,90,90,90,85,85,85,85,85,85,85,85,90,90,90,90,90,90,90,90,85,85,85,85,85,85,85,85\n";
  //# 27, mode 29
//  std::string data = "170,170,170,170,170,170,170,170,170,170,170,170,170,175,175,175,170,170,170,170,170,170,170,170,170,170,170,170,170,175,175,175,170,170,170,170,170,170,170,170,170,170,170,170,170,175,175,175,170,170,170,170,170,170,170,170,170,170,170,170,170,175,175,175,170,170,170,170,170,170,170,170,170,170,170,170,170,175,175,175,170,170,170,170,170,170,170,170,170,170,170,170,170,175,175,175,170,170,170,170,170,170,170,170,170,170,170,170,170,175,175,175,170,170,170,170,170,170,170,170,170,170,170,170,170,175,175,175,170,170,170,170,170,170,170,170,170,170,170,170,170,175,175,175,170,170,170,170,170,170,170,170,170,170,170,170,170,175,181,181,170,170,170,170,170,170,170,170,170,170,170,170,170,181,181,181,170,170,170,170,170,170,170,170,170,170,170,170,175,181,181,181,170,170,170,170,170,170,170,170,170,170,170,170,181,181,181,181,170,170,170,170,170,170,170,170,170,170,170,170,181,181,181,181,170,170,170,170,170,170,170,170,170,170,170,175,181,181,181,181,170,170,170,170,170,170,170,170,170,170,170,175,181,181,181,181\n";
  //# 7 mode 9
//  std::string data = "14,14,14,14,14,14,14,14,14,14,14,14,14,16,16,16,14,14,14,14,14,14,14,14,14,14,14,14,14,16,16,16,14,14,14,14,16,14,14,14,14,14,14,14,14,16,16,16,14,14,14,14,16,16,14,14,14,14,14,14,16,16,18,18,14,14,14,14,14,14,14,14,14,14,14,16,18,18,21,21,16,14,14,14,14,14,50,50,50,50,50,50,50,50,50,50,52,50,50,50,50,50,52,53,53,53,57,58,59,59,58,53,55,55,55,53,53,53,59,59,59,59,59,59,59,59,59,59,57,57,57,57,57,59,59,59,59,59,59,59,59,59,59,59,58,58,58,58,58,59,59,59,59,59,59,59,59,59,59,59,58,58,58,58,58,59,59,59,59,59,59,59,59,59,59,59,59,59,58,58,58,59,59,59,59,59,59,59,59,59,59,59,62,62,62,62,63,62,59,59,59,59,59,59,59,59,59,59,64,64,64,64,63,63,62,59,59,59,59,59,59,59,59,59,64,64,64,64,63,63,62,59,59,59,59,59,59,59,59,57,66,64,64,64,64,63,62,59,59,59,59,59,59,59,57,57\n";
//************************ a few testing data end

//  strtk::token_grid grid(data, data.size(), ",");

//  strtk::token_grid::row_type r = grid.row(0);
  // set values and copy to ``input_tensor`` using for loop
//  for (int row = 0; row < BLOCK_WIDTH_2; ++row)
//    for (int col = 0; col < BLOCK_WIDTH_2; ++col)
//      input_tensor_mapped_2(0, row, col,
//                            0) = r.get<int>(size_t(row * BLOCK_WIDTH_2 + col)); // this is where we get the pixels

  // Actually run the image through the model.
  std::vector<Tensor> outputs;
//  std::vector<Tensor> outputs_2;

  Status run_status = session->Run({{input_layer, input_tensor}},
                                   {output_layer}, {}, &outputs);

//  Status run_status_2 = session_2->Run({{input_layer, input_tensor_2}},
//                                       {output_layer}, {}, &outputs_2);
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
  }

  // Do something interesting with the results we've generated.
  Status print_status = PrintTopLabels(outputs, labels);

//  Status print_status_2 = PrintTopLabels(outputs_2, labels);

  if (!print_status.ok()) {
    LOG(ERROR) << "Running print failed: " << print_status;
    return -1;
  }
#endif

  TAppEncTop  cTAppEncTop;

  // print information
  fprintf( stdout, "\n" );
#if NH_MV
  fprintf( stdout, "3D-HTM Software: Encoder Version [%s] based on HM Version [%s]", NV_VERSION, HM_VERSION );  
#else
  fprintf( stdout, "HM software: Encoder Version [%s] (including RExt)", NV_VERSION );
#endif
  fprintf( stdout, NVM_ONOS );
  fprintf( stdout, NVM_COMPILEDBY );
  fprintf( stdout, NVM_BITS );
  fprintf( stdout, "\n\n" );

  // create application encoder class
  cTAppEncTop.create();

  // parse configuration
  try
  {
    if(!cTAppEncTop.parseCfg( argc, argv ))
    {
      cTAppEncTop.destroy();
#if ENVIRONMENT_VARIABLE_DEBUG_AND_TEST
      EnvVar::printEnvVar();
#endif
      return 1;
    }
  }
  catch (df::program_options_lite::ParseFailure &e)
  {
    std::cerr << "Error parsing option \""<< e.arg <<"\" with argument \""<< e.val <<"\"." << std::endl;
    return 1;
  }

#if PRINT_MACRO_VALUES
  printMacroSettings();
#endif

#if ENVIRONMENT_VARIABLE_DEBUG_AND_TEST
  EnvVar::printEnvVarInUse();
#endif

  // starting time
  Double dResult;
  clock_t lBefore = clock();

  // call encoding function
  cTAppEncTop.encode();

  // ending time
  dResult = (Double)(clock()-lBefore) / CLOCKS_PER_SEC;
  printf("\n Total Time: %12.3f sec.\n", dResult);

  // destroy application encoder class
  cTAppEncTop.destroy();

  return 0;
}

//! \}
