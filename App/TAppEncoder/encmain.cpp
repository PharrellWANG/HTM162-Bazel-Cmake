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
#include "program_options_lite.h"
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
\

#endif

//! \ingroup TAppEncoder
//! \{

#include "Debug.h"
#include "TimeCost.h"

// ====================================================================================================================
// Main function
// ====================================================================================================================
double g_dmm1TimeCost(0);
//extern double getDmm1TimeCost();

int main(int argc, char* argv[])
{
#if ENABLE_RESNET
  string homeDir= getenv("HOME");
  ///1st graph
  // path for the first graph
  // 1024x768
  string secPart = "/frozen_graphs/frozen_resnet_for_fdc_blk8x8_batchsize12288_step133049.pb";
  // 1920x1088
//  string secPart = "/frozen_graphs/frozen_resnet_for_fdc_blk8x8_batchsize32640_step133049.pb";
  string nameOfGraphOne = homeDir + secPart;
  string graph = nameOfGraphOne;
  // end first graph

  // path for label text file
  string secondPartLabelFile = "/labels/labels_for_fdc_32_classes.txt";
  string labels = homeDir + secondPartLabelFile;
  // path for label file end

  string input_layer = "input";
  string output_layer = "logits/fdc_output_node";
  string root_dir = "";

  std::unique_ptr<tensorflow::Session> session;024
  string graph_path = tensorflow::io::JoinPath(root_dir, graph);
  Status load_graph_status = LoadGraph(graph_path, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }
  ///2nd graph
  string secPart2 = "/frozen_graphs/frozen_resnet_for_fdc_blk16x16_batchsize3072_step304857.pb";
  // 1920x1088
//  string secPart2 = "/frozen_graphs/frozen_resnet_for_fdc_blk16x16_batchsize8160_step304857.pb";
  string nameOfGraph2 = homeDir + secPart2;
  string graph2 = nameOfGraph2;
  std::unique_ptr<tensorflow::Session> session2;
  string graph_path2 = tensorflow::io::JoinPath(root_dir, graph2);
  Status load_graph_status2 = LoadGraph(graph_path2, &session2);
  if (!load_graph_status2.ok()) {
    LOG(ERROR) << load_graph_status2;
    return -1;
  }
  ///3nd graph
  string secPart3 = "/frozen_graphs/frozen_resnet_for_fdc_blk32x32_batchsize768_step304857.pb";
  // 1920x1088
//  string secPart3 = "/frozen_graphs/frozen_resnet_for_fdc_blk32x32_batchsize2040_step304857.pb";
  string nameOfGraph3 = homeDir + secPart3;
  string graph3 = nameOfGraph3;
  std::unique_ptr<tensorflow::Session> session3;
  string graph_path3 = tensorflow::io::JoinPath(root_dir, graph3);
  Status load_graph_status3 = LoadGraph(graph_path3, &session3);
  if (!load_graph_status3.ok()) {
    LOG(ERROR) << load_graph_status3;
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

//  std::chrono::system_clock::time_point time_before = std::chrono::system_clock::now();

  // call encoding function
  cTAppEncTop.encode(&session, &session2, &session3);

  // ending time
  dResult = (Double)(clock()-lBefore) / CLOCKS_PER_SEC;
  printf("\n Total Time: %12.3f sec.\n", dResult);
#if DMM1_TIME_MEASURE
  std::cout<< "\nTotal Time for DMM1 (xSearchDmm1Wedge):\n";
  std::cout << g_dmm1TimeCost << std::endl;
#endif
//  std::chrono::system_clock::time_point time_after = std::chrono::system_clock::now();
//  printf("[real-world total time]  %12.9f seconds \n", std::chrono::duration_cast<std::chrono::microseconds>(time_after - time_before).count() / 1000000.0);
//  std::cout << std::endl;

  // destroy application encoder class
  cTAppEncTop.destroy();

  return 0;
}

//! \}
