// A minimal but useful C++ example showing how to load an Imagenet-style object
// recognition TensorFlow model, prepare input images for it, run them through
// the graph, and interpret the results.
//
// It's designed to have as few dependencies and be as clear as possible, so
// it's more verbose than it could be in production code. In particular, using
// auto for the types of a lot of the returned values from TensorFlow calls can
// remove a lot of boilerplate, but I find the explicit types useful in sample
// code to make it simple to look up the classes involved.
//
// To use it, compile and then run in a working directory with the
// learning/brain/tutorials/label_image/data/ folder below it, and you should
// see the top k labels for the example Lena image output. You can then
// customize it to use your own models or images by changing the file names at
// the top of the main() function.
//

#include <fstream>
#include <vector>
#include "strtk.hpp"
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

#pragma clang diagnostic push
#pragma ide diagnostic ignored "OCDFAInspection"
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
#pragma clang diagnostic pop

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

  const int how_many_labels = std::min(5, static_cast<int>(label_count));
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

//function for parsing csv data
//void foo() {
//   std::string data = "1,2,3,4,5\n"
//                      "0,2,4,6,8\n"
//                      "1,3,5,7,9\n";
//  //shoudl be mode 10
//  std::string data = "8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11,11,11,11,10,10,10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,28,28,28,28,28,28,28,28,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,28,28,28,28,28,28,28,29,29,29,29,29,29,29,29,29,29,29,29,29,11,11,11,11,11,11,11,11,24,24,25,25,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29\n";
//
//  strtk::token_grid grid(data, data.size(), ",");
//
//  for (std::size_t i = 0; i < grid.row_count(); ++i) {
//    strtk::token_grid::row_type r = grid.row(i);
//    for (std::size_t j = 0; j < r.size(); ++j) {
//      std::cout << r.get<int>(j) << "\t";
//    }
//    std::cout << std::endl;
//  }
//  std::cout << std::endl;
//}

int main(int argc, char *argv[]) {
  // These are the command-line flags the program can understand.
  // They define where the graph and input data is located, and what kind of
  // input the model expects. If you train your own model, or use something
  // other than inception_v3, then you'll need to update these.
  string graph =
    "/Users/Pharrell_WANG/resnet_logs_bak/size_08_log/resnet/graphs/frozen_resnet_for_fdc_blk08x08_133049.pb";
  string graph_2 =
    "/Users/Pharrell_WANG/resnet_logs_bak/size_16_log/resnet/graphs/frozen_resnet_for_fdc_blk16x16_304857.pb";
  string labels =
    "/Users/Pharrell_WANG/labels/labels_for_fdc_32_classes.txt";
  string input_layer = "input";
  string output_layer = "logits/fdc_output_node";
  string root_dir = "";
  std::vector<Flag> flag_list = {
    Flag("graph", &graph, "graph to be executed"),
    Flag("labels", &labels, "name of file containing labels"),
    Flag("input_layer", &input_layer, "name of input layer"),
    Flag("output_layer", &output_layer, "name of output layer"),
    Flag("root_dir", &root_dir,
         "interpret image and graph file names relative to this directory"),
  };
  string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }

  // We need to call this to set up global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }

  // First we load and initialize the model.
  std::unique_ptr<tensorflow::Session> session;
  string graph_path = tensorflow::io::JoinPath(root_dir, graph);
  Status load_graph_status = LoadGraph(graph_path, &session);
  if (!load_graph_status.ok()) {
//        LOG(ERROR) << load_graph_status;
    return -1;
  }

  // First we load and initialize the model.
  std::unique_ptr<tensorflow::Session> session_2;
  string graph_path_2 = tensorflow::io::JoinPath(root_dir, graph_2);
  Status load_graph_status_2 = LoadGraph(graph_path_2, &session_2);
  if (!load_graph_status_2.ok()) {
//        LOG(ERROR) << load_graph_status;
    return -1;
  }

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

  tensorflow::Tensor input_tensor_2(tensorflow::DT_FLOAT,
                                    tensorflow::TensorShape({1, 16, 16, 1}));
  auto input_tensor_mapped_2 = input_tensor_2.tensor<float, 4>();

  // Assign block width
  int BLOCK_WIDTH_2 = 16;
  //************************ a few testing data start
  // this data should be mode 25
//  std::string data = "138,138,133,128,122,122,122,117,117,117,117,117,117,117,117,117,138,138,133,128,122,122,122,117,117,117,117,117,117,117,117,117,138,138,133,122,122,122,117,117,117,117,117,117,117,117,117,117,138,138,133,122,122,122,117,117,117,117,117,117,117,117,117,117,138,138,133,122,122,122,117,117,117,117,117,117,117,117,117,117,138,138,133,122,122,122,117,117,117,117,117,117,117,117,117,117,138,138,133,122,122,122,117,117,117,117,117,117,117,117,117,117,138,133,133,128,122,117,117,117,117,117,117,117,117,117,117,117,138,133,133,128,122,117,117,117,117,117,117,117,117,117,117,117,133,133,133,133,128,117,117,117,117,117,117,117,117,117,117,117,133,133,133,133,128,117,117,117,117,117,117,117,117,117,117,117,133,133,133,133,128,117,117,117,117,117,117,117,117,117,117,117,133,133,133,133,128,117,117,117,117,117,117,117,117,117,117,117,133,133,133,133,128,117,117,117,117,117,117,117,117,117,117,117,133,133,133,133,128,117,117,117,117,117,117,117,117,117,117,117,133,133,133,133,128,117,117,117,117,117,117,117,117,117,117,117\n";

  // shoud be mode 0
  std::string data = "90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,85,85,85,85,90,90,90,90,90,90,90,90,90,90,90,85,85,85,85,85,90,90,90,90,90,90,90,90,90,90,90,85,85,85,85,85,90,90,90,90,90,90,90,90,90,90,90,85,85,85,85,85,90,90,90,90,90,90,90,90,90,90,90,85,85,85,85,85,90,90,90,90,90,90,90,90,90,90,85,85,85,85,85,85,90,90,90,90,90,90,90,90,90,85,85,85,85,85,85,85,90,90,90,90,90,90,90,90,90,85,85,85,85,85,85,85,90,90,90,90,90,90,90,90,85,85,85,85,85,85,85,85,90,90,90,90,90,90,90,90,85,85,85,85,85,85,85,85,90,90,90,90,90,90,90,90,85,85,85,85,85,85,85,85,90,90,90,90,90,90,90,90,85,85,85,85,85,85,85,85,90,90,90,90,90,90,90,90,85,85,85,85,85,85,85,85\n";
//************************ a few testing data end

  strtk::token_grid grid(data, data.size(), ",");

  strtk::token_grid::row_type r = grid.row(0);
  // set values and copy to ``input_tensor`` using for loop
  for (int row = 0; row < BLOCK_WIDTH_2; ++row)
    for (int col = 0; col < BLOCK_WIDTH_2; ++col)
      input_tensor_mapped_2(0, row, col,
                            0) = r.get<int>(size_t(row * BLOCK_WIDTH_2 + col)); // this is where we get the pixels

  // Actually run the image through the model.
  std::vector<Tensor> outputs;
  std::vector<Tensor> outputs_2;

  Status run_status = session->Run({{input_layer, input_tensor}},
                                   {output_layer}, {}, &outputs);

  Status run_status_2 = session_2->Run({{input_layer, input_tensor_2}},
                                       {output_layer}, {}, &outputs_2);
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
  }

  // Do something interesting with the results we've generated.
  Status print_status = PrintTopLabels(outputs, labels);

  Status print_status_2 = PrintTopLabels(outputs_2, labels);

  if (!print_status.ok()) {
    LOG(ERROR) << "Running print failed: " << print_status;
    return -1;
  }

  return 0;
}
