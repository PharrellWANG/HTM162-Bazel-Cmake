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
// see the top five labels for the example Lena image output. You can then
// customize it to use your own models or images by changing the file names at
// the top of the main() function.
//
// The googlenet_graph.pb file included by default is created from Inception.

#include <fstream>
#include <vector>
//#include <typeinfo>
//#include <time.h>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
//#include "tensorflow/core/framework/graph.pb.h"
//#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
//#include "tensorflow/core/lib/core/errors.h"
//#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
//#include "tensorflow/core/platform/logging.h"
//#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
//#include "tensorflow/core/framework/tensor_shape.h"
//#include "tensorflow/cc/framework/ops.h"

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

#define VERBOSE false
// if you want to use the model from workspace,
// then set it to true;
// if you want to use slim models,
// set it as false
#define ENABLE_MY_RESNET true
// when you want to read a line of data in
// c++ instead of real image, set this to true
#define CUSTOMIZED_READER true

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

#if ENABLE_MY_RESNET

Status ReadTensorFromBlkPel(const int block_size,
                            std::vector<Tensor> *out_tensors) {
  const int wanted_size = 8;
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

//    string input_name = "block_reader";
  string output_name = "block_for_classification";
  // Bilinearly resize the image to fit the required dimensions.
  tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT,
                                  tensorflow::TensorShape({1, 8, 8, 1}));

  // input_tensor_mapped is
  // 1. an interface to the data of ``input_tensor``
  // 1. It is used to copy data into the ``input_tensor``
  auto input_tensor_mapped = input_tensor.tensor<float, 4>();

  // set values and copy to ``input_tensor`` using for loop
  for (int row = 0; row < block_size; ++row)
    for (int col = 0; col < block_size; ++col)
      input_tensor_mapped(0, row, col,
                          0) = 3.0;
  // this is where we get the pixels,
  // add pel pointer in the args // pha.zx


//    if (8 < block_size < 64) {
//        auto resized_input_tensor = tensorflow::ops::ResizeBilinear(
//                root, input_tensor,
//                Const(root.WithOpName("size"), {wanted_size, wanted_size}));
//    }
  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
    tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(session->Run({}, {output_name}, {}, out_tensors));
  // if this is too slow, please use below method, which can turn 32
  // to 16, turn 16 to 8, 32-->16-->8
  //y[i/2][j/2] = (x[i][j] + x[i+1][j] + x[i][j+1] + x[i+1][j+1]) / 4;
  return Status::OK();
}

#else

// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
Status ReadTensorFromImageFile(string file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor> *out_tensors) {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    string input_name = "file_reader";
    string output_name = "normalized";

    clock_t lBefore_0 = clock();
    auto file_reader =
            tensorflow::ops::ReadFile(root.WithOpName(input_name), file_name);
    double dResult_0 = (double) (clock() - lBefore_0) / CLOCKS_PER_SEC;
    printf("\n Total Time for read a image: %12.3f sec.\n", dResult_0);
    // Now try to figure out what kind of file it is and decode it.
    const int wanted_channels = 3;
    tensorflow::Output image_reader;
    if (tensorflow::StringPiece(file_name).ends_with(".png")) {
        image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
                                 DecodePng::Channels(wanted_channels));
    } else if (tensorflow::StringPiece(file_name).ends_with(".gif")) {
        image_reader = DecodeGif(root.WithOpName("gif_reader"), file_reader);
    } else {
        // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
        image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                                  DecodeJpeg::Channels(wanted_channels));
    }
    // Now cast the image data to float so we can do normal math on it.
    auto float_caster =
            Cast(root.WithOpName("float_caster"), image_reader,
                 tensorflow::DT_FLOAT);

    // The convention for image ops in TensorFlow is that all images are expected
    // to be in batches, so that they're four-dimensional arrays with indices of
    // [batch, height, width, channel]. Because we only have a single image, we
    // have to add a batch dimension of 1 to the start with ExpandDims().
    auto dims_expander = ExpandDims(root, float_caster, 0);
    // Bilinearly resize the image to fit the required dimensions.
    auto resized = ResizeBilinear(
            root, dims_expander,
            Const(root.WithOpName("size"), {input_height, input_width}));
    // Subtract the mean and divide by the scale.
    Div(root.WithOpName(output_name), Sub(root, resized, {input_mean}),
        {input_std});

    // This runs the GraphDef network definition that we've just constructed, and
    // returns the results in the output tensor.

    clock_t lBefore = clock();

    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

    std::unique_ptr<tensorflow::Session> session(
            tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_RETURN_IF_ERROR(session->Create(graph));
    TF_RETURN_IF_ERROR(session->Run({}, {output_name}, {}, out_tensors));
    // if this is too slow, please use below method, which can turn 32
    // to 16, turn 16 to 8, 32-->16-->8
    //y[i/2][j/2] = (x[i][j] + x[i+1][j] + x[i][j+1] + x[i+1][j+1]) / 4;

    double dResult = (double) (clock() - lBefore) / CLOCKS_PER_SEC;
    printf("\n Total Time for construct a graph and run a session: %12.3f sec.\n",
           dResult);

    return Status::OK();
}

#endif

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
//        LOG(ERROR) << read_labels_status;
    return read_labels_status;
  }

  const int how_many_labels = std::min(20, static_cast<int>(label_count));
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
//                LOG(INFO) << labels[label_index] << " (" << label_index << "): "
//                  << score;
  }
  LOG(INFO) << "";
  return Status::OK();
}

int main(int argc, char *argv[]) {
  // These are the command-line flags the program can understand.
  // They define where the graph and input data is located, and what kind of
  // input the model expects. If you train your own model, or use something
  // other than inception_v3, then you'll need to update these.
#if ENABLE_MY_RESNET
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
#else
  string image =
          "/Users/Pharrell_WANG/tensorflow/tensorflow/examples/label_image/"
                  "data/grace_hopper.jpg";
  string graph =
          "/Users/Pharrell_WANG/tensorflow/tensorflow/examples/label_image/data/"
                  "inception_v3_2016_08_28_frozen.pb";
  string labels =
          "/Users/Pharrell_WANG/tensorflow/tensorflow/examples/label_image/data/"
                  "imagenet_slim_labels.txt";
  int32 input_width = 299;
  int32 input_height = 299;
  int32 input_mean = 0;
  int32 input_std = 255;
  string input_layer = "input";
  string output_layer = "InceptionV3/Predictions/Reshape_1";
  bool self_test = false;
  string root_dir = "";
  std::vector<Flag> flag_list = {
          Flag("image", &image, "image to be processed"),
          Flag("graph", &graph, "graph to be executed"),
          Flag("labels", &labels, "name of file containing labels"),
          Flag("input_width", &input_width,
               "resize image to this width in pixels"),
          Flag("input_height", &input_height,
               "resize image to this height in pixels"),
          Flag("input_mean", &input_mean, "scale pixel values to this mean"),
          Flag("input_std", &input_std,
               "scale pixel values to this std deviation"),
          Flag("input_layer", &input_layer, "name of input layer"),
          Flag("output_layer", &output_layer, "name of output layer"),
          Flag("self_test", &self_test, "run a self test"),
          Flag("root_dir", &root_dir,
               "interpret image and graph file names relative to this directory"),
  };
#endif
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

#if !ENABLE_MY_RESNET && !CUSTOMIZED_READER // THIS means not using models/resnet and use the image reader
  // Get the image from disk as a float array of numbers, resized and normalized
  // to the specifications the main graph expects.
  std::vector<Tensor> resized_tensors;
  string image_path = tensorflow::io::JoinPath(root_dir, image);
  Status read_tensor_status =
          ReadTensorFromImageFile(image_path, input_height, input_width,
                                  input_mean,
                                  input_std, &resized_tensors);
  if (!read_tensor_status.ok()) {
      //        LOG(ERROR) << read_tensor_status;
      return -1;
  }
  const Tensor &resized_tensor = resized_tensors[0];
#elif !ENABLE_MY_RESNET && CUSTOMIZED_READER // THIS means not using models/resnet and use the csv reader
  // Get the image from disk as a float array of numbers, resized and normalized
// to the specifications the main graph expects.
  std::vector<Tensor> resized_tensors;
//    string image_path = tensorflow::io::JoinPath(root_dir, image);
//    Status read_tensor_status =
//            ReadTensorFromImageFile(image_path, input_height, input_width,
//                                    input_mean,
//                                    input_std, &resized_tensors);
//    if (!read_tensor_status.ok()) {
//                LOG(ERROR) << read_tensor_status;
//        return -1;
//    }
//    const Tensor &resized_tensor = resized_tensors[0];


  LOG(INFO) << "Experiment goes wild";
  LOG(INFO) << "";
  tensorflow::Tensor resized_tensor(tensorflow::DT_FLOAT,
                                    tensorflow::TensorShape({1, 299, 299, 3}));
// input_tensor_mapped is
// 1. an interface to the data of ``input_tensor``
// 1. It is used to copy data into the ``input_tensor``
  auto input_tensor_mapped = resized_tensor.tensor<float, 4>();

// Assign block width
  int BLOCK_WIDTH = 299;

// set values and copy to ``input_tensor`` using for loop
  for (int row = 0; row < BLOCK_WIDTH; ++row)
      for (int col = 0; col < BLOCK_WIDTH; ++col)
          input_tensor_mapped(0, row, col,
                              0) = 3.0; // this is where we get the pixels

  for (int row = 0; row < BLOCK_WIDTH; ++row)
      for (int col = 0; col < BLOCK_WIDTH; ++col)
          input_tensor_mapped(0, row, col,
                              1) = 4.0; // this is where we get the pixels


  for (int row = 0; row < BLOCK_WIDTH; ++row)
      for (int col = 0; col < BLOCK_WIDTH; ++col)
          input_tensor_mapped(0, row, col,
                              2) = 5.0; // this is where we get the pixels

  LOG(INFO) << "Q: The DebugString of the tensor?";
  LOG(INFO) << resized_tensor.DebugString();
  LOG(INFO) << "Q: The dimension of the tensor?";
  LOG(INFO) << resized_tensor.dims();
  LOG(INFO) << "Q: Is this tensor initialized?";
  LOG(INFO) << resized_tensor.IsInitialized();
#endif

#if ENABLE_MY_RESNET

#if VERBOSE
  LOG(INFO) << "==========================================Playing Start";
  LOG(INFO) << "Experiment goes wild";
  LOG(INFO) << "";
#endif
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

  // set values and copy to ``input_tensor`` using for loop
  for (int row = 0; row < BLOCK_WIDTH_2; ++row)
    for (int col = 0; col < BLOCK_WIDTH_2; ++col)
      input_tensor_mapped_2(0, row, col,
                            0) = 3.0; // this is where we get the pixels
#if VERBOSE
  LOG(INFO) << "Q: The DebugString of the tensor?";
  LOG(INFO) << input_tensor.DebugString();
  LOG(INFO) << "Q: The dimension of the tensor?";
  LOG(INFO) << input_tensor.dims();
  LOG(INFO) << "Q: Is this tensor initialized?";
  LOG(INFO) << input_tensor.IsInitialized();
  LOG(INFO) << "========================================== End";
#endif
#else
  LOG(INFO) << resized_tensor.DebugString();
#endif
  // Actually run the image through the model.
  std::vector<Tensor> outputs;
  std::vector<Tensor> outputs_2;
#if ENABLE_MY_RESNET

  Status run_status = session->Run({{input_layer, input_tensor}},
                                   {output_layer}, {}, &outputs);

  Status run_status_2 = session_2->Run({{input_layer, input_tensor_2}},
                                       {output_layer}, {}, &outputs_2);
#else
  Status run_status = session->Run({{input_layer, resized_tensor}},
                                   {output_layer}, {}, &outputs);
#endif
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
