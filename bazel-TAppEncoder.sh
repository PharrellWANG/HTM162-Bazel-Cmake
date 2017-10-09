#!/usr/bin/env bash
cd
cd tensorflow
bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.2 --config=cuda -k //HTM162/App/TAppEncoder/...
cd
cd tensorflow/HTM162