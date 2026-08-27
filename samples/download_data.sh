#!/usr/bin/env bash
set -euo pipefail

mkdir -p models
wget --https-only --output-document models/resnet18.onnx https://mmdeploy-oss.openmmlab.com/model/mmcls/resnet18-b7eb3f.onnx
wget --https-only --output-document models/mobilenet_v3_small.onnx https://mmdeploy-oss.openmmlab.com/model/mmcls/mobilenet-v3-small-5461e6.onnx
wget --https-only --output-document models/bertsquad.onnx 'https://media.githubusercontent.com/media/onnx/models/main/validated/text/machine_comprehension/bert-squad/model/bertsquad-12.onnx?download=true'
wget --https-only --output-document Imagenet.tar.gz https://archive.spacemit.com/spacemit-ai/ModelZoo/dataset/Imagenet.tar.gz
tar -xf Imagenet.tar.gz
