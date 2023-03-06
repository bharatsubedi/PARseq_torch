
from __future__ import print_function

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

# Use autoprimaryctx if available (pycuda >= 2021.1) to
# prevent issues with other modules that rely on the primary
# device context.
try:
    import pycuda.autoprimaryctx
except ModuleNotFoundError:
    import pycuda.autoinit

import sys, os

sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common
# from downloader import getFilePath

TRT_LOGGER = trt.Logger()

import torch
from torchvision.transforms import Normalize

from skimage import io
from skimage.transform import resize
from skimage.color import gray2rgb
import matplotlib.pyplot as plt

from model.tokenizer_utils import Tokenizer

def preprocess_image(img):
    norm = Normalize(0.5, 0.5)
    result = norm(torch.from_numpy(img).transpose(0,2).transpose(1,2))
    return np.array(result, dtype=np.float16)

def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            common.EXPLICIT_BATCH
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser, trt.Runtime(
            TRT_LOGGER
        ) as runtime:
            config.max_workspace_size = 1 << 28  # 256MiB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print(
                    "ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.".format(onnx_file_path)
                )
                exit(0)
            print("Loading ONNX file from path {}...".format(onnx_file_path))
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [1, 3, 32, 128]
            print("Completed parsing of ONNX file")
            print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def main():
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""

    # Try to load a previously generated YOLOv3-608 network graph in ONNX format:
    onnx_file_path = "parseq_ar_r1.onnx"
    engine_file_path = "parseq_ar_r1.trt"
#     test_folder_path = ''
    
    # Download a dog image and save it to the following file path:
#     input_image_path = getFilePath("samples/python/yolov3_onnx/dog.jpg")

    img_path = 'sample.png'
    img = resize(gray2rgb(io.imread(img_path)), (32, 128))
    input_batch = np.array(np.repeat(np.expand_dims(np.array(img, dtype=np.float32), axis=0), 1, axis=0), dtype=np.float32)
    print('input shape: ', input_batch.shape)
    preprocessed_images = np.array([preprocess_image(image) for image in input_batch])

    # Do inference with TensorRT
    trt_outputs = []
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Do inference
        print("Running inference on image {}...".format(img_path))
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        inputs[0].host = preprocessed_images
        trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    trt_outputs = [out.reshape(1, 26, 1794) for out in trt_outputs]
    print('Type of Generated outputs: ', type(trt_outputs))
    print('Len of Generated outputs: ', (trt_outputs[0].shape))
    print('Generated outputs: ', trt_outputs)
    
    pred_labels = post_process(trt_outputs)
    print('Final results: ', pred_labels)

    
def post_process(trt_outputs):
    with open('./char_dicts/charset.txt', 'r') as f:
        charset = f.read()
    tokenizer = Tokenizer(charset)
    trt_logits = [torch.nn.functional.softmax(torch.from_numpy(out), dim=-1) for out in trt_outputs]
    final_outputs = []
    for logit in trt_logits:
        preds, probs = tokenizer.decode(logit)
        final_outputs.append([preds[0], probs])
        
    return final_outputs
if __name__ == "__main__":
    main()