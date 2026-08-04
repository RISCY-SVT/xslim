"""RISCY-SVT regression coverage for ONNX ReduceMax edge semantics."""

import unittest

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper, numpy_helper, version_converter

from xslim.ppq_decorator.ppq.core import TargetPlatform
from xslim.ppq_decorator.ppq.executor.op import (
    DEFAULT_BACKEND_TABLE,
    TorchBackendContext,
)
from xslim.ppq_decorator.ppq.executor.torch import TorchExecutor
from xslim.ppq_decorator.ppq.IR import Operation, Variable
from xslim.ppq_decorator.ppq.IR.base.opdef import Opset
from xslim.ppq_decorator.ppq.parser.onnx_parser import OnnxParser


CTX = TorchBackendContext(executing_device="cpu")


def make_reduce(attributes=None, inputs=2, opset=18):
    variables = [Variable(name=f"input_{index}") for index in range(inputs)]
    output = Variable(name="output")
    operation = Operation(
        name="reduce_max",
        op_type="ReduceMax",
        attributes=attributes or {},
        platform=TargetPlatform.UNSPECIFIED,
        inputs=variables,
        outputs=[output],
    )
    operation.opset = Opset(version=opset)
    return operation


def reduce(value, axes=None, keepdims=1, noop=0, opset=18):
    attributes = {"keepdims": keepdims, "noop_with_empty_axes": noop}
    values = [value]
    inputs = 1
    if axes is not None and opset >= 18:
        values.append(torch.tensor(axes, dtype=torch.int64))
        inputs = 2
    elif axes is not None:
        attributes["axes"] = axes
    return DEFAULT_BACKEND_TABLE["ReduceMax"](
        make_reduce(attributes, inputs, opset), values, CTX
    )


class TestRiscyReduceMaxRegressions(unittest.TestCase):
    def test_opset13_attribute_axes_convert_to_opset18_input(self):
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 1])
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("ReduceMax", ["x"], ["y"], axes=[1])],
                "reduce13",
                [x],
                [y],
            ),
            opset_imports=[helper.make_opsetid("", 13)],
        )
        converted = version_converter.convert_version(model, 18)
        node = converted.graph.node[-1]
        self.assertEqual(node.op_type, "ReduceMax")
        self.assertEqual(len(node.input), 2)
        expected = np.asarray([[3.0], [6.0]], dtype=np.float32)
        actual = ort.InferenceSession(
            converted.SerializeToString(), providers=["CPUExecutionProvider"]
        ).run(None, {"x": np.arange(1, 7, dtype=np.float32).reshape(2, 3)})[0]
        np.testing.assert_array_equal(actual, expected)

    def test_opset18_multiple_and_negative_axes(self):
        value = torch.arange(24.0).reshape(2, 3, 4)
        actual = reduce(value, axes=[-1, 1], keepdims=0)
        torch.testing.assert_close(actual, torch.amax(value, dim=(-1, 1)))

    def test_keepdims_zero_and_one(self):
        value = torch.arange(24.0).reshape(2, 3, 4)
        self.assertEqual(reduce(value, [1], keepdims=0).shape, (2, 4))
        self.assertEqual(reduce(value, [1], keepdims=1).shape, (2, 1, 4))

    def test_empty_axes_noop_zero_reduces_all(self):
        value = torch.tensor([[1.0, 4.0], [3.0, 2.0]])
        actual = reduce(value, axes=[], keepdims=1, noop=0)
        torch.testing.assert_close(actual, torch.tensor([[4.0]]))

    def test_empty_axes_noop_one_is_identity(self):
        value = torch.tensor([[1.0, 4.0], [3.0, 2.0]])
        actual = reduce(value, axes=[], keepdims=0, noop=1)
        self.assertIs(actual, value)

    def test_empty_float_reduction_uses_negative_infinity(self):
        value = torch.empty((0, 3), dtype=torch.float32)
        actual = reduce(value, axes=[0], keepdims=1)
        self.assertEqual(actual.shape, (1, 3))
        self.assertTrue(torch.isneginf(actual).all())

    def test_empty_integer_reduction_uses_dtype_minimum(self):
        value = torch.empty((0, 3), dtype=torch.int8)
        actual = reduce(value, axes=[0], keepdims=0)
        self.assertEqual(actual.shape, (3,))
        torch.testing.assert_close(
            actual, torch.full((3,), torch.iinfo(torch.int8).min, dtype=torch.int8)
        )

    def test_empty_unreduced_dimension_stays_empty(self):
        value = torch.empty((0, 3), dtype=torch.float32)
        actual = reduce(value, axes=[1], keepdims=1)
        self.assertEqual(actual.shape, (0, 1))
        self.assertEqual(actual.numel(), 0)

    def test_conv_reduce_max_matches_onnxruntime(self):
        x_info = helper.make_tensor_value_info(
            "x", TensorProto.FLOAT, [1, 1, 3, 3]
        )
        y_info = helper.make_tensor_value_info(
            "y", TensorProto.FLOAT, [1, 1, 1, 1]
        )
        weight = numpy_helper.from_array(
            np.asarray([[[[1.0, -0.5], [0.25, 2.0]]]], dtype=np.float32),
            "weight",
        )
        axes = numpy_helper.from_array(np.asarray([2, 3], dtype=np.int64), "axes")
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node("Conv", ["x", "weight"], ["conv"]),
                    helper.make_node(
                        "ReduceMax", ["conv", "axes"], ["y"], keepdims=1
                    ),
                ],
                "conv_reduce_max",
                [x_info],
                [y_info],
                [weight, axes],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
        )
        onnx.checker.check_model(model)
        input_value = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)
        expected = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        ).run(None, {"x": input_value})[0]
        graph = OnnxParser().build(model)
        actual = TorchExecutor(graph=graph, device="cpu").forward(
            torch.from_numpy(input_value)
        )[0].detach().cpu().numpy()
        # Conv implementations are compared numerically, not called byte-exact.
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
