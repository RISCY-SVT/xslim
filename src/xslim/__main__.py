#!/usr/bin/env python3
# Copyright (c) 2023 SpacemiT. All rights reserved.
import argparse
import json
from typing import Optional, Sequence

from xslim.defs import _get_version


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xslim",
        description="Quantize, convert, or simplify an ONNX model.",
        epilog=(
            "examples:\n"
            "  xslim --config config.json\n"
            "  xslim -i model.onnx -o model.dynamic.onnx --dynq\n"
            "  xslim -i model.onnx -o model.fp16.onnx --fp16"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"xslim {_get_version()}")
    parser.add_argument("-c", "--config", default=None, help="JSON quantization configuration path.")
    parser.add_argument("-i", "--input_path", default=None, help="Input ONNX model path.")
    parser.add_argument("-o", "--output_path", default=None, help="Output ONNX model path.")
    parser.add_argument("--fp16", action="store_true", help="Convert the input model to FP16.")
    parser.add_argument("--dynq", action="store_true", help="Apply dynamic quantization.")
    parser.add_argument("--ignore_op_types", default="", help="Comma-separated operator types to exclude.")
    parser.add_argument("--ignore_op_names", default="", help="Comma-separated operator names to exclude.")
    parser.add_argument(
        "--opset",
        type=int,
        default=None,
        help="Convert the default ai.onnx opset to the target version.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.config is None and (args.input_path is None or args.output_path is None):
        parser.print_help()
        return 1

    if args.config is None:
        from .logger import logger

        precision_level = 100
        if args.fp16:
            precision_level = 4
        elif args.dynq:
            precision_level = 3

        if precision_level == 3:
            logger.info("No config provided, using default config, dynamic quantization...")
        elif precision_level == 4:
            logger.info("No config provided, using default config, convert onnx model to fp16...")
        elif precision_level >= 100:
            logger.info("No config provided, using default config, only simplify onnx model...")

        args.config = {
            "quantization_parameters": {
                "precision_level": precision_level,
                "ignore_op_types": [item for item in args.ignore_op_types.split(",") if item != ""],
                "ignore_op_names": [item for item in args.ignore_op_names.split(",") if item != ""],
            },
        }
        if args.opset is not None:
            args.config["model_parameters"] = {"opset": args.opset}
    elif args.opset is not None:
        with open(args.config, "r", encoding="utf-8") as fp:
            args.config = json.load(fp)
        args.config.setdefault("model_parameters", {})["opset"] = args.opset

    from .xslim_pipeline import quantize_onnx_model

    quantize_onnx_model(args.config, args.input_path, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
