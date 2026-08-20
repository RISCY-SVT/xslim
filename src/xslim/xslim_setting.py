#!/usr/bin/env python3
# Copyright (c) 2023 SpacemiT. All rights reserved.
# Modified by RISCY-SVT in 2026: add fail-closed local constrained range settings.
import copy
import json
import os
from enum import Enum
from typing import Dict, Literal, Optional, Pattern, Sequence, Union

import onnx

from xslim.logger import logger

from .defs import XQUANT_CONFIG, AutoFinetuneLevel, PrecisionLevel
from .ppq_decorator import BaseGraph


class SettingSerialize:
    def check(self, qsetting):
        pass

    def from_json(self, obj_setting: dict, qsetting):
        for key, value in obj_setting.items():
            if key in self.__dict__:
                if "builtin" in self.__dict__[key].__class__.__module__:
                    if isinstance(value, list) and hasattr(self, "from_list"):
                        self.__dict__[key] = getattr(
                            self, "from_list")(key, value, qsetting)
                    else:
                        self.__dict__[key] = copy.deepcopy(value)
                elif isinstance(self.__dict__[key], Enum):
                    self.__dict__[key] = self.__dict__[key].__class__(value)
                else:
                    assert isinstance(value, dict)
                    if isinstance(self.__dict__[key], SettingSerialize):
                        self.__dict__[key].from_json(value, qsetting)

        self.check(qsetting)

    def to_json(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4, ensure_ascii=False)

    def to_dict(self) -> str:
        return {k: v for k, v in self.__dict__.items()}


class ModelParameterSetting(SettingSerialize):
    def __init__(self) -> None:
        self.onnx_model: str = None
        self.output_prefix: str = None
        self.working_dir: str = None
        self.skip_onnxsim: bool = False
        self.opset: Optional[int] = None

    def check(self, qsetting):
        # if not os.path.exists(self.onnx_model):
        #    raise FileExistsError(self.onnx_model)

        if self.opset is not None and not isinstance(self.opset, int):
            raise TypeError("opset type error, {} .vs int".format(type(self.opset)))

        if self.opset is not None and self.opset < 1:
            raise ValueError("opset should be a positive integer.")

        if self.working_dir is None:
            if isinstance(self.onnx_model, str) and os.path.exists(self.onnx_model):
                self.working_dir = os.path.dirname(self.onnx_model)
            else:
                self.working_dir = os.path.join(os.curdir, "temp")
            logger.info(
                "Not set working_dir, deatults to {}.".format(self.working_dir))

        if self.output_prefix is None:
            if isinstance(self.onnx_model, str) and os.path.exists(self.onnx_model):
                model_name = os.path.splitext(
                    os.path.basename(self.onnx_model))[0]
                self.output_prefix = "{}.q".format(model_name)
            else:
                self.output_prefix = "xslim.q"
            logger.info("Not set output_prefix, deatults to {}.".format(
                self.output_prefix))


class InputParameterSetting(SettingSerialize):
    def __init__(self) -> None:
        self.input_name: str = None
        self.input_shape: Sequence[int] = None
        self.file_type: Literal["img", "npy", "raw"] = "img"
        self.color_format: Literal["rgb", "bgr"] = "bgr"
        self.mean_value: Sequence[float] = None
        self.std_value: Sequence[float] = None
        self.preprocess_file: str = None
        self.data_list_path: str = None
        self.dtype: str = "float32"

    def check(self, qsetting):
        if self.preprocess_file is not None and not isinstance(self.preprocess_file, str):
            raise TypeError(
                "preprocess_file type error, {} .vs str".format(self.preprocess_file))

        if self.data_list_path is not None and not isinstance(self.data_list_path, str):
            raise TypeError(
                "data_list_path type error, {} .vs str".format(self.data_list_path))

        if isinstance(self.data_list_path, str) and not os.path.exists(self.data_list_path):
            raise FileExistsError(self.data_list_path)

        if self.mean_value is not None and not isinstance(self.mean_value, list):
            raise TypeError(
                "mean_value type error, {} .vs str".format(self.mean_value))

        if self.std_value is not None and not isinstance(self.std_value, list):
            raise TypeError(
                "std_value type error, {} .vs str".format(self.std_value))

        if self.file_type not in {"img", "npy", "raw"}:
            raise NotImplementedError(
                "file_type {} not implemented yet.".format(self.file_type))


class ConstrainedRangePolicySetting(SettingSerialize):
    """Optional signed per-tensor range override for an explicitly selected domain."""

    def __init__(self) -> None:
        self.enabled: bool = False
        self.strict: bool = True
        self.objective: str = "constrained-mse"
        self.preserve_zero: bool = True
        self.required_real_min: Optional[float] = None
        self.required_real_max: Optional[float] = None
        self.semantic_floor: Optional[Union[str, float]] = None
        self.required_intervals: Sequence[Dict[str, object]] = []
        self.minimum_positive_codes: int = 0
        self.minimum_negative_codes: int = 0
        self.maximum_clipping_fraction: Optional[float] = None
        self.maximum_rail_fraction: Optional[float] = None
        self.percentile: float = 0.9999
        self.search_steps: int = 32
        self.scale_epsilon: float = 1.0e-12
        self.lock_qparams: bool = True

    def from_json(self, obj_setting: dict, qsetting):
        unknown = sorted(set(obj_setting) - set(self.__dict__))
        if unknown:
            raise ValueError("unknown range_policy fields: {}".format(", ".join(unknown)))
        super().from_json(obj_setting, qsetting)

    def check(self, qsetting):
        from .range_policy import ConstrainedRangeSpec

        ConstrainedRangeSpec(
            objective=self.objective,
            preserve_zero=self.preserve_zero,
            required_real_min=self.required_real_min,
            required_real_max=self.required_real_max,
            semantic_floor=self.semantic_floor,
            required_intervals=tuple(self.required_intervals),
            minimum_positive_codes=self.minimum_positive_codes,
            minimum_negative_codes=self.minimum_negative_codes,
            maximum_clipping_fraction=self.maximum_clipping_fraction,
            maximum_rail_fraction=self.maximum_rail_fraction,
            percentile=self.percentile,
            search_steps=self.search_steps,
            scale_epsilon=self.scale_epsilon,
        )
        if not isinstance(self.enabled, bool):
            raise TypeError("range_policy.enabled must be bool")
        if not isinstance(self.strict, bool):
            raise TypeError("range_policy.strict must be bool")
        if not isinstance(self.lock_qparams, bool):
            raise TypeError("range_policy.lock_qparams must be bool")

    def to_spec_dict(self) -> Dict[str, object]:
        return {
            "objective": self.objective,
            "preserve_zero": self.preserve_zero,
            "required_real_min": self.required_real_min,
            "required_real_max": self.required_real_max,
            "semantic_floor": self.semantic_floor,
            "required_intervals": list(self.required_intervals),
            "minimum_positive_codes": self.minimum_positive_codes,
            "minimum_negative_codes": self.minimum_negative_codes,
            "maximum_clipping_fraction": self.maximum_clipping_fraction,
            "maximum_rail_fraction": self.maximum_rail_fraction,
            "percentile": self.percentile,
            "search_steps": self.search_steps,
            "scale_epsilon": self.scale_epsilon,
        }


class CustomQuantizationParameterSetting(SettingSerialize):
    def __init__(self) -> None:
        self.name: str = None
        self.input_names: Sequence[str] = None
        self.output_names: Sequence[str] = None
        self.tensor_names: Sequence[str] = None
        self.max_percentile: float = None
        self.precision_level: int = None
        self.calibration_type: str = None
        self.range_policy: ConstrainedRangePolicySetting = ConstrainedRangePolicySetting()

    def check(self, qsetting):
        for field_name in ("input_names", "output_names", "tensor_names"):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, list) or not all(isinstance(item, str) and item for item in value)
            ):
                raise TypeError("{} must be a list of non-empty tensor names".format(field_name))
        if self.name is not None and (not isinstance(self.name, str) or not self.name):
            raise TypeError("custom_setting.name must be a non-empty string")
        if self.calibration_type is not None and self.calibration_type not in {
            "default",
            "minmax",
            "percentile",
            "kl",
            "mse",
        }:
            raise ValueError("unsupported custom calibration_type {}".format(self.calibration_type))
        if self.max_percentile is not None and not 0.5 < float(self.max_percentile) <= 1.0:
            raise ValueError("custom max_percentile must be in (0.5, 1.0]")
        if self.range_policy.enabled and not self.tensor_names and not (self.input_names and self.output_names):
            raise ValueError(
                "enabled range_policy requires tensor_names or a bounded input_names/output_names selector"
            )


class QuantizationParameterSetting(SettingSerialize):
    def __init__(self) -> None:
        self.precision_level: PrecisionLevel = PrecisionLevel.LEVEL_0
        self.max_percentile: float = None
        self.finetune_level: AutoFinetuneLevel = AutoFinetuneLevel.LEVEL_1
        self.custom_setting: Sequence[CustomQuantizationParameterSetting] = None
        self.range_policy_manifest_path: str = None
        self.analysis_enable: bool = True
        self.truncate_var_names: Sequence[str] = []
        self.ignore_op_types: Sequence[str] = []
        self.ignore_op_names: Sequence[str] = []

    def from_list(self, value_name, obj_setting, qsetting):
        if value_name == "custom_setting":
            custom_setting = []
            for item_dict in obj_setting:
                setting_item = CustomQuantizationParameterSetting()
                setting_item.from_json(item_dict, qsetting)
                custom_setting.append(setting_item)
            return custom_setting
        else:
            return obj_setting

    def check(self, qsetting):
        if self.precision_level.value > PrecisionLevel.LEVEL_0.value:
            logger.info("set higher precision level.")
        if self.finetune_level.value > AutoFinetuneLevel.LEVEL_1.value:
            logger.info("set higher finetune level.")
        constrained = [
            item for item in self.custom_setting or [] if item.range_policy.enabled
        ]
        if constrained and not self.range_policy_manifest_path:
            raise ValueError("enabled range_policy requires range_policy_manifest_path")


class CalibrationParameterSetting(SettingSerialize):
    def __init__(self) -> None:
        self.calibration_batch_size: int = 1
        self.calibration_step: int = 500
        self.calibration_device: str = "cuda" if XQUANT_CONFIG.cuda_support else "cpu"
        self.calibration_type: str = "default"
        self.input_parameters: Sequence[InputParameterSetting] = None

    def from_list(self, value_name: str, obj_setting: Sequence, qsetting):
        if value_name == "input_parameters" or value_name == "input_parametres":
            input_parameters = []
            for item_dict in obj_setting:
                setting_item = InputParameterSetting()
                setting_item.from_json(item_dict, qsetting)
                input_parameters.append(setting_item)
            return input_parameters
        else:
            return obj_setting

    def check(self, qsetting):
        if self.calibration_device == "cuda" and not XQUANT_CONFIG.cuda_support:
            logger.warning(
                "Specifies that cuda is used but not detected. Turn to cpu.")
            self.calibration_device = "cpu"

        if self.calibration_step > 1000 or self.calibration_step < 10:
            logger.warning(
                "Specifies that calibration_step is too large or too small, it should be in [10, 1000].")

        assert len(
            self.input_parameters) > 0, "Calibration input_parameters setting not detected."

        if self.calibration_type not in {"default", "minmax", "percentile", "kl", "mse"}:
            raise NotImplementedError(
                "calibration_type {} not implemented yet.".format(self.calibration_type))

        if self.calibration_type != "default":
            logger.info("set calibration_type {}.".format(
                self.calibration_type))

    def check_input_parameters(self, ppq_ir: BaseGraph):
        assert len(self.input_parameters) == len(
            ppq_ir.inputs
        ), "Calibration input_parameters size should <= model inputs size."

        pb_input = ppq_ir._detail.get("pb_input", [])
        for in_type, calib_parameter in zip(pb_input, self.input_parameters):
            input_shape = [
                i.dim_value if isinstance(
                    i.dim_value, int) and i.dim_value > 0 else None
                for i in in_type.type.tensor_type.shape.dim
            ]
            if isinstance(in_type.type.tensor_type.elem_type, int):
                input_dtype = onnx.helper.tensor_dtype_to_np_dtype(
                    in_type.type.tensor_type.elem_type
                ).name
            else:
                input_dtype = None
            if input_dtype is not None:
                calib_parameter.dtype = input_dtype
            if calib_parameter.input_name is None:
                calib_parameter.input_name = in_type.name
            if calib_parameter.input_shape is None and len(input_shape) > 0:
                input_shape[0] = input_shape[0] if isinstance(
                    input_shape[0], int) else 1
                calib_parameter.input_shape = input_shape


class XSlimSetting(SettingSerialize):
    def __init__(self) -> None:
        self.model_parameters = ModelParameterSetting()
        self.calibration_parameters = CalibrationParameterSetting()
        self.quantization_parameters = QuantizationParameterSetting()


class XSlimSettingFactory:
    def from_json(json_obj: Union[str, dict]) -> XSlimSetting:
        if isinstance(json_obj, str):
            setting_dict = json.loads(json_obj)
        else:
            setting_dict = json_obj

        setting = XSlimSetting()
        setting.from_json(setting_dict, setting)

        return setting
