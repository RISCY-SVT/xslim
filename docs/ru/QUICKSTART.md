# Быстрый старт XSlim

Сначала установите CPython 3.12.3 и фиксированные зависимости по
[INSTALL.md](../../INSTALL.md). Это общий рецепт PTQ, не выполненный в
maintenance и не воспроизводящий замороженные B2/C2. Пример, который можно
выполнить без модели и датасета: [reconstruction API](../RECONSTRUCTION_GUIDE.md).

## 1. Подготовьте данные

Нужны FP32-модель `model.onnx`, файл `calibration.txt` с одним изображением на
строку и `config.json`. Калибровочный preprocessing должен точно совпадать с
рабочим: RGB/BGR, resize/letterbox, padding, scale, mean/std, NCHW и dtype.

## 2. Создайте конфигурацию

```json
{
  "model_parameters": {
    "onnx_model": "./model.onnx",
    "working_dir": "./output",
    "output_prefix": "model_s8_qdq"
  },
  "calibration_parameters": {
    "calibration_step": 100,
    "calibration_batch_size": 1,
    "calibration_device": "cpu",
    "calibration_type": "default",
    "input_parameters": [{
      "input_name": "images",
      "input_shape": [1, 3, 640, 640],
      "file_type": "img",
      "color_format": "rgb",
      "mean_value": [0, 0, 0],
      "std_value": [1, 1, 1],
      "data_list_path": "./calibration.txt"
    }]
  },
  "quantization_parameters": {
    "precision_level": 0,
    "finetune_level": 1,
    "analysis_enable": true
  }
}
```

## 3. Запустите

```bash
xslim --config config.json
```

Ожидается INT8 Q/DQ ONNX в `output` и аналитические отчёты. Сохраните SHA-256
исходной/готовой модели, версию XSlim, конфигурацию, команду и SHA списка.

Если XSlim обнаружил Q/DQ во входной модели, используйте исходный FP32 export.
Нельзя удалять Q/DQ вручную только ради повторного PTQ.

## 4. Проверьте результат

```bash
python - <<'PY'
import onnx

model = onnx.load("output/model_s8_qdq.onnx")
onnx.checker.check_model(model)
onnx.shape_inference.infer_shapes(model)
print("pass")
PY
```

После этого выполните fixed fixtures и полную оценку задачи одним runner для
FP32 и INT8. Для K1X отдельно докажите EP placement на плате.
