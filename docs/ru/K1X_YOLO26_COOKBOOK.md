# Cookbook: YOLO26 на K1X

## Назначение

Проверенный маршрут использует квантованную split-модель с шестью выходами и
отдельный точный float tail. Модели, датасеты и vendor runtime в релиз не входят.

## Контракт

```text
input: images, float32, 1x3x640x640
outputs:
  P3 bbox
  P3 confidence
  P4 bbox
  P4 confidence
  P5 bbox
  P5 confidence
tail: независимо захэшированная float ONNX-модель
```

Используйте реальные имена из проверенного source/QDQ mapping. Не угадывайте их
по форме тензора или названию архитектурного блока.

## Калибровка

1. Создайте детерминированный список репрезентативных изображений.
2. Исключите пересечение с candidate-selection и final-evaluation поверхностями.
3. Зафиксируйте SHA-256 списка и preprocessing.
4. Внесите точные шесть `truncate_var_names` в конфигурацию.

Санитизированный пример:

```bash
cp samples/k1x_yolo26_split/config_stage64_repro.json config.json
xslim --config config.json
```

Замените только placeholders и точные tensor names вашего графа.

## Структурная проверка

Требуются signed INT8 Q/DQ, отсутствие QLinear/UINT8/FP16, per-tensor activation,
per-channel symmetric Conv weights, явный `kernel_shape`, точные шесть выходов
и неизменный float tail.

```bash
xslim-spacemit-profile-check --help
xslim-qdq-boundary-audit --help
```

Profile pass не доказывает EP placement. На K1X нужно отдельно проверить
provider partition, correctness, метрики, latency и stability.

## Оценка и threshold

Сравните FP32 и INT8 одним preprocessing/decode/evaluator. Публикуйте mAP,
AP/AR по размерам и классам, prediction count, failures и uncertainty.

Порог score выбирается отдельно по TP/FP/FN и стоимости ошибок приложения.
Более высокий mAP не гарантирует более высокий recall при конкретном пороге.

## Direct E2E

YoloDecode есть в исходниках, но B2/C2 split-графы его не используют.
Исторический direct-E2E YOLO26 дал score collapse на 100/100 изображениях.
Рекомендация: six-output split + exact float tail.
