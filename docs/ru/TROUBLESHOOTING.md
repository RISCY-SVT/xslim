# Устранение проблем

## Не найдена модель или calibration list

Проверьте рабочую директорию и читаемость файлов:

```bash
test -r model.onnx
test -r calibration.txt
```

В переносимых примерах используйте относительные placeholders; в приватной
runtime-конфигурации допустимы абсолютные пути.

## Не совпадает input name/shape/dtype

Выведите реальные входы ONNX и перенесите их в config без догадок. Особенно
проверьте layout и конкретные размеры dynamic axes.

## Score collapse

Остановите оценку кандидата. Проверьте preprocessing, порядок шести выходов,
tail/decode, finite values, qparams и Q/DQ topology. Не компенсируйте collapse
понижением threshold.

## Profile pass, но EP fallback

Структурный profile не управляет runtime. Проверьте точные ORT/core/EP bytes,
provider options и профиль placement на плате. Изменённые qparams могут влиять
на compilation route даже при одинаковой topology.

## Strict selector ничего не нашёл

Это корректный fail-closed результат. Получите точные post-fusion tensor names
из source/QDQ mapping. Не отключайте strict mode, чтобы скрыть устаревший config.

## Ограничения диапазона невыполнимы

INT8-домен не может одновременно представить запрошенный диапазон и code
budgets. Пересмотрите физическое требование и данные, но не ослабляйте правило
после просмотра итоговых метрик.

## Не прошла проверка checksum

Не устанавливайте файл. Скачайте asset заново и сравните имя, размер и SHA-256
между GitHub и GitLab release.
