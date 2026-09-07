# XSlim 2.1.2+riscy.2.1: локальная maintenance-версия

Это неофициальный downstream RISCY-SVT на базе исходного кода XSlim 2.1.2.
Исправления maintenance существуют локально; push, новый тег и релиз не
выполнялись. Опубликованный riscy.2 и замороженные B2/C2 сохранены.

XSlim выполняет офлайн-квантование ONNX-моделей: статический INT8 Q/DQ,
динамический INT8 и преобразование FP16. Downstream-релиз добавляет локальные
ограниченные диапазоны, детерминированное адаптивное округление и структурную
проверку signed-S8 split-контракта для K1X SpaceMIT.

## Установка

Проверено исполнением: CPython 3.12.3, Ubuntu 24.04, Linux x86_64, CPU.
Метаданные допускают >=3.12.3,<3.13; другие patch-версии отдельно не
сертифицированы. Старое обещание Python 3.9 неверно: ONNX требует >=3.10,
а принятый NumPy 2.5.2 требует >=3.12. Полный фиксированный набор зависимостей
и установка sdist описаны в [INSTALL.md](../../INSTALL.md).

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --constraint requirements-certified-python312.txt --extra-index-url https://download.pytorch.org/whl/cpu ./xslim-2.1.2+riscy.2.1-py3-none-any.whl
xslim --version
```

Ожидаемый результат: `xslim 2.1.2+riscy.2.1`. Wheel берётся из локального
handoff, а не из ещё не существующего нового remote-релиза.

Если установилась другая версия, удалите её и установите скачанный wheel по
явному пути. Команда `pip install xslim` не является способом установки этого
fork.

## Минимальный запуск

Подготовьте FP32 ONNX, список калибровочных изображений и JSON-конфигурацию из
[QUICKSTART.md](QUICKSTART.md), затем:

Это общий рецепт с пользовательскими входами, а не точное воспроизведение
B2/C2. В maintenance он не выполнялся. Исполняемый пример без датасета
находится в [RECONSTRUCTION_GUIDE.md](../RECONSTRUCTION_GUIDE.md).

```bash
xslim --config config.json
```

После генерации обязательно отдельно проверьте ONNX-граф, фиксированные
примеры, метрики задачи и размещение на целевом runtime. Успешный парсинг модели
не доказывает выполнение на SpaceMIT EP.

## Проверенный YOLO26-путь

Для K1X подтверждён split-контракт с шестью bbox/confidence-выходами и точным
float CPU tail. Поддержка YoloDecode в исходниках присутствует, но проверенные
B2/C2 split-графы её не используют. Исторический direct-E2E вариант дал
коллапс score на 100/100 изображениях.

Практический маршрут описан в
[K1X_YOLO26_COOKBOOK.md](K1X_YOLO26_COOKBOOK.md).

## Важная граница reconstruction

Релиз содержит **BRECQ-inspired layer-local adaptive rounding
infrastructure**. В YOLO-кампании проверены семь локальных single-Conv целей.
Полный BRECQ для C2f/residual-блоков, whole-head reconstruction, QDrop,
task-loss reconstruction и QAT не подтверждены.

## Документация

- [Текущие исправления и пределы проверки](../MAINTENANCE_ERRATA.md)
- [Быстрый старт](QUICKSTART.md)
- [Cookbook K1X YOLO26](K1X_YOLO26_COOKBOOK.md)
- [Устранение проблем](TROUBLESHOOTING.md)
- [Полное руководство на английском](../USER_GUIDE.md)
- [Описание конфигурации](../CONFIG_REFERENCE.md)
- [Ограничения](../LIMITATIONS.md)
- [Релиз и происхождение](../RELEASE_AND_PROVENANCE.md)

B2 остаётся универсальным контролем и откатом. C2 разрешён существующим
TIER-1 waiver только как отдельный замороженный higher-AP профиль; исторический
универсальный FAIL сохранён. Перед default приложения нужен собственный
score threshold C2. Нового waiver или продвижения runtime здесь нет.
