# Классификатор овощей по фото

6 классов: Capsicum, Carrot, Cucumber, Potato, Raddish, Squash. MobileNetV3-Small, трек A.
Выводы и метрики: [REPORT.md](REPORT.md). Разбор данных: [reports/eda/FINDINGS.md](reports/eda/FINDINGS.md).

## Установка
Python 3.12. Точные версии библиотек заданы в `requirements.txt` (PyTorch 2.15.0, torchvision 0.29.2, CPU).

```
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Проверено на Windows 11. На других системах не проверялось. Если `pip` не принимает сертификат антивируса, обновите его: в новых версиях используется хранилище сертификатов системы.

## Предсказание
Веса лежат в `weights/model.pt`. Работает на CPU и без интернета.

```
python predict.py --input путь/к/папке --output preds.csv
```

Форматы: jpg, jpeg, png, webp, bmp. Нечитаемые файлы и вложенные папки пропускаются с предупреждением и в CSV не попадают. HEIC не поддерживается. `unknown` не реализован.

## Обучение с нуля
1. Скачать [датасет](https://www.kaggle.com/datasets/preethi866/vegetable-dataset) (Apache 2.0) как zip.
2. Распаковать, проанализировать и разбить:

```
python src/split_list.py путь/к/archive.zip
python -m src.eda
python -m src.eda_report
python -m src.splits
```

3. Обучить (все 6 запусков заняли около 90 минут на i5-1135G7; для обучения нужен интернет, чтобы один раз скачать веса ImageNet):

```
python -m src.train --split group --fold 0
python -m src.train --split group --fold 1
python -m src.train --split random --fold 0
python -m src.train --split random --fold 1
python -m src.train --split author
python -m src.train --split full
```

4. Метрики, матрицы ошибок, Grad-CAM и скорость:

```
python -m src.evaluate
python -m src.benchmark
```

Seed 42 задан везде; на другом железе или GPU цифры могут немного отличаться. Предсказания наших запусков лежат в `runs/`, поэтому `src.evaluate` воспроизводит таблицы без переобучения; для Grad-CAM нужны фото из `data/Dataset`.

## Тесты
```
python -m pytest
```
