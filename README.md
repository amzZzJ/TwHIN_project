# TwHIN — FaveGraph preprocessing

Препроцессинг датасета **TwitterFaveGraph** в единый формат триплетов для обучения
графовых эмбеддингов.

## Что делает код
Ноутбук `notebooks/favegraph.ipynb`:
1. загружает данные из `data/raw/TwitterFaveGraph.csv.zip`;
2. берет подвыборку по пользователям;
3. дедуплицирует ребра, строит глобальные сквозные ID для всех сущностей;
4. делает временной train/val/test сплит по `time_chunk`, без утечки (из val/test
   убираются сущности, которых нет в train);
5. считает EDA (активность по времени, распределения степеней);
6. пишет результат в `data/processed/favegraph_<вариант подвыборки>/`.

## Исходные данные
- Источник: HuggingFace — **`Twitter/TwitterFaveGraph`**
  (https://huggingface.co/datasets/Twitter/TwitterFaveGraph)
- Архив `TwitterFaveGraph.csv.zip` нужно положить в папку `data/raw/`

## Установка
```bash
git clone git@github.com:amzZzJ/TwHIN_project.git
cd TwHIN_project
python3 -m venv .venv && source .venv/bin/activate
python -m pip install polars pyarrow huggingface_hub matplotlib ipykernel ipywidgets
```
В VS Code открыть `notebooks/favegraph.ipynb` и выбрать kernel из `.venv`.

## Запуск

Настройки в ячейке конфига:

| параметр | смысл | по умолчанию |
|---|---|---|
| `USER_FRAC` | доля юзеров в подвыборке (`None` = весь граф) | `0.02` |
| `TRAIN_MAX` / `VAL_MAX` | границы временного сплита по `time_chunk` (1..192) | `180` / `186` |
| `SEED` | сид подвыборки | `42` |

Каждый `USER_FRAC` пишется в свою папку (`favegraph_frac0.02`, `favegraph_full`, …)

## Что на выходе
Папка `data/processed/favegraph_<вариант подвыборки>/`:

**`triples.parquet`** — ребра графа:

| колонка | тип | смысл |
|---|---|---|
| `lhs` | int64 | глобальный ID source-сущности |
| `rel` | int32 | ID типа отношения |
| `rhs` | int64 | глобальный ID target-сущности |
| `split` | str | `train` / `val` / `test` |

**`entities.parquet`** — словарь сущностей:

| колонка | тип | смысл |
|---|---|---|
| `entity_id` | int64 | глобальный ID (как в триплетах) |
| `entity_type` | str | `user` / `tweet` |
| `orig_id` | str | исходный ID из датасета |

**`relations.parquet`** — словарь отношений (`rel` int32 -> `rel_name` str; здесь только `fave`).

**`split_report.json`** — счетчики сплита и параметры прогона.

### Соглашения
- ID **глобальные и сквозные**: одно пространство для users и tweets, коллизий нет.
- Ребра **направленные**.
- Дубликаты ребер удалены; при повторах берется ранний `time_chunk`.
