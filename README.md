# TwHIN: воспроизведение на открытых данных

Учебный проект (НИУ ВШЭ): разбор, критика и частичное воспроизведение статьи
**TwHIN — Embedding the Twitter Heterogeneous Information Network for Personalized
Recommendation** (El-Kishky et al., KDD 2022) на открытых данных.

Оригинальная модель обучена на закрытом графе Twitter (>10⁹ узлов, >10¹¹ рёбер),
поэтому прямое воспроизведение невозможно. Мы воспроизводим саму идею метода на
публичных датасетах меньшего масштаба и проверяем ключевые тезисы статьи.

## Что воспроизводится

- **Модель:** TransE со скорингом-дотом `f(s,r,t) = (θ_s + θ_r)·θ_t`,
  обучение через negative sampling (формула 2), оптимизатор Adagrad.
- **Mixture-of-embeddings**: пользователь как смесь кластеров его интересов.
- **Лоссы:** negative sampling, а также sampled-softmax и sampled-softmax + logQ-коррекция.
- **Negative sampling:** порча source/target сущностями того же типа; режимы uniform и
  пропорционально частоте (frequency).

## Исследовательские вопросы

1. Воспроизводится ли эффект гетерогенности (совместное обучение разных типов рёбер) на открытых данных?
2. Какие типы рёбер вносят наибольший вклад?
3. Выигрывает ли современный лосс (sampled-softmax + logQ) у классического negative sampling?

## Датасеты

Два датасета закрывают разные оси экспериментов.

### Yelp Open Dataset — гетерогенность
Гетерогенный граф (один штат, PA): сущности `user` / `business`, отношения
`review` (high-coverage), `tip` (low-coverage), `friend` (high-coverage).
Используется для проверки эффекта гетерогенности и mixture-of-embeddings.

### TwitterFaveGraph — лоссы и негативы
Открытый граф лайков `user → tweet` (HuggingFace `Twitter/TwitterFaveGraph`),
~283M рёбер; берётся подвыборка 2% пользователей. Однореляционный, поэтому используется
не для гетерогенности, а для сравнения функций потерь и режимов негативного сэмплирования.

## Структура репозитория

```
data/processed/
  favegraph_frac0.02/      обработанный FaveGraph
  yelp/                    обработанный Yelp

notebooks/
  favegraph.ipynb              обработка датасета FaveGraph -> граф
  favegraph_twhin_done.ipynb   обучение и оценка на FaveGraph
  yelp-d.ipynb                 обработка датасета Yelp -> гетерограф
  yelp_twhin_normalized.ipynb  обучение и оценка на Yelp (гетерогенность, ablation, mixture)
```

## Пайплайн

Для каждого датасета: **обработка** (ноутбук-препроцессинг строит граф из исходных
данных и сохраняет триплеты/словари) -> **обучение и оценка** (ноутбук обучает TransE
и считает downstream-метрики).

- **FaveGraph:** `favegraph.ipynb` -> `favegraph_twhin_done.ipynb`
- **Yelp:** `yelp-d.ipynb` -> `yelp_twhin_normalized.ipynb`

## Форматы данных

**`triples.parquet`** — рёбра графа:

| колонка | тип | смысл |
|---|---|---|
| `lhs` | int64 | глобальный ID source-сущности |
| `rel` | int32 | ID типа отношения |
| `rhs` | int64 | глобальный ID target-сущности |
| `split` | str | `train` / `val` / `test` |

**`entities.parquet`** — словарь сущностей (`entity_id`, `entity_type`, `original_id`).
ID глобальные и сквозные: единое пространство для всех типов сущностей, без коллизий.

**`relations.parquet`** — словарь отношений.

## Метрики

Оценка на замороженных эмбеддингах:

- **Candidate generation** (аналог Who-to-Follow): Recall@10/20/50, MRR; сравнение
  unimodal vs mixture; бейзлайны random и most-popular.
- **Engagement ranking** (аналог ad ranking): ROC-AUC, PR-AUC, RCE.
- **Контент-классификация** (аналог offensive detection, на Yelp — метка `useful`): PR-AUC, ROC-AUC.

## Запуск

```bash
git clone git@github.com:amzZzJ/TwHIN_project.git
cd TwHIN_project
python3 -m venv .venv && source .venv/bin/activate
python -m pip install torch pandas pyarrow scikit-learn faiss-cpu tqdm matplotlib huggingface_hub ipykernel
```

Открыть нужный ноутбук, при необходимости поправить пути в ячейке конфига (раздел 0),
запустить сверху вниз. Обучение рассчитано на одну GPU (~12 ГБ VRAM).

## Команда

Никита Шириков, Анна Попова, Амина Джалилова - НИУ ВШЭ.
