"""Build the README research report and a paginated PDF from measured outputs."""
from html import escape
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, Preformatted

from .common import ROOT, read_json

NAMES = {"popularity": "Популярность", "recency": "Недавние треки", "base": "GPT-2",
         "sft_control": "SFT-контроль", "completion_weak_kl": "Доля / слабый KL",
         "completion_strong_kl": "Доля / сильный KL", "listening_time": "Время", "balanced": "Баланс"}


def md_table(headers, rows):
    return "| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers) + " |\n" + "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows)


def main():
    cfg = read_json(ROOT / "configs/main.json")
    audit = read_json(ROOT / "data/processed/audit.json")
    agg = read_json(ROOT / "results/aggregate.json")
    rm = read_json(ROOT / "results/reward_metrics.json")
    env = read_json(ROOT / "results/environment.json")
    seed_summaries = [read_json(ROOT / f"results/seed_{s}/summary.json") for s in cfg["seeds"]]
    names = ["popularity", "recency", "base", *cfg["variants"]]

    def m(name, key):
        return agg["models"][name]["test"][key]["mean"]

    def sd(name, key):
        return agg["models"][name]["test"][key]["std"]

    def roll(name, key):
        return agg["models"][name]["rollout"][key]["mean"]

    def ms(name, key, digits=5):
        return f"{m(name, key):.{digits}f} ± {sd(name, key):.{digits}f}"

    def interval_row(name, control="base"):
        key = "paired_bootstrap_vs_base" if control == "base" else "paired_bootstrap_vs_sft_control"
        d = agg[key][name]
        return [NAMES[name], *[f"{d[k]['mean_delta']:+.5f} [{d[k]['low_95']:+.5f}; {d[k]['high_95']:+.5f}]" for k in ["ndcg@10", "predicted_completion@10"]]]

    retrieval_rows = [[NAMES[n], ms(n, "recall@10"), ms(n, "ndcg@10"), f"{m(n, 'catalog_ndcg@10'):.5f}"] for n in names]
    reward_rows = [[NAMES[n], ms(n, "predicted_completion@10", 4), f"{m(n, 'history_repeat@10'):.3f}", f"{m(n, 'catalog_coverage@10'):.3f}"] for n in names]
    behavior_rows = [[NAMES[n], f"{m(n, 'sampled_completion'):.4f}", f"{m(n, 'sampled_history_repeat'):.3f}",
                      f"{m(n, 'top1_max_share'):.3f}", f"{m(n, 'mean_duration@10_seconds'):.1f}"] for n in ["base", *cfg["variants"]]]
    rollout_rows = [[NAMES[n], f"{roll(n, 'discounted_model_return'):.3f}", f"{roll(n, 'within_slate_duplicate_fraction'):.3f}",
                    f"{roll(n, 'original_history_repeat_fraction'):.3f}", f"{roll(n, 'unique_generated_items'):.1f}"] for n in ["base", *cfg["variants"]]]
    rmrows = []
    for key, label in [("global_mean", "Среднее"), ("item_mean_shrunk", "Среднее по треку"), ("neural", "История + трек"), ("neural_shuffled_history", "Перемешанные истории")]:
        v, t = rm["splits"]["validation"][key], rm["splits"]["test"][key]
        rmrows.append([label, f"{v['mse']:.5f}", f"{t['mse']:.5f}", f"{t['mae']:.5f}", f"{t['auc_completion_ge_90pct']:.3f}"])
    ci_rows = [interval_row(n) for n in cfg["variants"]]
    ci_sft_rows = [interval_row(n, "sft_control") for n in cfg["variants"] if n != "sft_control"]
    base_steps = [read_json(ROOT / f"results/seed_{s}/base_selection.json")["step"] for s in cfg["seeds"]]
    train_seconds = sum(s[n]["train_seconds"] for s in seed_summaries for n in ["base", *cfg["variants"]]) + rm["train_seconds"]
    weak_delta = m("completion_weak_kl", "predicted_completion@10") - m("base", "predicted_completion@10")
    best_relevance = max(["base", *cfg["variants"]], key=lambda n: m(n, "ndcg@10"))
    coverage = audit["catalog_event_coverage_by_split"][2]
    summary_text = (
        f"На Yambda-50M обучена компактная GPT-2 ({seed_summaries[0]['base']['parameters']:,} параметра), "
        f"отдельная модель доли прослушивания и четыре RL-варианта. Проведены три запуска (17, 29, 43), "
        f"добавлен контроль продолжения обычного обучения. Награда за долю со слабым KL изменила "
        f"предсказанную долю в Top-10 на {weak_delta:+.4f}. Это изменение суррогата, а не измерение реальной пользы. "
        f"Лучший средний NDCG@10 среди нейросетевых вариантов: {NAMES[best_relevance]} ({m(best_relevance, 'ndcg@10'):.5f}); "
        f"недавние треки дают {m('recency', 'ndcg@10'):.5f}. Нейросетевая модель отклика уступает "
        f"среднему по треку. Эти ограничения не позволяют объявить RL улучшением пользовательского опыта."
    )
    conclusions = [
        f"GPT-2 превосходит популярностный ориентир по NDCG@10 ({m('base', 'ndcg@10'):.5f} против {m('popularity', 'ndcg@10'):.5f}), но уступает недавним трекам ({m('recency', 'ndcg@10'):.5f}). Это сильная и необходимая проверка в музыке с повторными прослушиваниями.",
        "Рост предсказанной награды и рост точности - разные результаты. Изменения NDCG оцениваются вместе с парными интервалами, а не только по знаку среднего.",
        "SFT-контроль проверяет вклад продолжения обучения на прежних next-item целях. Для вариантов Время и Баланс совпадает также beta=0.1; у двух completion-вариантов beta отличается, поэтому их сравнение с этим контролем не изолирует только reward-градиент.",
        "Модель отклика использует историю: перемешивание историй ухудшает MSE. При этом простой сглаженный приор по треку лучше по тестовому MSE; нейросетевой симулятор слишком слаб для выводов о долгосрочной пользе.",
        "Изменение функции награды меняет состав рекомендаций. Штраф за повтор применяется к сэмплированному действию и не обязан уменьшать долю повторов в детерминированном Top-10.",
        f"Каталог ограничен 2048 популярными обучающими треками и покрывает {coverage:.1%} всех тестовых событий. Основные метрики учитывают OOV как промахи; условные метрики по каталогу показаны отдельно.",
        "Пятишаговое продолжение - диагностика гипотетических рекомендаций при предположении их потребления, а не восстановленная реакция реального пользователя. Удержание и накопленная реальная полезность не оценены.",
    ]
    limitations = [
        "Ограниченный каталог и короткий контекст: 2048 из 739583 обучающих треков, 32 последних события. OOV остаётся в истории отдельным токеном, поэтому нельзя интерпретировать результаты как полноразмерную систему.",
        "Разбиение глобальное по времени, а не leave-one-out по каждому пользователю. Пользователи могут встречаться в разных периодах. В тестовом контексте разрешены уже произошедшие validation/test взаимодействия: это rolling-history оценка без обновления весов.",
        "Сэмплирование: до 40 известных обучающих целей на пользователя; по одной случайной цели на пользователя в каждом отложенном периоде. Пользователи имеют равный вес в оценке. Это не все события и не протокол официального бенчмарка Yambda.",
        "Неопределённый порядок событий с одинаковым временем устранён исключением всех таких событий из контекста цели. Источник содержит 8840469 соседних совпадений времени. Внутренние единицы времени не переводятся в календарные даты.",
        "Доля больше 100% в исходных данных отражает повторное прослушивание. Она намеренно ограничена единицей в целевой функции; информация о повторном времени при этом теряется. Временная награда использует среднюю обучающую длительность трека, ограниченную 600 секундами.",
        "Отклик доступен только для фактически прослушанных треков. Пропенсити логирующей политики нет; IPS/DR и причинная оценка не выполнялись. Ненаблюдаемые рекомендации не объявляются отрицательными примерами.",
        "RL здесь - одноступенчатый contextual bandit на фиксированных реальных контекстах с замороженной reward model. Многошаговый RL, PPO и поправка off-policy из статьи 2607.02818 не реализованы и не заявляются.",
        "Три seed относятся к политике; модель отклика одна. Bootstrap по пользователям условен на этой модели, датасете и каталоге. Он не покрывает все источники неопределённости и не корректирует множественные сравнения.",
        "Число шагов и коэффициенты зафиксированы в конфигурации. После тестового сравнения гиперпараметры не подбирались. Размеры модели ограничены CPU, качество базы не доведено до уровня сильных полноразмерных рекомендателей.",
    ]
    commands = """python -m venv .venv
# PowerShell / Windows:
.\\.venv\\Scripts\\Activate.ps1
# Linux/macOS instead: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pytest -q
python -m recsys.verify --require-results
python -m recsys.recommend --demo 0 --k 10"""
    reproduction = """# Reproduce all training runs from the included prepared data:
python run_all.py

# Regenerate the prepared data from the pinned 426 MB source:
python -m recsys.data --download

# Individual stages:
python -m recsys.experiment
python -m recsys.analysis
python -m recsys.report"""
    refs = [
        ("Yandex Yambda: dataset card, schema, license, timestamp ordering and replay interpretation", "https://huggingface.co/datasets/yandex/yambda/blob/main/README.md"),
        ("Petrov, Macdonald. Aligning GPTRec with Beyond-Accuracy Goals with Reinforcement Learning (2024)", "https://arxiv.org/abs/2403.04875"),
        ("Matveev et al. Session-Level Optimization for Large-Scale Retrieval using REINFORCE with Multi-Step Off-Policy Correction (2026)", "https://arxiv.org/abs/2607.02818"),
        ("Hugging Face Transformers: GPT-2 model documentation", "https://huggingface.co/docs/transformers/model_doc/gpt2"),
    ]
    readme = f"""# RL-дообучение GPT-2 для музыкальных рекомендаций

Даниил Унковский · Т-Образование · 25 октября 2026

**Статус: эксперимент выполнен.** В репозитории есть код, подготовленные данные, веса,
сырые метрики, журналы обучения, графики и PDF. Это исследовательский прототип на
ограниченном каталоге, не промышленный рекомендатель и не полное воспроизведение статей.

## Основной результат

{summary_text}

![Сравнение вариантов](results/figures/main_comparison.png)

## 1. Постановка и гипотезы

По прошлой истории прослушиваний предсказывается следующий трек. После supervised
обучения GPT-2 политика дообучается на предсказанном отклике. Проверяются три гипотезы:
(1) награда меняет распределение рекомендаций; (2) рост суррогатной награды не обязан
повышать релевантность; (3) регуляризация и выбор reward меняют компромисс.

Токены - идентификаторы треков; текстовый токенизатор и предобученные текстовые веса
не используются. `GPT2LMHeadModel` из Transformers создаётся со случайными весами.

## 2. Данные, аудит и отсутствие утечек

Источник: Yandex Yambda-50M, `sequential/50m/listens.parquet`, {audit['bytes']:,} байт.
SHA-256: `{audit['sha256']}`. Ревизия репозитория: `dd6f3a19eef5866e346c3270e098baa641a44948`.
Лицензия исходных данных: Apache-2.0; условия и изменения перечислены в DATA_LICENSE.md.

Фактически прочитаны **{audit['users']:,} пользователей и {audit['events']:,} прослушиваний**.
В конкретном файле 9238 строк; общие 10000 пользователей из карточки относятся к
версии датасета в целом и не подменяют измеренный размер файла listens.
Использованы `uid`, `timestamp`, `item_id`, `played_ratio_pct`, `track_length_seconds`, `is_organic`.

Глобальные временные границы заданы по 80/10/10% длительности наблюдений:

{md_table(['Период', 'Условие timestamp', 'Все события', 'События каталога', 'Покрытие'], [[n, cond, f'{audit["events_by_split"][i]:,}', f'{audit["known_catalog_events_by_split"][i]:,}', f'{audit["catalog_event_coverage_by_split"][i]:.2%}'] for i,(n,cond) in enumerate([('Train','t < 20800000'),('Validation','20800000 ≤ t < 23400000'),('Test','23400000 ≤ t ≤ 26000000')])])}

Каталог - top-2048 треков только по частотам train. PAD=0, OOV=1, треки=2..2049.
Ни частоты, ни длительности, ни параметры моделей не обучаются на validation/test.
Подготовлено {audit['train_examples']:,} обучающих контекстов и по 6000 отложенных запросов;
в validation известны {audit['validation_known_targets']} целей, в test - {audit['test_known_targets']}.
Внешние цели сохраняются как OOV и считаются промахами в основных retrieval-метриках.

Контекст содержит последние 32 **строго более ранних** события. Все события с тем же
timestamp, что и цель, исключаются. OOV-события сохраняются в истории: удаление могло
бы создать ложную смежность треков. Аудит обнаружил {audit['equal_adjacent_timestamps']:,}
соседних совпадений timestamp, {audit['ratio_above_100']:,} долей выше 100%, 0
неупорядоченных пользователей и 0 нулевых длительностей.

Доля прослушивания: `y = min(played_ratio_pct / 100, 1)`. Значения выше 100% допустимы
в источнике и здесь ограничиваются как осознанное определение цели, а не очистка шума.

## 3. Модели и обучение

**GPT-2:** 2 слоя, 4 головы, embedding=64, контекст=32, dropout=0.1,
{seed_summaries[0]['base']['parameters']:,} параметра. На последнем наблюдаемом состоянии
вычисляется full-catalog cross entropy по следующему треку. Выходные PAD/OOV исключены.
AdamW, lr=0.001, batch=128, 1800 обновлений, warmup=100, cosine decay до 10% lr,
weight decay=0.01, clip norm=1. Checkpoint выбирается по validation NDCG@10 каждые 300
шагов. Выбранные шаги для seeds 17/29/43: {base_steps}. Оценка test не используется для выбора.

**Модель отклика:** независимые embedding=32; среднее истории, последний трек,
кандидат и произведение среднего с кандидатом подаются в MLP 128→64→32→1, sigmoid.
MSE по наблюдаемой доле прослушивания. AdamW lr=0.001, batch=256, 1400 обновлений;
выбор по validation MSE каждые 200 шагов. Затем веса заморожены во всех RL-запусках.
История и кандидат являются входом; `uid`, время и фактическая будущая доля не подаются.

### Качество модели отклика на известных целях

{md_table(['Модель', 'Val MSE', 'Test MSE', 'Test MAE', 'AUC доля ≥0.9'], rmrows)}

Сглаженный средний отклик по треку: `(sum_y + 20 * global_mean) / (count + 20)`.
Эти простые ориентиры используют все известные train-события; нейросеть - выбранные
контексты. Neural Spearman на test: {rm['splits']['test']['neural']['spearman']:.4f}.
Перемешивание историй сохраняет кандидатов и цели, но разрушает соответствие истории:
это проверка зависимости модели от контекста, а не самостоятельная оценка причинности.

![Модель отклика](results/figures/reward_validation.png)

## 4. RL: contextual-bandit REINFORCE

Состояние `s` - реальный прошлый контекст train; действие `a` - сэмплированный трек;
награждает замороженная модель `r_hat(s,a)`. Нет пропенсити логирования, поэтому не
заявляется корректная off-policy оценка. RL выполняет on-policy sampling внутри
**обученного суррогатного окружения**, а не внутри реального сервиса.

На контекст независимо с возвращением сэмплируются четыре действия. Для действия j
baseline равен среднему reward остальных трёх действий, `A_j = r_j - mean(r_other)`.
Baseline независим от собственного действия при данном контексте.

`L = -mean(A * log π(a|s)) + β KL(π || π_base) + 0.1 CE(π, logged_next_item)`.

KL рассчитывается точно по всем 2048 действиям. Reward и advantage отсоединены от
графа градиентов. Dropout в RL выключен, чтобы распределение сэмплирования совпадало
с дифференцируемым. AdamW lr=0.00008, 400 шагов, batch=128, clip=1, без weight decay.
Последний checkpoint фиксированного шага сохраняется без подбора по test.
Каждый вариант начинается с одинаковой базы соответствующего seed и получает один
и тот же поток train-контекстов и начальный RNG для сэмплирования.

{md_table(['Вариант', 'Reward', 'β KL', 'REINFORCE'], [['SFT-контроль','нет reward-градиента; только KL + 0.1 CE','0.1','выключен'],['Доля / слабый KL','r_hat','0.02','включён'],['Доля / сильный KL','r_hat','0.2','включён'],['Время','r_hat * min(train_mean_duration,600) / 600','0.1','включён'],['Баланс','r_hat + 0.15 novelty - 0.25 recent_repeat','0.1','включён']])}

`novelty` - нормированная отрицательная log-частота по train, а не семантическое
разнообразие. `recent_repeat`=1, если действие есть в последних 32 событиях.
SFT-контроль получает то же число дополнительных шагов и ту же CE/KL часть, но
нулевой коэффициент REINFORCE. Он необходим для отделения эффекта RL от extra training.
Его beta=0.1 совпадает с вариантами Время и Баланс. Для completion beta=0.02/0.2,
поэтому разница с SFT-контролем включает также изменение силы KL и не является
чистой абляцией одного reward-градиента.

## 5. Протокол и основные результаты

Три policy seed: 17, 29, 43; модель отклика общая (seed=17). Все варианты ранжируют
весь одинаковый каталог из 2048 треков, без sampled negatives. Повторы не маскируются,
так как музыка допускает повторное потребление. Один отложенный next item на запрос:
Recall@K=HitRate@K; NDCG даёт вес `1/log2(rank+1)` при попадании. Также сохранены K=5/20.

Ниже среднее ± выборочное стандартное отклонение по трём policy seed. Основные
метрики считаются по всем 6000 запросам, включая OOV-промахи. `Catalog NDCG` условен
на {audit['test_known_targets']} целях в каталоге и не заменяет основной результат.

{md_table(['Модель', 'Recall@10', 'NDCG@10', 'Catalog NDCG@10'], retrieval_rows)}

{md_table(['Модель', 'Pred. completion@10', 'Повтор истории@10', 'Покрытие каталога@10'], reward_rows)}

Predicted completion@10 - среднее предсказание замороженной модели по Top-10,
а не наблюдаемое качество рекомендаций. Повтор истории - доля Top-10 из последних
32 событий; покрытие - доля каталога, появившаяся хотя бы в одном Top-10.

### Парные различия и неопределённость

На каждом тестовом пользователе метрика усредняется по трём seed. Затем 2000 раз
с возвращением сэмплируются пользователи, вычисляется парная разница. Показаны
95% percentile-интервалы. Это условная неопределённость по пользователям, не полная
неопределённость обучения; один reward model и каталог остаются фиксированными.

Разница с исходной GPT-2:

{md_table(['Вариант', 'Δ NDCG@10 [95% CI]', 'Δ Pred. completion@10 [95% CI]'], ci_rows)}

Разница с SFT-контролем (проверка добавочной пользы reward-градиента):

{md_table(['RL-вариант', 'Δ NDCG@10 [95% CI]', 'Δ Pred. completion@10 [95% CI]'], ci_sft_rows)}

## 6. Поведение, повторы и гипотетические продолжения

{md_table(['Модель', 'Sampled reward', 'Sampled recent repeat', 'Доля самого частого Top-1', 'Длительность Top-10, c'], behavior_rows)}

Sampled reward и repeat усреднены по 16 действиям на тестовый контекст с фиксированным
RNG. Это ближе к обучаемому стохастическому распределению, чем жёсткий Top-10.
При отсутствии предсказанного ответа реальные будущие истории не изменяются.

![Поведение](results/figures/behavior.png)

Для первых 512 тестовых контекстов генерируется пять жадных рекомендаций. Каждая
рекомендация добавляется в историю как **гипотетически потреблённая**, реальный отклик
не подменяется наблюдением. Рассчитан `sum(gamma^t * r_hat_t)`, gamma=0.9.
Результаты - модельная диагностика, не доказательство session return или удержания.

{md_table(['Модель', 'Model return', 'Повторы внутри 5', 'Из исходной истории', 'Различных треков'], rollout_rows)}

![Пятишаговая генерация](results/figures/rollout.png)

В `results/seed_*/..._evaluation.json` сохранены примеры raw item ID и предсказаний,
в том числе оценка с исходным контекстом без сгенерированного суффикса. Названий,
жанров и аудиопризнаков в этом эксперименте нет; семантические объяснения треков не выдумываются.

## 7. Выводы

""" + "\n".join(f"{i+1}. {t}" for i, t in enumerate(conclusions)) + """

## 8. Ограничения и дальнейшие проверки

""" + "\n\n".join(f"- {t}" for t in limitations) + f"""

Следующие эксперименты: увеличить каталог/контекст и качество next-item базы;
сравнить reward model с temporal cross-fitting и более сильными приорами; добавить
ансамбль/неопределённость reward, консервативное ограничение политики, несколько
инициализаций reward model. Долгосрочные прокси (следующее взаимодействие, длина
сессии, возврат после перерыва) требуют отдельного определения сессий и проверки
смешения временных факторов. Причинные выводы потребуют корректного OPE при
известных propensity или онлайн-эксперимента; по этим данным они не получены.

## 9. Связь с материалами задания

Работа 2403.04875 мотивирует поэлементную генерацию и оптимизацию целей сверх
точности; здесь реализован компактный next-item GPT-2 с двумя стадиями обучения,
но teacher-student стадия и семантическая diversity из статьи не воспроизводятся.
Работа 2607.02818 рассматривает session-level обучение и feedback model на Yambda;
здесь взята идея отдельной модели отклика, но используется GPT-2 вместо two-tower,
один шаг REINFORCE вместо multi-step off-policy correction и нет DR/OPE.
Это самостоятельный небольшой эксперимент по постановке задания, а не репликация
результатов перечисленных статей.

## 10. Воспроизведение и запуск

Проверенная среда: Python {env['python']}, PyTorch {env['torch']}, CPU, 4 потока,
Transformers {env['versions']['transformers']}. GPU не требовался. На этом CPU
суммарное время обучения (reward + 3 базы + 15 продолжений) составило
{train_seconds:.1f} сек.; сюда не включены скачивание, подготовка, тесты и построение отчёта.
Рекомендуется 4+ ГБ RAM для запуска готовых весов и 8+ ГБ для воспроизведения.
Python 3.12 выбран для воспроизводимости; старая учебная .venv с Python 3.14 не нужна.

Из корня клонированного или распакованного проекта:

```powershell
{commands}
```

Если PowerShell запрещает Activate.ps1, активация не обязательна: каждую команду
`python ...` можно запустить как `& .\\.venv\\Scripts\\python.exe ...`.

```bash
{reproduction}
```

`run_all.py` заново обучает и заменяет результаты в results; для сохранения исходного результата можно использовать отдельную копию репозитория. Для одного seed: `python -m recsys.experiment --seeds 17 --out results_one_seed`.
Для GPU при наличии подходящей сборки PyTorch: `python -m recsys.experiment --device cuda`;
этот вариант не проверялся, CPU-метрики могут не совпасть бит-в-бит.
Команды `analysis/report/recommend` по умолчанию читают основной каталог results.

Структура:

- `recsys/data.py`: скачивание с SHA-256, аудит, временное разбиение, подготовка.
- `recsys/models.py`: GPT-2 и отдельная reward model.
- `recsys/experiment.py`: обучение, REINFORCE, SFT-контроль, оценка.
- `recsys/evaluate.py`: full-catalog метрики и гипотетические rollout.
- `recsys/analysis.py`: агрегирование seed, bootstrap, графики, CSV.
- `recsys/recommend.py`: просмотр рекомендаций готового checkpoint.
- `recsys/verify.py`, `tests/`: проверки протокола, causal mask и градиента REINFORCE.
- `configs/main.json`: все размеры, seed и гиперпараметры.
- `data/processed/`: подготовленные примеры и отображение токенов.
- `results/`: веса `.pt`, сырые метрики, per-query `.npz`, логи, графики, environment.
- `report.pdf`: самостоятельный читаемый отчёт.

Большой исходный Parquet в репозиторий не включён. Подготовленные данные и веса позволяют
проверить результаты без повторного скачивания. Для полного воспроизведения подготовки
используется закреплённый источник и SHA-256. На CPU при фиксированном стеке ожидается
повторяемость; файлы NPZ могут иметь иные байтовые хеши из-за метаданных ZIP, поэтому
для сравнения повторно подготовленных данных проверяются массивы и аудит.

## Источники

""" + "\n".join(f"- [{title}]({url})" for title, url in refs) + "\n"
    (ROOT / "README.md").write_text(readme, encoding="utf-8")

    # PDF: fixed pages make the report readable independently from the code archive.
    font_dir = ROOT / "assets/fonts"
    if not font_dir.exists():
        # Optional override for environments not using the bundled fonts.
        import os
        font_dir = Path(os.environ.get("REPORT_FONT_DIR", str(font_dir)))
    for name, filename in [("DV", "DejaVuSans.ttf"), ("DV-Bold", "DejaVuSans-Bold.ttf"), ("DV-Mono", "DejaVuSansMono.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
    pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-Bold", italic="DV", boldItalic="DV-Bold")
    ink, blue, teal = colors.HexColor("#203646"), colors.HexColor("#193b59"), colors.HexColor("#148d82")
    styles = {
        "title": ParagraphStyle("title", fontName="DV-Bold", fontSize=25, leading=30, textColor=blue, spaceAfter=14),
        "subtitle": ParagraphStyle("sub", fontName="DV", fontSize=10, leading=15, textColor=colors.HexColor("#637583"), spaceAfter=14),
        "h1": ParagraphStyle("h1", fontName="DV-Bold", fontSize=19, leading=24, textColor=blue, spaceAfter=14),
        "h2": ParagraphStyle("h2", fontName="DV-Bold", fontSize=11, leading=15, textColor=teal, spaceBefore=8, spaceAfter=7),
        "body": ParagraphStyle("body", fontName="DV", fontSize=9.4, leading=14, textColor=ink, spaceAfter=8),
        "small": ParagraphStyle("small", fontName="DV", fontSize=8, leading=11.5, textColor=ink, spaceAfter=6),
        "cell": ParagraphStyle("cell", fontName="DV", fontSize=7.6, leading=10.3, textColor=ink),
        "headcell": ParagraphStyle("headcell", fontName="DV-Bold", fontSize=7.5, leading=10.5, textColor=colors.white),
        "code": ParagraphStyle("code", fontName="DV-Mono", fontSize=7.1, leading=10, textColor=ink, spaceAfter=9),
    }
    story = []

    def p(text, style="body"):
        # ASCII hyphens in the PDF to avoid uncommon dash glyph rendering issues.
        text = text.replace("–", "-").replace("—", "-").replace("‑", "-")
        story.append(Paragraph(escape(text).replace("\n", "<br/>"), styles[style]))

    def table(headers, rows, widths=None):
        data = [[Paragraph(escape(str(c)), styles["headcell"]) for c in headers]]
        data += [[Paragraph(escape(str(c)), styles["cell"]) for c in row] for row in rows]
        if widths is None:
            widths = [170 * mm / len(headers)] * len(headers)
        t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), blue),
                               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f5f7"), colors.white]),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                               ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                               ("LINEBELOW", (0, -1), (-1, -1), .5, colors.HexColor("#d7e3e8"))]))
        story.append(t)
        story.append(Spacer(1, 9))

    def fig(name, width=170 * mm, height=None):
        image = Image(str(ROOT / "results/figures" / name))
        if height is None:
            height = width * image.imageHeight / image.imageWidth
        image.drawWidth, image.drawHeight = width, height
        story.append(image)
        story.append(Spacer(1, 8))

    def page(title):
        story.append(PageBreak())
        p(title, "h1")

    p("GPT-2: рекомендации\nи обучение с наградой", "title")
    p("Даниил Унковский · Т-Образование · 25 октября 2026\nВоспроизводимый эксперимент на Yambda-50M", "subtitle")
    p(summary_text)
    fig("main_comparison.png")
    p("Что входит в результат", "h2")
    p("Готовый код и веса, подготовленные данные, временной аудит, три seed политики, независимая проверка отклика, четыре RL-варианта, SFT-контроль, full-catalog оценка, интервалы различий и пятишаговая диагностика.")
    p("Основные ограничения", "h2")
    p(f"Каталог покрывает {coverage:.1%} всех тестовых событий. Reward model уступает простому приору по треку. Награда и rollout рассчитаны моделью: реальные изменения пользовательского опыта и удержания не измерены.")

    page("01 / Данные и временной протокол")
    p(f"Yandex Yambda-50M, sequential/50m/listens.parquet. Измерено {audit['users']:,} пользователей и {audit['events']:,} прослушиваний. Исходный файл занимает 426,25 МБ; его контрольная сумма закреплена в коде и аудите. Лицензия данных - Apache-2.0.")
    table(["Период", "Граница timestamp", "События", "Покрытие каталога"], [[n, cond, f"{audit['events_by_split'][i]:,}", f"{audit['catalog_event_coverage_by_split'][i]:.2%}"] for i, (n,cond) in enumerate([("Train","< 20 800 000"),("Validation","20 800 000 ... 23 399 999"),("Test","23 400 000 ... 26 000 000")])], [25*mm,65*mm,40*mm,40*mm])
    p("Границы соответствуют 80/10/10% длительности наблюдений, а не числа событий. Внутренние временные единицы не превращаются в календарные даты. Все статистики каталога, длительностей и награды вычислены только по train.")
    table(["Подготовленные примеры", "Train", "Validation", "Test"], [["Число контекстов", f"{audit['train_examples']:,}", "6000", "6000"], ["Известные цели", f"{audit['train_known_targets']:,}", str(audit['validation_known_targets']), str(audit['test_known_targets'])]], [65*mm,35*mm,35*mm,35*mm])
    p("Правила построения", "h2")
    p("Каталог: 2048 самых частых train-треков из 739583. PAD=0; OOV=1. До 40 известных train-целей на пользователя; одна случайная цель на пользователя в каждом отложенном периоде, максимум 6000. Внешние цели в оценке считаются промахами.")
    p("История: последние 32 строго более ранних события. OOV сохраняется. Все события с timestamp цели исключаются; это исключает неопределённый порядок внутри временного бина. Для test используются уже наблюдённые взаимодействия, включая более ранние события отложенных периодов, без обновления весов.")
    p(f"Аудит: {audit['equal_adjacent_timestamps']:,} соседних совпадений времени; {audit['ratio_above_100']:,} долей выше 100%; ноль неупорядоченных историй и ноль нулевых длительностей. Значения выше 100% - допустимые повторы; цель completion намеренно ограничена единицей.")
    p("Защита от утечки проверяется по глобальным границам split, максимальному времени контекста, диапазонам токенов, causal mask GPT-2 и независимости тестовых целей от обучения.")

    page("02 / GPT-2 и модель отклика")
    p("Рекомендательная база", "h2")
    p(f"Официальная GPT2LMHeadModel из Transformers, случайная инициализация. 2 слоя, 4 головы, размер состояния 64, dropout 0.1, {seed_summaries[0]['base']['parameters']:,} параметра. Текстовые веса и токенизация не используются. Полная cross entropy по 2048 действиям на следующем треке.")
    p(f"AdamW lr=0.001, batch=128, 1800 шагов, warmup=100, cosine decay, weight decay=0.01, clip=1. Выбор по validation NDCG@10 каждые 300 шагов. Выбранные шаги для seed 17/29/43: {base_steps}.")
    p("Отклик: история + кандидат", "h2")
    p("Независимые embedding=32; среднее истории, последний трек, кандидат и их произведение; MLP 128→64→32→1 с sigmoid. Цель min(played_ratio_pct/100,1), MSE. AdamW lr=0.001, batch=256, 1400 шагов. Выбор по validation MSE, затем заморозка весов.")
    table(["Ориентир / модель", "Val MSE", "Test MSE", "Test MAE", "AUC ≥0.9"], rmrows, [56*mm,29*mm,29*mm,29*mm,27*mm])
    fig("reward_validation.png")
    p("Нейросеть чуть лучше постоянного среднего, но хуже среднего по треку со сглаживанием 20 наблюдениями. Перемешивание историй ухудшает качество: контекст используется, но этого недостаточно для надёжного симулятора. Baseline-приоры рассчитаны по всем известным train-событиям, нейросеть - по выбранным контекстам.", "small")

    page("03 / Дообучение с наградой")
    p("Постановка", "h2")
    p("Одноступенчатый contextual bandit: состояние - реальный train-контекст, действие - трек, награда - замороженный предиктор. Сэмплирование on-policy внутри суррогатного окружения. Это не онлайн-взаимодействие и не off-policy оценка реальных пользователей.")
    p("На контекст с возвращением сэмплируются четыре действия. Baseline для каждого равен среднему reward остальных трёх. Advantage отсоединён от градиента; тест проверяет ожидаемый градиент на полностью перечисленном двухдействийном примере.")
    p("L = -mean(A · log π(a|s)) + β KL(π || π_base) + 0.1 CE", "h2")
    p("KL рассчитывается по всему каталогу. CE сохраняет обучение на наблюдавшемся следующем треке. Dropout выключен при RL, чтобы сэмплируемая политика совпадала с дифференцируемой. Для всех вариантов: AdamW lr=0.00008, batch=128, 400 шагов, clip=1.")
    table(["Вариант", "Награда", "β"], [["SFT-контроль", "Только KL + 0.1 CE; REINFORCE=0", "0.1"], ["Доля / слабый KL", "Предсказанная доля", "0.02"], ["Доля / сильный KL", "Предсказанная доля", "0.2"], ["Время", "Доля × min(train_mean_duration,600)/600", "0.1"], ["Баланс", "Доля + 0.15 novelty - 0.25 recent_repeat", "0.1"]], [45*mm,105*mm,20*mm])
    p("novelty - нормированная отрицательная log-частота по train; recent_repeat показывает наличие трека в последних 32 событиях. Ни жанры, ни артисты, ни аудиопризнаки не используются; это не семантическая diversity.")
    p("Контроль и воспроизводимость", "h2")
    p("Все продолжения начинают с того же checkpoint соответствующего seed, с одинаковым потоком контекстов и начальным RNG. SFT-контроль имеет те же 400 обновлений и CE/KL часть, но без reward-градиента. После фиксированного числа шагов сохраняется последний checkpoint; test не используется для выбора.")

    page("04 / Релевантность и интервалы")
    p("Все модели ранжируют один полный каталог из 2048 треков. Sampled negatives не используются. Музыкальные повторы разрешены. Основные метрики включают OOV-промахи; Catalog NDCG рассчитан только по известным целям. Среднее ± std по трём seed.")
    table(["Модель", "Recall@10", "NDCG@10", "Catalog NDCG"], retrieval_rows, [44*mm,45*mm,45*mm,36*mm])
    p("Парная разница с исходной GPT-2: среднее [95% CI]", "h2")
    table(["Вариант", "Δ NDCG@10", "Δ предсказанной доли@10"], ci_rows, [44*mm,63*mm,63*mm])
    p("Для каждой цели метрика сначала усредняется по трём seed; затем 2000 раз с возвращением сэмплируются пользователи. Интервал условен на общем reward model, каталоге и данных. Наличие нуля внутри интервала не позволяет считать знак разницы надёжно установленным. Множественные сравнения не скорректированы.", "small")
    p("GPT-2 лучше популярностного ориентира, но уступает сильному для музыки правилу недавних треков. Низкие абсолютные метрики частично объясняются ограниченным каталогом, но это не оправдывает игнорирование recency baseline.")

    page("05 / Контроль, награда и поведение")
    p("Парная разница с продолжением обычного обучения", "h2")
    table(["RL-вариант", "Δ NDCG@10 [95% CI]", "Δ доли@10 [95% CI]"], ci_sft_rows, [44*mm,63*mm,63*mm])
    p("Изменение относительно базы смешивает влияние RL и дополнительных supervised шагов. SFT-контроль имеет beta=0.1, как Время и Баланс. Для completion beta отличается: сравнение с этим контролем не изолирует только reward-градиент.")
    table(["Модель", "Pred. доля@10", "Повтор истории@10", "Покрытие каталога@10"], reward_rows[2:], [44*mm,47*mm,40*mm,39*mm])
    fig("behavior.png")
    p("Предсказанная доля измеряет суррогатную награду, не реальный результат. Покрытие каталога и концентрация Top-1 проверяют схлопывание рекомендаций. Штраф повтора в RL относится к сэмплированным действиям; его эффект может отличаться от изменения детерминированного Top-10.", "small")

    page("06 / Генерация пяти рекомендаций")
    p("В 512 фиксированных тестовых контекстах генерируется по пять жадных действий. Сгенерированный трек добавляется как гипотетически потреблённый; затем предсказывается следующий. Это допущение модели перехода, а не восстановленная реакция пользователя.")
    fig("rollout.png")
    table(["Модель", "Model return", "Повторы внутри 5", "Из исходной истории", "Разных треков"], rollout_rows, [44*mm,31*mm,31*mm,34*mm,30*mm])
    p("Model return = сумма gamma^t · r_hat_t при gamma=0.9. Повторы внутри 5 = 1 - число уникальных действий/5; повторы исходной истории считаются отдельно. Разных треков - число уникальных рекомендаций по всем 512 продолжениям, усреднённое по seed.", "small")
    p("В JSON сохранены raw item ID, предсказания на изменяющемся и на исходном контексте. Это позволяет увидеть зависимость от сгенерированного суффикса. Без реального отклика и корректного OPE рост такого return нельзя интерпретировать как рост длины сессии, удержания или удовлетворённости.")

    page("07 / Выводы и ограничения")
    for i, text in enumerate(conclusions[:5], 1):
        p(f"{i}. {text}")
    p("Что ограничивает вывод", "h2")
    for text in [limitations[0], limitations[5], limitations[6], limitations[7]]:
        p("• " + text, "small")
    p("Что проверять дальше", "h2")
    p("Сначала усилить next-item базу и модель отклика, увеличить каталог и сравнить с recency; затем cross-fitting reward model, ансамбль и консервативные ограничения политики. Для долгосрочного эффекта определить сессии и использовать наблюдаемые возвраты/продолжения с учётом смешения факторов. Подтверждение пользовательской пользы требует отдельного OPE или онлайн-эксперимента.")

    page("08 / Воспроизведение и материалы")
    p(f"Проверено: Python {env['python']}, PyTorch {env['torch']}, Transformers {env['versions']['transformers']}, Linux CPU, 4 потока. Суммарное обучение: {train_seconds:.1f} сек. Подготовка и отчёт в это время не включены. Ориентир памяти - 8 ГБ для воспроизведения; исходный Parquet скачивается отдельно.")
    p("Запуск из распакованного проекта", "h2")
    story.append(Preformatted(commands, styles["code"]))
    p("Полное повторение", "h2")
    story.append(Preformatted("python run_all.py\n# Подготовка заново из исходного Parquet:\npython -m recsys.data --download", styles["code"]))
    p("Архив включает README, requirements, конфигурацию, recsys/, tests/, подготовленные NPZ, отображение каталога, все веса, per-query метрики, журналы и графики. Для проверки готового результата повторное обучение не требуется. Полные формулы, интервалы, команды и ограничения приведены в README.")
    p("Связь со статьями", "h2")
    p("2403.04875 мотивирует генерацию и оптимизацию целей сверх точности. Здесь нет teacher-student стадии этой работы. Из 2607.02818 взята идея отдельной feedback model; two-tower, multi-step off-policy correction и DR/OPE не воспроизводятся. Это самостоятельный компактный эксперимент.", "small")
    for title, url in refs:
        p(title + "\n" + url, "small")

    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#d6e1e6"))
        canvas.line(20*mm, 18*mm, 190*mm, 18*mm)
        canvas.setFont("DV", 7.5)
        canvas.setFillColor(colors.HexColor("#637583"))
        canvas.drawString(20*mm, 13*mm, "Даниил Унковский · Yambda / GPT-2 / RL · 25.10.2026")
        canvas.drawRightString(190*mm, 13*mm, str(doc.page))

    doc = SimpleDocTemplate(str(ROOT / "report.pdf"), pagesize=A4, rightMargin=20*mm,
                            leftMargin=20*mm, topMargin=19*mm, bottomMargin=23*mm,
                            title="GPT-2: рекомендации и обучение с наградой", author="Даниил Унковский")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("README.md and report.pdf generated from measured results.")


if __name__ == "__main__":
    main()
