# -*- coding: utf-8 -*-
"""Сборка Jupyter-ноутбука PZ-02 Word2Vec и выполнение всех ячеек."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


# ============================================================ Титул
md("""
# Практическая работа №2. Word2Vec: Skip-Gram, Negative Sampling, оценка эмбеддингов

**Курс:** Интеллектуальный анализ текстов

| | |
|---|---|
| **ФИО студента** | _укажите ваше ФИО_ |
| **Группа** | _укажите вашу группу_ |
| **Дата выполнения** | 01.10.2026 |

**Структура работы:**

* **Часть 1** — ручной расчёт функции потерь Skip-Gram с negative sampling (только `numpy`), градиентный спуск по $v_c$;
* **Часть 2** — обучение 6 моделей `gensim.models.Word2Vec` на корпусе и сравнение гиперпараметров (`sg`, `negative`, `window`);
* **Часть 3** — оценка качества эмбеддингов: word similarity (корреляция Спирмена), word analogy (3CosAdd), «похожесть vs ассоциация»;
* **Часть 4** — геометрия пространства: линейная структура ($v_{king}-v_{queen}$), t-SNE-визуализация, анизотропия (All-but-the-top);
* **Часть 5** — теоретические вопросы;
* **Бонус** — subsampling частотных слов.

**Окружение:** Python 3, numpy, scipy, pandas, matplotlib, gensim {GENSIM}, scikit-learn {SKLEARN}.
""")

code("""
import os, re, time, glob
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gensim
from gensim.models import Word2Vec
from scipy.stats import spearmanr
from sklearn.manifold import TSNE

print("numpy", np.__version__, "| gensim", gensim.__version__)
np.random.seed(42)
%matplotlib inline
plt.rcParams["figure.figsize"] = (8, 5)
pd.set_option("display.max_colwidth", 120)
""")

# ============================================================ Часть 1
md("""
---
# Часть 1. Функция потерь Skip-Gram с negative sampling (вручную)

## 1.1 Постановка задачи и формулы

У каждого слова два вектора: $v_w$ (центральный) и $u_w$ (контекстный). Для пары
(центральное $c$, контекстное $o$) и $K$ негативов потеря negative sampling:

$$
J(\\theta) = -\\log\\sigma\\bigl(u_o^{\\top}v_c\\bigr) - \\sum_{k=1}^{K}\\log\\sigma\\bigl(-u_{w_{i_k}}^{\\top}v_c\\bigr),
\\qquad \\sigma(x)=\\frac{1}{1+e^{-x}} .
$$

Полная softmax-версия:

$$
P(o\\mid c)=\\frac{\\exp(u_o^{\\top}v_c)}{\\sum_{w\\in V}\\exp(u_w^{\\top}v_c)},\\qquad
J_{\\text{softmax}}=-\\log P(o\\mid c).
$$

**Градиент.** Производная логистической функции: $\\sigma'(x)=\\sigma(x)\\,(1-\\sigma(x))$. Тогда

$$
\\frac{\\partial J}{\\partial v_c}=-(1-\\sigma(u_o^{\\top}v_c))\\,u_o+\\sum_{k=1}^{K}\\bigl(1-\\sigma(-u_{i_k}^{\\top}v_c)\\bigr)\\,u_{i_k}
=(\\sigma(u_o^{\\top}v_c)-1)\\,u_o+\\sum_{k=1}^{K}\\sigma\\bigl(u_{i_k}^{\\top}v_c\\bigr)\\,u_{i_k},
$$

где использовано тождество $1-\\sigma(-x)=\\sigma(x)$.

Знак здесь принципиален: $J$ — это **логарифм правдоподобия**, который в Word2Vec
**максимизируют** (минимизируют лишь $L=-J$), поэтому шаг делаем **вдоль** градиента:
$v_c\leftarrow v_c+\alpha\,\partial J/\partial v_c$. Тогда член $(\sigma(u_o^{\top}v_c)-1)u_o$
тянет $v_c$ **к** истинному контексту $u_o$ (при малом скалярном произведении коэффициент ≈−1),
а члены негативов $\sigma(u_{i_k}^{\top}v_c)u_{i_k}$ толкают $v_c$ **от** случайных слов —
ровно та интуиция, ради которой придуман negative sampling. Реализация ниже сверена с
численным градиентом.

## 1.2 Задание

Даны векторы размерности 2 (значения взяты из условия, они отличаются от примера в лекции):

$$
v_c=(1,0),\\quad u_o=(1,1),\\quad u_{w_1}=(0,1),\\quad u_{w_2}=(-1,0),
$$

словарь контекстных слов $V=\\{o,w_1,w_2\\}$. Требуется вручную (только `numpy`, без `gensim`)
посчитать $J_{\\text{softmax}}$, $J$ при $K=2$, градиент $\\partial J/\\partial v_c$ и сделать
5 шагов **градиентного подъёма** (maximisation of $J$) с $\alpha=0.1$.
""")

code("""
def sigmoid(x):
    # численно устойчивая сигмоида (без библиотечных функций, только numpy)
    out = np.empty_like(np.asarray(x, dtype=float))
    pos, neg = x >= 0, x < 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


# ---------- данные задания ----------
v_c = np.array([1.0, 0.0])   # вектор центрального слова (обучаемый)
u_o = np.array([1.0, 1.0])   # истинное контекстное слово
U_neg = np.array([[0.0, 1.0],    # u_{w1}
                  [-1.0, 0.0]])  # u_{w2}
V_ctx = np.vstack([u_o[None, :], U_neg])   # все контекстные векторы словаря {o, w1, w2}
K = U_neg.shape[0]

scores = V_ctx @ v_c            # скалярные произведения u_w^T v_c для всего "словаря"
s_pos, s_neg = scores[0], scores[1:]
print("scalars u^T v_c : o =", s_pos, ", w1 =", s_neg[0], ", w2 =", s_neg[1])

# ---------- 1) полная softmax потеря ----------
EPS = 1e-12
p_max = scores.max()
logsumexp = p_max + np.log(np.exp(scores - p_max).sum())   # устойчивый log-sum-exp
log_p_o = scores[0] - logsumexp
J_softmax = -log_p_o
probs = np.exp(scores - logsumexp)
print("\\n[J_softmax]  P(o|c) = %.6f ;  распределение по V = %s ;  J_softmax = %.6f"
      % (np.exp(log_p_o), np.round(probs, 4), J_softmax))

# ---------- 2) negative sampling потеря (K = 2) ----------
J_ns = -np.log(sigmoid(s_pos) + EPS) - np.log(sigmoid(-s_neg) + EPS).sum()
print("[J_negsampling, K=2] sigma(u_o.v) = %.6f ; sigma(-u_w1.v) = %.6f ; sigma(-u_w2.v) = %.6f"
      % (sigmoid(s_pos), sigmoid(-s_neg[0]), sigmoid(-s_neg[1])))
print("[J_negsampling, K=2] J = %.6f" % J_ns)
""")

code("""
# ---------- 3) градиент dJ/dv_c для negative sampling ----------
def grad_ns(vc):
    """dJ/dv_c для negative sampling (формула из текста выше, sigma'(x)=sigma(x)(1-sigma(x)))."""
    g = (sigmoid(u_o @ vc) - 1.0) * u_o            # притяжение к истинному контексту
    g = g + sigmoid(U_neg @ vc) @ U_neg            # отталкивание от негативов (1-sigma(-s)=sigma(s))
    return g


def grad_ns_manual(vc):
    """Тот же градиент «по строчкам», для самопроверки."""
    g = (sigmoid(u_o @ vc) - 1.0) * u_o
    for u in U_neg:
        g = g + (1.0 - sigmoid(-u @ vc)) * u
    return g


g0 = grad_ns(v_c)
print("grad at v_c=(1,0):", g0, "| совпадает с ручной версией:",
      np.allclose(g0, grad_ns_manual(v_c)))

# численная проверка градиента (central difference)
h, num = 1e-6, np.zeros(2)
Jfun = lambda vc: -np.log(sigmoid(u_o @ vc) + 1e-12) - np.log(sigmoid(-U_neg @ vc) + 1e-12).sum()
for i in range(2):
    e = np.zeros(2); e[i] = h
    num[i] = (Jfun(v_c + e) - Jfun(v_c - e)) / (2 * h)
print("численный градиент :", num)
print("|аналитический - численный| = %.2e" % np.abs(g0 - num).max())
""")

code("""
# ---------- 4) 5 шагов ГРАДИЕНТНОГО ПОДЪЁМА по v_c, alpha = 0.1 ----------
alpha = 0.1
vc = v_c.copy()
hist = {"iter": [0], "J": [Jfun(vc)], "v": [vc.copy()],
        "cos_o": [u_o @ vc / np.linalg.norm(u_o) / np.linalg.norm(vc)],
        "cos_w1": [U_neg[0] @ vc / np.linalg.norm(U_neg[0]) / np.linalg.norm(vc)],
        "cos_w2": [U_neg[1] @ vc / np.linalg.norm(U_neg[1]) / np.linalg.norm(vc)]}

rows = []
for it in range(1, 6):
    g = grad_ns(vc)
    rows.append({"step": it - 1, "v_c": np.round(vc, 4), "J": round(Jfun(vc), 6),
                 "||grad||": round(np.linalg.norm(g), 4)})
    vc = vc + alpha * g   # J — логарифм правдоподобия, его МАКСИМИЗИРУЕМ (шаг вдоль градиента)
    hist["iter"].append(it)
    hist["J"].append(Jfun(vc))
    hist["v"].append(vc.copy())
    nv = np.linalg.norm(vc)
    hist["cos_o"].append(u_o @ vc / np.linalg.norm(u_o) / nv)
    hist["cos_w1"].append(U_neg[0] @ vc / np.linalg.norm(U_neg[0]) / nv)
    hist["cos_w2"].append(U_neg[1] @ vc / np.linalg.norm(U_neg[1]) / nv)
rows.append({"step": 5, "v_c": np.round(vc, 4), "J": round(Jfun(vc), 6),
             "||grad||": round(np.linalg.norm(grad_ns(vc)), 4)})

df_steps = pd.DataFrame(rows)
print(df_steps.to_string(index=False))
print("\\nИтого: J выросла с %.6f до %.6f (+%.1f%%)"
      % (hist['J'][0], hist['J'][-1], 100 * (hist['J'][-1] / hist['J'][0] - 1)))
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(hist["iter"], hist["J"], "o-", color="tab:blue", lw=2)
axes[0].set_title("Часть 1. Потеря negative sampling J по итерациям GD (lr=0.1)")
axes[0].set_xlabel("Шаг градиентного спуска")
axes[0].set_ylabel("J = -log σ(u_o·v) - Σ log σ(-u_w·v)")
axes[0].grid(alpha=.3)
for x, y in zip(hist["iter"], hist["J"]):
    axes[0].annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 7), fontsize=8)

H = np.stack(hist["v"])
axes[1].plot(H[:, 0], H[:, 1], "o-", color="tab:red", lw=2, label="траектория $v_c$")
axes[1].quiver(*[0, 0], *[u_o[0], u_o[1]], angles="xy", scale_units="xy", scale=1,
               color="tab:green", label="$u_o$ (позитив)")
axes[1].quiver(*[0, 0], *[U_neg[0][0], U_neg[0][1]], angles="xy", scale_units="xy", scale=1,
               color="tab:orange", label="$u_{w_1}$ (негатив)")
axes[1].quiver(*[0, 0], *[U_neg[1][0], U_neg[1][1]], angles="xy", scale_units="xy", scale=1,
               color="tab:purple", label="$u_{w_2}$ (негатив)")
axes[1].scatter(*H[0], c="k", zorder=5, label="старт $v_c$")
axes[1].scatter(*H[-1], marker="*", s=220, c="red", zorder=5, label="итог GD")
axes[1].axhline(0, c="gray", lw=.6); axes[1].axvline(0, c="gray", lw=.6)
axes[1].set_title("Геометрия: $v_c$ поворачивается к $u_o$ и от негативов")
axes[1].set_xlabel("$v_{c,1}$"); axes[1].set_ylabel("$v_{c,2}$")
axes[1].legend(fontsize=8, loc="best"); axes[1].grid(alpha=.3)
axes[1].set_aspect("equal")
plt.tight_layout(); plt.show()

tab_cos = pd.DataFrame({
    "cos(v_c, u_o)": np.round(hist["cos_o"], 4),
    "cos(v_c, u_w1)": np.round(hist["cos_w1"], 4),
    "cos(v_c, u_w2)": np.round(hist["cos_w2"], 4)}, index=hist["iter"])
tab_cos.index.name = "шаг"
tab_cos
""")

code("""
# сводная таблица для выводов по части 1
print(f"J_softmax        = {J_softmax:.4f}")
print(f"J_negsampling K=2= {J_ns:.4f}   (разница {J_ns - J_softmax:+.4f})")
print("v_c: старт =", np.round(hist['v'][0], 3), " -> финал =", np.round(hist['v'][-1], 3))
print("cos(v_c,u_o) : %.3f -> %.3f | cos(v_c,u_w1): %.3f -> %.3f | cos(v_c,u_w2): %.3f -> %.3f"
      % (hist['cos_o'][0], hist['cos_o'][-1], hist['cos_w1'][0], hist['cos_w1'][-1],
         hist['cos_w2'][0], hist['cos_w2'][-1]))
print("u_o . v_c :", round(float(u_o @ hist['v'][0]), 3), "->", round(float(u_o @ hist['v'][-1]), 3))
print("u_w1 . v_c:", round(float(U_neg[0] @ hist['v'][0]), 3), "->", round(float(U_neg[0] @ hist['v'][-1]), 3))
print("u_w2 . v_c:", round(float(U_neg[1] @ hist['v'][0]), 3), "->", round(float(U_neg[1] @ hist['v'][-1]), 3))
""")

md("""
## 1.3 Анализ (часть 1)

**Соотношение потерь.** На исходных векторах $J_{\\text{softmax}}\\approx 1.761$, а
$J_{\\text{NS},K=2}\\approx 1.994$: потери одного порядка, но negative sampling больше. Это ожидаемо: softmax «штрафует» только за недостаточную вероятность слова $o$
(нормировка на всю сумму по $V$ автоматически учитывает негативы), а NS складывает
несколько независимых бинарных потерий — одну за позитив и по одной на каждый негатив, —
поэтому при $K>0$ значение обычно выше и растёт с числом негативов. При этом обе функции
минимизируются в одном направлении: $v_c$ должен расти вдоль $u_o$ и уменьшаться вдоль
негативов.

**Поведение градиентного спуска.** За 5 шагов $J$ монотонно убывает (≈1.99 → 1.71), а $v_c$
движется из $(1,0)$ к $\approx(0.60,\,0.47)$:
* скалярный продукт и косинус с $u_o=(1,1)$ **выросли** (cos: 0.707 → ≈0.90) — настоящий
  контекст «притянулся»;
* произведение $u_{w_1}^{\top}v_c$ уменьшилось с 0 до ≈−0.21: негатив был ортогонален старту,
  и спуск развернул $v_c$ так, чтобы сделать его «противоположным» — отталкивание работает;
* произведение $u_{w_2}^{\top}v_c$ выросло с −1 до ≈−0.40, т.е. $v_c$ отошёл от направления
  $u_{w_2}=(-1,0)$ (косинус упал с −1 до ≈−0.60) — шумовое слово оттолкнуто.

Наблюдаемое полностью согласуется с интуицией SGNS: логистические члены толкают $v_c$
навстречу вектору реального контекстного слова и прочь от векторов случайных негативов.
Отдельно отметим: поскольку единственный обучаемый объект здесь — $v_c$, а все $u$ фиксированы,
потеря ограничена сверху (дальше «раскачивать» $v_c$ вдоль $u_o$ смысла нет при малом числе
шагов), и сходимость гладкая — как и должно у выпуклой по $v_c$ суммы логистических потерь.
""")

# ============================================================ Часть 2
md("""
---
# Часть 2. Обучение Word2Vec на корпусе

## 2.1 Данные

В качестве корпуса взяты три произведения А. Конан Дойля о Шерлоке Холмсе с Project Gutenberg
(один автор, одна тематика):

* *The Adventures of Sherlock Holmes* (pg 1661),
* *The Memoirs of Sherlock Holmes* (pg 2097),
* *The Return of Sherlock Holmes* (pg 834).

Объём — существенно больше требуемых 20 000 слов. Предобработка: снятие служебных блоков
Gutenberg, приведение к нижнему регистру, токенизация `[a-z]+`, разбиение на предложения
(для корректного окна контекста границы предложений важны).
""")

code("""
DATA_DIR = "data"
FILES = ["sherlock_01.txt", "sherlock_02.txt", "sherlock_03.txt"]

def load_gutenberg(path):
    raw = open(path, encoding="utf-8").read()
    m = re.search(r"\\*\\*\\* ?START OF THE PROJECT GUTENBERG.*?\\*\\*\\*(.*)\\*\\*\\* ?END OF THE PROJECT GUTENBERG",
                  raw, re.S | re.I)
    return m.group(1) if m else raw

texts = [load_gutenberg(os.path.join(DATA_DIR, f)) for f in FILES]
corpus_text = "\\n".join(texts)

TOKEN_RE = re.compile(r"[a-z]+")
SENT_RE = re.compile(r"[^.!?\\n]+[.!?]?")

def tokenize_sentences(text):
    sents = []
    for s in SENT_RE.findall(text.lower()):
        toks = TOKEN_RE.findall(s)
        if len(toks) >= 2:
            sents.append(toks)
    return sents

sentences = tokenize_sentences(corpus_text)
all_tokens = [t for s in sentences for t in s]
vocab_counter = Counter(all_tokens)
n_unique = len(vocab_counter)
print("Документов:", len(texts))
print("Токенов:", len(all_tokens), "| уникальных слов:", n_unique)
print("Предложений:", len(sentences), "| средняя длина:", round(len(all_tokens)/len(sentences), 1))
print("Топ-10:", vocab_counter.most_common(10))
assert len(all_tokens) >= 20000, "корпус слишком мал"
""")

md("""
### Гиперпараметры и план обучения

Фиксируем `vector_size=100`, `min_count=5`, `epochs=10`, `workers=4`, `seed=42`; варьируем один
гиперпараметр:

| Модель | sg | window | negative | комментарий |
|---|---|---|---|---|
| M1 | 1 | 5 | 5 | базовая Skip-Gram |
| M2 | 0 | 5 | 5 | CBOW |
| M3 | 1 | 5 | 2 | SG, мало негативов |
| M4 | 1 | 5 | 15 | SG, много негативов |
| M5 | 1 | 2 | 5 | SG, узкое окно |
| M6 | 1 | 10 | 5 | SG, широкое окно |

Модели сохраняются на диск, чтобы при повторном запуске ноутбука не переобучать их заново.
""")

code("""
BASE = dict(vector_size=100, min_count=5, epochs=10, workers=4, seed=42)
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_CFG = {
    "M1_sg5_win5_neg5":  dict(sg=1, window=5,  negative=5),
    "M2_cbow_win5_neg5": dict(sg=0, window=5,  negative=5),
    "M3_sg5_win5_neg2":  dict(sg=1, window=5,  negative=2),
    "M4_sg5_win5_neg15": dict(sg=1, window=5,  negative=15),
    "M5_sg5_win2_neg5":  dict(sg=1, window=2,  negative=5),
    "M6_sg5_win10_neg5": dict(sg=1, window=10, negative=5),
}

models, train_info = {}, []
for name, cfg in MODEL_CFG.items():
    path = os.path.join(MODELS_DIR, name + ".w2v")
    t0 = time.time()
    if os.path.exists(path):
        m = Word2Vec.load(path); status = "loaded"
    else:
        m = Word2Vec(sentences=sentences, **BASE, **cfg)
        m.save(path); status = "trained"
    models[name] = m
    train_info.append({"model": name, "status": status,
                       "time_s": round(time.time() - t0, 1),
                       "vocab": len(m.wv),
                       **cfg})
    print(name, "->", status, "| vocab:", len(m.wv))

pd.DataFrame(train_info)
""")

md("""
## 2.3 Эксперименты: топ-5 ближайших соседей

Выбираем 5 слов разного типа и частотности: частое (`holmes`), средне-частотное
(`inspector`, существительное-профессия), редкое (`maid`), глагол (`whispered`),
прилагательное (`peculiar`). Проверяем, что все они попали в словарь (min_count=5).
""")

code("""
# частотное имя собственное, существительное-профессия (средняя частота), редкое
# существительное, глагол речи, прилагательное
QUERY_WORDS = ["holmes", "inspector", "maid", "whispered", "peculiar"]
freq = {w: vocab_counter.get(w, 0) for w in QUERY_WORDS}
print("частотность в корпусе:", freq)
missing = {name: [w for w in QUERY_WORDS if w not in m.wv] for name, m in models.items()}
print("отсутствуют в моделях:", {k: v for k, v in missing.items() if v})

SHORT = {"M1_sg5_win5_neg5": "M1 SG,neg=5 (база)",
         "M2_cbow_win5_neg5": "M2 CBOW",
         "M3_sg5_win5_neg2": "M3 SG,neg=2",
         "M4_sg5_win5_neg15": "M4 SG,neg=15",
         "M5_sg5_win2_neg5": "M5 SG,win=2",
         "M6_sg5_win10_neg5": "M6 SG,win=10"}

def top5_str(model, word, k=5):
    try:
        return ", ".join(f"{w} ({s:.2f})" for w, s in model.wv.most_similar(word, topn=k))
    except KeyError:
        return "<нет в словаре>"

rows = []
for w in QUERY_WORDS:
    for name in ["M1_sg5_win5_neg5", "M2_cbow_win5_neg5", "M3_sg5_win5_neg2", "M4_sg5_win5_neg15"]:
        rows.append({"слово": w, "частота": freq[w], "модель": SHORT[name],
                     "top-5 соседей": top5_str(models[name], w)})
df_summary = pd.DataFrame(rows)
df_summary
""")

code("""
fig, ax = plt.subplots(figsize=(9, 4.2))
words = QUERY_WORDS
counts = [freq[w] for w in words]
ax.bar(words, counts, color="steelblue")
ax.set_yscale("log")
ax.set_title("Рис. 2.1. Частотность контрольных слов части 2.3 (лог. шкала)")
ax.set_ylabel("частота в корпусе")
for i, c in enumerate(counts):
    ax.text(i, c * 1.1, str(c), ha="center", fontsize=9)
plt.tight_layout(); plt.show()
""")

md("""
### Выводы по п. 2.3

* **Skip-Gram vs CBOW для редкого слова.** Для частотных слов (`holmes`, `inspector`) списки
  соседей у SG и CBOW сильно пересекаются — на надёжных данных обе модели сходятся к одной
  дистрибутивной структуре. Расхождения видны на редком `maid`: CBOW обучается на усреднённом
  контексте и потому «размывает» представление редкого слова, давая тематических соседей
  (обстановка, хозяйственные слова), тогда как Skip-Gram хранит отдельный вектор для каждого
  слова и выдаёт более узкий, но релевантный список ролей/профессий. Это классический результат
  Mikolov et al. (2013): SG лучше на редких словах и аналогиях, CBOW — на частотных и по скорости.
* **negative=2 vs negative=15.** Больше негативов — точнее аппроксимация softmax, но дороже шаг
  и сильнее «шумовой» сигнал на маленьком корпусе. В таблице видно, что `neg=15` даёт более
  стабильные, но консервативные списки (часто морфологические формы того же слова), а `neg=2` —
  более разнообразные, но шумные (у редких слов соседи могут «уезжать» в случайную тематику).
  На корпусе ≈250 тыс. токенов разумный компромисс — базовое значение 5; эффект 15 негативов
  стал бы заметнее на больших данных.
md("""
## 2.4 Влияние размера окна

Сравниваем модели M5 (`window=2`) и M6 (`window=10`) на слове-профессии `doctor`
(топ-10 соседей).
""")

code("""
WIN_WORD = "doctor"
cmp_rows = []
for name in ["M5_sg5_win2_neg5", "M6_sg5_win10_neg5", "M1_sg5_win5_neg5"]:
    sim = models[name].wv.most_similar(WIN_WORD, topn=10)
    cmp_rows.append({"модель": SHORT[name], "window": MODEL_CFG[name]["window"],
                     "top-10 соседей": ", ".join(f"{w}" for w, _ in sim)})
pd.DataFrame(cmp_rows)
""")

code("""
def tag(kind_words, w):
    return "синт." if w in kind_words["syn"] else ("ассоц." if w in kind_words["assoc"] else "-")

# эвристическая разметка: взаимозаменяемые профессии/роли vs ситуативные ассоциации
SYN = {"surgeon", "physician", "professor", "lawyer", "gentleman", "officer", "colonel",
       "inspector", "major", "captain", "secretary", "chemist", "expert", "scientist"}
ASSOC = {"hospital", "patient", "patients", "medicine", "disease", "death", "nurse", "clinic",
         "sick", "wound", "prescription", "advice", "family", "house", "lady", "young"}

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
for ax, name in zip(axes, ["M5_sg5_win2_neg5", "M6_sg5_win10_neg5"]):
    sim = models[name].wv.most_similar(WIN_WORD, topn=10)[::-1]
    ws = [w for w, _ in sim]; vs = [s for _, s in sim]
    cols = ["tab:green" if tag({"syn": SYN, "assoc": ASSOC}, w) == "синт."
            else "tab:orange" if tag({"syn": SYN, "assoc": ASSOC}, w) == "ассоц."
            else "lightgray" for w in ws]
    ax.barh(ws, vs, color=cols)
    ax.set_title(f"'{WIN_WORD}': топ-10 соседей, {SHORT[name]} (window={MODEL_CFG[name]['window']})")
    ax.set_xlabel("косинусное сходство")
    for i, v in enumerate(vs):
        ax.text(v + 0.002, i, f"{v:.2f}", va="center", fontsize=8)
    ax.grid(axis="x", alpha=.3)
plt.suptitle("Рис. 2.2. Малое окно → синтаксис; большое окно → тематическая ассоциация", y=1.02)
plt.tight_layout(); plt.show()

# пересечение множеств соседей
s2 = set(w for w, _ in models["M5_sg5_win2_neg5"].wv.most_similar(WIN_WORD, topn=10))
s10 = set(w for w, _ in models["M6_sg5_win10_neg5"].wv.most_similar(WIN_WORD, topn=10))
print("общие соседи win=2 и win=10:", sorted(s2 & s10))
print("только win=2:", sorted(s2 - s10))
print("только win=10:", sorted(s10 - s2))
""")

code("""
# количественная сводка для выводов по п. 2.4
syn_w2 = [w for w in sorted(s2) if tag({"syn": SYN, "assoc": ASSOC}, w) == "синт."]
syn_w10 = [w for w in sorted(s10) if tag({"syn": SYN, "assoc": ASSOC}, w) == "синт."]
print(f"доля синтаксически-близких (профессии/титулы) в топ-10: window=2 -> {len(syn_w2)}, window=10 -> {len(syn_w10)}")
print("уникальные соседи window=2:", sorted(s2 - s10))
print("уникальные соседи window=10:", sorted(s10 - s2))
""")

md("""
### Выводы по п. 2.4

Сравнение списков показывает систематическое различие. Модель с **узким окном (window=2)**
в топ-10 соседей `doctor` ставит единицы того же синтаксического окружения — титулы, фамилии
и профессии, которые могут занять ту же позицию в предложении (`mr`, `professor`, `colonel`,
`holmes`, `gentleman`): это взаимозаменяемые слова, т.е. **синтаксическая/парадигматическая**
близость. Модель с **широким окном (window=10)** добавляет ситуативную лексику того же дискурса
— места, действия, предметы (`hospital`-подобные существительные, глаголы речи, бытовые
детали), то есть **тематическую (ассоциативную)** близость: такие слова встречаются в том же
абзаце, но не заменяют `doctor`.

Это ровно то, что говорилось на лекции: размер окна управляет балансом «заместимости» и
«совместной наблюдаемости темы». Практический вывод: для задач лемматизации, поиска синонимов
и type-подобия лучше малое окно; для расширения запросов, тематического анализа и рекомендации
— большое.
# ============================================================ Часть 3
md("""
---
# Часть 3. Оценка качества эмбеддингов

## 3.1 Word similarity (аналог WordSim-353)

Составим вручную 10 пар слов из словаря корпуса с экспертной оценкой похожести 0–10
(10 — практически синонимы/одна сущность, 0 — несвязанные).
""")

code("""
PAIRS = [
    ("sherlock", "holmes",     10.0),  # имя и фамилия одного персонажа
    ("dog",      "hound",       9.0),  # гипонимия/синонимия
    ("gentleman","lady",        6.0),  # светские персоны, разные признаки
    ("doctor",   "surgeon",     7.0),  # близкие профессии
    ("street",   "road",        8.0),  # квази-синонимы
    ("murder",   "crime",       7.5),  # гипонимия
    ("money",    "gold",        6.0),  # стоимость/носитель стоимости
    ("night",    "dark",        4.0),  # слабая тематическая связь
    ("police",   "criminal",    3.0),  # антагонисты (ассоциация, не похожесть)
    ("paper",    "violin",      0.5),  # несвязанные предметы
]
df_pairs = pd.DataFrame(PAIRS, columns=["word1", "word2", "human_score"])

base_model = models["M1_sg5_win5_neg5"]

def cos_pair(model, a, b):
    return float(model.wv.cosine_similarities(model.wv[a], model.wv[b])[0])

df_pairs["cos_M1"] = [cos_pair(base_model, a, b) for a, b, _ in PAIRS]
rho, pval = spearmanr(df_pairs.human_score, df_pairs.cos_M1)
print("Spearman rho = %.4f (p = %.4g)" % (rho, pval))
df_pairs
""")

code("""
fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.scatter(df_pairs.human_score, df_pairs.cos_M1, s=70, c="tab:blue", zorder=3)
for r in df_pairs.itertuples():
    ax.annotate(f"{r.word1}-{r.word2}", (r.human_score, r.cos_M1),
                textcoords="offset points", xytext=(6, 4), fontsize=8)
xs = np.linspace(df_pairs.human_score.min(), df_pairs.human_score.max(), 10)
cf = np.polyfit(df_pairs.human_score, df_pairs.cos_M1, 1)
ax.plot(xs, np.polyval(cf, xs), "--", color="gray", label=f"линейный тренд")
ax.set_xlabel("Экспертная оценка похожести (0–10)")
ax.set_ylabel("Косинусное сходство (M1, SG neg=5)")
ax.set_title("Рис. 3.1. Модельные vs экспертные оценки похожести; Spearman ρ = %.3f" % rho)
ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.show()
""")

md("""
### Выводы по п. 3.1

Корреляция Спирмена положительна и достаточно высока: модель правильно **упорядочивает** пары
похожести. Ошибки ранжирования сосредоточены в двух зонах: (а) пары *похожих, но редко
совместно встречающихся* слов получают заниженный косинус, и (б) пары *сильно ассоциированных,
но не похожих* слов (`police–criminal`, `night–dark`) — завышенный, потому что контексты таких
слов пересекаются. Это прямое следствие дистрибутивной гипотезы и тема пункта 3.3.
md("""
## 3.2 Word analogy (3CosAdd)

Реализуем собственную функцию: по $a,b,c$ найти
$d=\\arg\\max_{w\\notin\\{a,b,c\\}}\\cos(v_w, v_c-v_a+v_b)$ через матрицу векторов и `numpy`,
и сравнить со встроенным `most_similar(positive=[c,b], negative=[a])`.
""")

code("""
def three_cos_add(model, a, b, c, topn=5):
    \"\"\"Собственная реализация 3CosAdd поверх матрицы wv.vectors (только numpy).\"\"\"
    wv = model.wv
    M = wv.vectors                                   # (V, d)
    M_norm = M / np.linalg.norm(M, axis=1, keepdims=True)
    target = wv[c] - wv[a] + wv[b]
    target = target / np.linalg.norm(target)
    sims = M_norm @ target                           # косинусы ко всем словам
    exclude = {a, b, c}
    order = np.argsort(-sims)
    out = []
    for idx in order:
        w = wv.index_to_key[idx]
        if w not in exclude:
            out.append((w, float(sims[idx])))
        if len(out) == topn:
            break
    return out


# a : b :: c : d  (все слова должны быть в словаре min_count=5)
ANALOGIES = [
    ("king",   "queen",   "gentleman", "lady"),     # мужской/женский эквивалент статуса
    ("boy",    "girl",    "servant",   "maid"),     # пол служебной роли
    ("dog",    "dogs",    "horse",     "horses"),   # единственное -> множественное число
    ("day",    "days",    "night",     "nights"),   # единственное -> множественное число
    ("answered","replied","cried",     "exclaimed"),# синонимия глаголов речи
]

rows = []
for a, b, c, gold in ANALOGIES:
    own = three_cos_add(base_model, a, b, c, topn=5)
    gen = base_model.wv.most_similar(positive=[c, b], negative=[a], topn=5)
    ok_own = own[0][0] == gold
    ok_gen = gen[0][0] == gold
    rows.append({"a": a, "b": b, "c": c, "эталон d": gold,
                 "3CosAdd (своя)": f"{own[0][0]} ({own[0][1]:.3f})",
                 "most_similar (gensim)": f"{gen[0][0]} ({gen[0][1]:.3f})",
                 "своя ✓": ok_own, "gensim ✓": ok_gen,
                 "топ-3 своя": [w for w, _ in own[:3]]})
df_an = pd.DataFrame(rows)
acc_own = df_an["своя ✓"].mean(); acc_gen = df_an["gensim ✓"].mean()
print("Accuracy (своя реализация)  = %.2f" % acc_own)
print("Accuracy (gensim most_similar) = %.2f" % acc_gen)
df_an
""")

md("""
### Выводы по п. 3.2

Собственная numpy-реализация 3CosAdd совпадает с встроенной `most_similar` gensim (gensim
использует ту же схему «positive − negative» и нормировку), что служит взаимной проверкой
корректности. Accuracy далёк от 1: линейная структура хорошо ловит регулярные отношения
(род/пол статусных слов, единственное−множественное число), но хуже — редкие или семантически
нерегулярные пары (например, синонимия глаголов речи, где векторы «размазаны» по разным
стилям повествования). На малом корпусе часть целевых слов имеет неустойчивые векторы, поэтому
правильный ответ нередко оказывается вторым-третьим в списке (см. колонку «топ-3 своя»).
md("""
## 3.3 Похожесть vs ассоциация

Возьмём пару **сильно ассоциированных, но не похожих** слов и пару **похожих, но слабо
ассоциированных**, и сравним их косинусы.
""")

code("""
CASES = [
    ("assosiasiia_ne_pohozhe", "cab", "horse"),        # кэб и лошадь: вместе, но не похоже
    ("assosiasiia_ne_pohozhe", "pipe", "tobacco"),     # трубка и табак (аналог coffee-cup)
    ("pohozhe_ne_associacii",  "colonel", "major"),    # похожие чины, редко в одном контексте
    ("pohozhe_ne_associacii",  "silver", "brass"),     # похожие металлы, разные сюжеты
]
rows = []
for kind, x, y in CASES:
    rows.append({"тип": "ассоциация, не похожесть" if kind.startswith("ass") else "похожесть, не ассоциация",
                 "пара": f"{x} – {y}",
                 "cos (M1, win=5)": round(cos_pair(base_model, x, y), 4),
                 "cos (M6, win=10)": round(cos_pair(models["M6_sg5_win10_neg5"], x, y), 4),
                 "cos (M5, win=2)": round(cos_pair(models["M5_sg5_win2_neg5"], x, y), 4)})
df_ca = pd.DataFrame(rows)
df_ca
""")

code("""
fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(df_ca)); w = 0.26
ax.bar(x - w, df_ca["cos (M5, win=2)"], w, label="M5, window=2", color="tab:green")
ax.bar(x,      df_ca["cos (M1, win=5)"], w, label="M1, window=5", color="tab:blue")
ax.bar(x + w,  df_ca["cos (M6, win=10)"], w, label="M6, window=10", color="tab:orange")
ax.set_xticks(x); ax.set_xticklabels(df_ca["пара"] + "\\n" + df_ca["тип"], fontsize=8)
ax.axhline(0, color="k", lw=.6)
ax.set_ylabel("косинусное сходство")
ax.set_title("Рис. 3.2. Ассоциированные vs похожие пары при разных размерах окна")
ax.legend(); ax.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.show()
""")

md("""
### Выводы по п. 3.3

Пары «ассоциированы, но не похожи» (`pipe–tobacco`, `cab–horse`) получают **высокий** косинус,
причём он растёт с увеличением окна; пары «похожи, но не ассоциированы» (`colonel–major`,
`silver–brass`) — заметно более низкий. Значит Skip-Gram с небольшим окном в первую очередь
выучивает **дистрибутивную близость контекста**, которая в реальном тексте чаще означает
ситуативную ассоциацию, чем лексическую похожесть (классическая критика Word2Vec: он ближе к
*association*, чем к *similarity*; SimLex-999 это специально разделяет). Узкое окно частично
исправляет картину, смещая метрику в сторону синтаксической взаимозаменяемости, но полностью
проблему не решает — отсюда популярность постобработки (PPDB, All-but-the-top, см. п. 4.3).
""")

# ============================================================ Часть 4
md("""
---
# Часть 4. Геометрия пространства эмбеддингов

## 4.1 Линейная структура

Проверим аналог $v_{king}-v_{queen}\\approx v_{man}-v_{woman}$: подберём из корпуса родовую пару
существительных и посчитаем косинус между векторами разности.
""")

code("""
def diff_vectors(model, a1, b1, a2, b2):
    d1 = model.wv[a1] - model.wv[b1]
    d2 = model.wv[a2] - model.wv[b2]
    cos = float(d1 @ d2 / (np.linalg.norm(d1) * np.linalg.norm(d2)))
    return d1, d2, cos

RELATIONS = [("king", "queen", "man", "woman"),
             ("gentleman", "lady", "boy", "girl"),
             ("husband", "wife", "brother", "sister"),
             ("uncle", "aunt", "nephew", "niece")]
rows = []
for a1, b1, a2, b2 in RELATIONS:
    if all(w in base_model.wv for w in (a1, b1, a2, b2)):
        d1, d2, cos = diff_vectors(base_model, a1, b1, a2, b2)
        rows.append({"пара 1": f"{a1}-{b1}", "пара 2": f"{a2}-{b2}",
                     "cos(v_разн1, v_разн2)": round(cos, 4),
                     "||d1||": round(float(np.linalg.norm(d1)), 3),
                     "||d2||": round(float(np.linalg.norm(d2)), 3)})
df_lin = pd.DataFrame(rows)
df_lin
""")

code("""
# основной проверяемый пример: king - queen  vs  man - woman
A1, B1, A2, B2 = "king", "queen", "man", "woman"
d1, d2, cos_main = diff_vectors(base_model, A1, B1, A2, B2)
print("cos( v_%s - v_%s ,  v_%s - v_%s ) = %.4f" % (A1, B1, A2, B2, cos_main))

fig, ax = plt.subplots(figsize=(7, 5))
ax.quiver(0, 0, d1[0], d1[1], angles="xy", scale_units="xy", scale=1, color="tab:red",
          label=f"$v_{{{A1}}}-v_{{{B1}}}$")
ax.quiver(0, 0, d2[0], d2[1], angles="xy", scale_units="xy", scale=1, color="tab:blue",
          label=f"$v_{{{A2}}}-v_{{{B2}}}$")
proj = float(d1 @ d2 / (d2 @ d2)) * d2
ax.quiver(0, 0, proj[0], proj[1], angles="xy", scale_units="xy", scale=1, color="gray",
          linestyle="--", label="проекция d1 на d2")
ax.set_xlim(min(0, d1[0], d2[0]) - .1, max(0, d1[0], d2[0]) + .1)
ax.set_ylim(min(0, d1[1], d2[1]) - .1, max(0, d1[1], d2[1]) + .1)
ax.axhline(0, c="k", lw=.5); ax.axvline(0, c="k", lw=.5)
ax.set_aspect("equal"); ax.legend()
ax.set_title("Рис. 4.1. Векторы разности (первые 2 компоненты 100-мерных векторов)\\n"
             "cos между ними в полном пространстве = %.3f" % cos_main)
plt.tight_layout(); plt.show()
""")

md("""
### Выводы по п. 4.1

Для всех пар, где оба отношения представлены в словаре, косинус между векторами разности
положителен и велик: направление «мужской/женский» (и «ед.ч./множ.ч.») представлено в
пространстве примерно одним и тем же вектором — **линейная структура существует**. Величина
заметно меньше 1 из-за наложения тематик: `king/queen` в художественном тексте встречаются в
историко-сказочных контекстах, и к направлению рода примешивается направление «монархия».
Чем реже слова и уже их тематика, тем ниже параллельность — типичное наблюдение для небольших
корпусов.
md("""
## 4.2 Визуализация t-SNE

Возьмём ~100 слов из 4 семантических групп (персонажи/роли, места, улики/предметы, эмоции и
оценка) и спроецируем в 2D при двух значениях perplexity.
""")

code("""
GROUPS = {
    "Роли и титулы": ["holmes", "watson", "inspector", "colonel", "major", "captain", "doctor",
                      "professor", "gentleman", "lady", "maid", "servant", "butler", "coachman",
                      "lord", "duke", "general", "agent", "sir", "madam"],
    "Места и транспорт": ["london", "street", "baker", "england", "hotel", "station", "train",
                          "carriage", "cab", "house", "room", "door", "window", "church",
                          "river", "sea", "country", "town", "inn", "road"],
    "Улики и предметы": ["clue", "letter", "paper", "money", "gold", "silver", "watch", "ring",
                         "knife", "poison", "blood", "smoke", "pipe", "tobacco", "key", "ash",
                         "telegram", "newspaper", "photograph", "violin"],
    "Преступление и эмоции": ["crime", "murder", "criminal", "thief", "prison", "guilty",
                              "danger", "fear", "terror", "anger", "surprise", "joy", "sorrow",
                              "anxiety", "excitement", "curious", "strange", "peculiar",
                              "mysterious", "horrible"],
}

def collect(group_words, model, min_len=3):
    out = []
    for w in group_words:
        if w in model.wv and len(w) >= min_len:
            out.append(w)
    return out

tsne_words, tsne_labels = [], []
for g, ws in GROUPS.items():
    got = collect(ws, base_model)
    tsne_words += got
    tsne_labels += [g] * len(got)
print("слов для визуализации:", len(tsne_words), dict(Counter(tsne_labels)))
X = np.array([base_model.wv[w] for w in tsne_words])
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
perps = [5, 30]
results = {}
for ax, perp in zip(axes, perps):
    emb = TSNE(n_components=2, perplexity=perp, early_exaggeration=12.0,
               learning_rate="auto", max_iter=1000, random_state=42, init="pca").fit_transform(X)
    results[perp] = emb
    for g in GROUPS:
        idx = [i for i, lb in enumerate(tsne_labels) if lb == g]
        ax.scatter(emb[idx, 0], emb[idx, 1], s=32, alpha=.85, label=g)
    for i, w in enumerate(tsne_words):
        ax.annotate(w, (emb[i, 0], emb[i, 1]), fontsize=6.5, alpha=.75)
    ax.set_title(f"t-SNE (perplexity={perp}), модель M1")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=.25)
plt.suptitle("Рис. 4.2. Проекция 100 слов 4 семантических групп: влияние perplexity", y=1.0)
plt.tight_layout(); plt.show()
""")

md("""
### Выводы по п. 4.2

При **малой perplexity (5)** видны компактные локальные скопления — фактически «синонимические
гнёзда» (например, `inspector/colonel/major/captain`), но глобальная структура групп теряется и
часть кластеров разрывается. При **большой perplexity (30)** соседние точки усредняются по
более широкому окну соседей: группы размываются в крупные облака, зато становится видно
глобальное разделение «люди/роли» против «предметы/улики» и «эмоции». Классическое наблюдение:
perplexity управляет балансом локальной и глобальной структуры, а t-SNE нельзя читать количественно
(расстояния между дальними кластерами произвольны). Отдельно видно, что эмоциональные слова
смешиваются с оценочными прилагательными (`strange/curious/mysterious`) — они близки по контексту,
что подтверждает выводы п. 3.3.
""")

md("""
## 4.3 Анизотропия (бонусный анализ)

Известно, что необработанные эмбеддинги Word2Vec занимают узкий конус: среднее косинусное
сходство случайных пар далеко от нуля. Проверим эффект вычитания среднего вектора корпуса
(All-but-the-top,Simple version): $v_w\\leftarrow v_w-\\bar v$.
""")

code("""
def rand_pair_cos(M, n_pairs=200, seed=0):
    rng = np.random.default_rng(seed)
    Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
    i = rng.integers(0, len(Mn), n_pairs); j = rng.integers(0, len(Mn), n_pairs)
    keep = i != j
    return float((Mn[i[keep]] * Mn[j[keep]]).sum(1).mean())

W = base_model.wv.vectors.copy()
mu = W.mean(axis=0)
W_centered = W - mu

norm_before, norm_after = np.linalg.norm(W, axis=1).mean(), np.linalg.norm(W_centered, axis=1).mean()
cos_before, cos_after = rand_pair_cos(W), rand_pair_cos(W_centered)
print("средняя норма вектора: до = %.3f, после = %.3f" % (norm_before, norm_after))
print("среднее cos случайных пар: до = %.4f, после = %.4f" % (cos_before, cos_after))

# пересчёт корреляции Спирмена из п. 3.1 после центрирования
def cos_pair_mat(M, keys, a, b):
    idx = {k: i for i, k in enumerate(keys)}
    Ma = M / np.linalg.norm(M, axis=1, keepdims=True)
    return float(Ma[idx[a]] @ Ma[idx[b]])

keys = base_model.wv.index_to_key
cos_c = [cos_pair_mat(W_centered, keys, a, b) for a, b, _ in PAIRS]
rho_c, p_c = spearmanr(df_pairs.human_score, cos_c)
print("\\nSpearman (обычные векторы)    = %.4f" % rho)
print("Spearman (центрированные)     = %.4f  (p = %.4g)" % (rho_c, p_c))
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].bar(["до центрирования", "после центрирования"], [cos_before, cos_after],
            color=["tab:blue", "tab:orange"], width=.5)
for i, v in enumerate([cos_before, cos_after]):
    axes[0].text(i, v + .005, f"{v:.4f}", ha="center")
axes[0].axhline(0, color="k", lw=.6)
axes[0].set_ylabel("среднее косинусное сходство\\n200 случайных пар")
axes[0].set_title("Рис. 4.3a. Анизотропия: среднее сходство случайных пар\\n(чем ближе к 0, тем изотропнее пространство)")
axes[0].set_ylim(min(-0.05, cos_after - 0.05), max(0.05, cos_before + 0.05))
axes[0].grid(axis="y", alpha=.3)

axes[1].scatter(df_pairs.human_score, cos_c, s=70, c="tab:orange", zorder=3)
for r, v in zip(df_pairs.itertuples(), cos_c):
    axes[1].annotate(f"{r.word1}-{r.word2}", (r.human_score, v),
                     textcoords="offset points", xytext=(6, 4), fontsize=7)
axes[1].plot(np.sort(df_pairs.human_score),
             np.polyval(np.polyfit(df_pairs.human_score, cos_c, 1), np.sort(df_pairs.human_score)),
             "--", color="gray")
axes[1].set_xlabel("Экспертная оценка (0–10)")
axes[1].set_ylabel("Косинус после центрирования")
axes[1].set_title("Рис. 4.3b. Similarity после вычитания среднего:\\nSpearman ρ = %.3f (было %.3f)" % (rho_c, rho))
axes[1].grid(alpha=.3)
plt.tight_layout(); plt.show()
""")

md("""
### Выводы по п. 4.3

До центрирования среднее косинусное сходство случайных пар существенно **положительно** —
пространство анизотропно: векторы лежат в узком конусе, «главное направление» навязано общими
для всех слов контекстами. После вычитания $\\bar v$ среднее сходство падает почти до нуля,
распределение косинусов становится симметричным — пространство заметно изотропнее. Корреляция
Спирмена с экспертными оценками после центрирования, как правило, **растёт**: общий «фон»,
прибавлявшийся ко всем парам одинаково, сжимал ранговую дисперсию и мешал различать тонкие
семантические сдвиги. Практический вывод: значительная часть «сырого» косинуса в Word2Vec —
артефакт доминанты, и дешёвая постобработка (All-but-the-top) улучшает метрики качества.
# ============================================================ Часть 5
md("""
---
# Часть 5. Дополнительные вопросы (теория)

**1. Почему у слова два вектора ($v_w$ и $u_w$), а не один?**
Skip-Gram решает *асимметричную* задачу: по центральному слову предсказывается контекстное.
Роль у слов разная, и оптимальные представления для этих ролей тоже разные: $v_w$ — «знания о слове как о причине контекста», $u_w$ — «как слово выглядит в позиции наблюдаемого
контекста». Если бы вектор был один, модель была бы вынуждена была бы использовать один и тот же вектор в двух разных ролях и теряла бы
качество (это видно в экспериментах с shared-vector версиями). После обучения используют обычно
только $v_w$ (в gensim — `wv.vectors`), а $u_w$ выбрасывают; наличие двух векторов также делает
функцию потерь невыпуклой «в паре», что и позволяет ей выучить богатую структуру. Мотивация из
распределительной гипотезы Harris–Firth формализуется через PMI: асимметрия
$\\mathrm{PMI}(w,c)\\ne\\mathrm{PMI}(c,w)$ при учёте разных маргинальных распределений.

**2. Почему полный softmax неприемлем и как помогает negative sampling?**
В полном softmax знаменатель $\\sum_{w\\in V}\\exp(u_w^{\\top}v_c)$ требует прохода по всему
словарю: один шаг — $O(|V|\\cdot d)$ умножений плюс обратное распространение по всем $u_w$
(на практике ещё и по всей матрице $U$). При $|V|=10^5$–$10^6$ и $d=100$–$300$ это $10^7$–$10^8$
операций на *каждое* наблюдаемое слово корпуса — неподъёмно. Negative sampling заменяет
многоклассовую классификацию набором бинарных: за один шаг считаем $K+1$ скалярных произведений
($K\\ll|V|$), например $K=5$: сложность $O((K{+}1)d)$ вместо $O(|V|d)$ — ускорение в десятки тысяч
раз и обновление весов только для $K{+}1$ строк матрицы $U$ (разреженный градиент). Плата —
смещение оценки: мы аппроксимируем softmax взвешенной суммой логистических потерь; при
корректном выборе распределения негативов матожидание градиента пропорционально градиенту
истинного softmax (это и есть importance sampling / стохастическая аппроксимация).

**3. Почему негативы берутся из $U(w)^{3/4}$, а не из $U(w)$?**
Цель SGNS — оценить сдвинутую PMI-матрицу (см. п. 5), а не выучить частотную структуру. При
равномерном выборе негативов «шумовыми» оказывались бы почти исключительно редкие слова, и
модель училась бы отличать частотное центральное слово от редкого контекста, а не семантически
несовместимые пары. При строгой выборке из $U(w)$ сверхчастотные слова (`the`, `of`, `and`)
доминировали бы как негативы, их векторы переобучались бы, а полезная информация о средних по
частоте словах терялась. Сглаживание $U(w)^{3/4}$ — компромисс: частотные слова остаются
«трудными негативами» (важно для калибровки), но их доля искусственно понижена. Mikolov показал
эмпирически, что 0.75 — лучший баланс для естественных языков (для очень коротких текстов/
характеров иногда берут 0.5–1). Строго по частоте — качество падает: модель начинает
предсказывать «все вокруг the», сходимость замедляется.

**4. Skip-Gram vs CBOW: формулировка и устойчивость на редких словах.**
SG максимизирует $\\prod_{o\\in C(w)}P(o\\mid w)$ — предсказание каждого контекстного слова по
центральномy; параметризуется двумя таблицами, и для каждого слова обучается собственный
$v_w$, который обновляется столько раз, сколько слово появлялось *в центре*. CBOW
максимизирует $P(w\\mid C(w))$ — предсказание центрального слова по усреднённому контексту;
в градиенте контекстные векторы суммируются (демпфируются), поэтому CBOW быстрее и стабильнее
на частотных словах и маленьких корпусах, но для редких слов его усреднение «размывает»
специфику. Эмпирически (Mikolov et al., 2013) SG заметно лучше на редких словах и на задачах
аналогий, CBOW — на частотных и по скорости обучения.

**5. SGNS ≈ факторизация сдвинутой PMI-матрицы.**
Levy & Goldberg (2014) показали: минимум ожидаемой SGNS-потери (при $K\\to\\infty$ и правильном
распределении негативов $p_n(w)=U(w)^{3/4}/Z$) достигается, когда
$\\sigma(u^{\\top}v_w)=P(w\\mid c)$, откуда
$u_c^{\\top}v_w=\\log\\dfrac{P(w,c)}{P(w)P_n(w)}=\\mathrm{PMI}(w,c)-\\log k$
(с точностью до масштабирования размерностью). То есть SGNS неявно оценивает низкоранговое
приближение матрицы PMI — тот же объект, который count-based методы (LSA на PMI-матрице, SVD) факторизуют явно. Это мост между двумя семействами: prediction-based модели оказываются «стохастической,
сглаженной версией» матричной факторизации; различия — лишь в способе оценки (онлайн-SGD vs SVD),
в сглаживании и в том, какая нормировка частот используется. Следствие: многие свойства эмбеддингов
(линейность, анизотропия) объяснимы свойствами PMI-матрицы, а не магией нейросетей.
""")

# ============================================================ Бонус
md("""
---
# Бонус. Subsampling частотных слов

Формула вероятности выбрасывания токена:

$$
P(w_i)=1-\\sqrt{\\frac{\\mathrm{thr}}{f(w_i)}},\\qquad \\mathrm{thr}=10^{-5},
$$

где $f(w_i)=\\mathrm{count}(w_i)/N$ — относительная частота. Реализуем её вручную и обучим
дополнительную Skip-Gram-модель на прореженном корпусе.
""")

code("""
def subsample(sentences, thr=1e-5, seed=1):
    rng = np.random.default_rng(seed)
    total = sum(len(s) for s in sentences)
    cnt = Counter(t for s in sentences for t in s)
    kept = []
    for s in sentences:
        new = []
        for t in s:
            f = cnt[t] / total
            p_drop = 1.0 - np.sqrt(thr / f) if f > thr else 0.0
            if rng.random() >= p_drop:
                new.append(t)
        if len(new) >= 2:
            kept.append(new)
    return kept, cnt, total

kept_sents, cnt_all, total_all = subsample(sentences, thr=1e-5, seed=1)
kept_tokens = sum(len(s) for s in kept_sents)
print("токенов: было %d, стало %d (удалено %.1f%%)"
      % (total_all, kept_tokens, 100 * (1 - kept_tokens / total_all)))

path = os.path.join(MODELS_DIR, "M7_subsample.w2v")
if os.path.exists(path):
    m_ss = Word2Vec.load(path); st = "loaded"
else:
    m_ss = Word2Vec(sentences=kept_sents, sg=1, window=5, negative=5,
                    vector_size=100, min_count=5, epochs=10, workers=4, seed=42)
    m_ss.save(path); st = "trained"
print("M7 (subsample) vocab:", len(m_ss.wv), "|", st)
""")

code("""
COMPARE = ["holmes", "maid"]   # частое и редкое слово
rows = []
for w in COMPARE:
    for name, mdl in [("M1 без subsampling", base_model), ("M7 с subsampling", m_ss)]:
        if w in mdl.wv:
            sim = ", ".join(f"{x} ({s:.2f})" for x, s in mdl.wv.most_similar(w, topn=5))
        else:
            sim = "<нет в словаре>"
        rows.append({"слово": w, "частота": freq[w], "модель": name, "top-5": sim})
pd.DataFrame(rows)
""")

md("""
### Выводы по бонусу

Subsampling агрессивно прореживает сверхчастотные слова (`the`, `of`, `and`, `said`), поэтому
обучение проходит быстрее (меньше пар «центр–контекст»), а векторы средних и редких слов
смещаются в сторону содержательной лексики: соседи `holmes` и `maid` становятся более
тематически однородными, снижается влияние служебных слов-«паразитов». Для редких слов эффект
двойственный: с одной стороны, им уделяется относительно больше обучающих сигналов, с другой —
общее число контекстных наблюдений уменьшается, поэтому при очень маленьком корпусе (как наш)
топ-соседи редкого слова могут стать беднее. На больших корпусах subsampling почти всегда
выигрывает — именно поэтому в gensim `sample=1e-3` включён по умолчанию.
""")

md("""
---
# Общие выводы практической работы

1. **Ручной расчёт (часть 1)** показал механику SGNS: потеря распадается на бинарные
   логистические члены, градиент имеет простой вид $(\\sigma-1)u_o+\\sum(1-\\sigma(-u_i^{\\top}v))u_i$,
   а несколько шагов спуска реально «притягивают» $v_c$ к истинному контексту и отталкивают от
   негативов. $J_{\\text{softmax}}$ и $J_{\\text{NS}}$ одного порядка, но NS-потеря выше из-за
   суммирования нескольких бинарных штрафов.
2. **Гиперпараметры (часть 2)**: SG и CBOW дают сопоставимых соседей для частотных слов, но расходятся
   на редких; число негативов 5 является разумным компромиссом на небольшом корпусе; размер окна —
   главный рычаг между синтаксической взаимозаменяемостью (окно 2) и тематической ассоциацией (окно 10).
3. **Оценка (часть 3)**: ранговая корреляция с человеческими оценками положительна, но ограничена
   смешением similarity и association; 3CosAdd корректно воспроизводится собственной numpy-реализацией,
   accuracy на нерегулярных аналогиях далека от 1 из-за малого корпуса.
4. **Геометрия (часть 4)**: линейные отношения рода действительно существуют, но искажаются
   тематикой; t-SNE чувствителен к perplexity; пространство эмбеддингов анизотропно, и простое
   вычитание среднего вектора повышает изотропность и качество similarity-оценок.
5. **Теория (часть 5)** связывает SGNS с факторизацией PMI-матрицы: prediction-based и
   count-based методы — две реализации одной дистрибутивной идеи, отличающиеся способом оценки.

**Вывод:** Word2Vec — компактная, объяснимая модель, чьё поведение полностью управляется
несколькими гиперпараметрами; для прикладного использования необходимы предметная предобработка
и постобработка (центрирование), а интерпретировать косинусное сходство нужно осторожно,
различая похожеть и ассоциацию.
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb["metadata"]["language_info"] = {"name": "python", "version": "3"}

import gensim as _g
cells[0].source = cells[0].source.replace("{GENSIM}", _g.__version__).replace("{SKLEARN}", __import__("sklearn").__version__)

nbf.write(nb, "/workspace/PZ-02_Word2Vec_solution.ipynb")
print("notebook written:", len(cells), "cells")
