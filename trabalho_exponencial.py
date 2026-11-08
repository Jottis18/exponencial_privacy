"""Trabalho 4 - Privacidade Diferencial: mecanismo Exponencial no r-N (Adult Income Dataset).

Etapas:
  1. pré-processa a base: One-Hot nas colunas categóricas e MinMax em age e educational-num;
  2. separa exatamente 1000 tuplas de teste (500 de cada classe) e deixa o resto para treino;
  3. para cada raio r, conta os vizinhos de cada classe dentro do raio e roda o r-N
     tradicional (sem privacidade);
  4. roda o r-N diferencialmente privado (Algoritmo 2 de Gursoy et al. (2017), mecanismo
     Exponencial) para cada raio r e cada ε;
  5. salva as predições, as tabelas e os gráficos.

Uso:
  python trabalho_exponencial.py                roda tudo (lê o adult.csv desta pasta)
  python trabalho_exponencial.py --detalhar 7   roda tudo e mostra o passo a passo da
                                                tupla de teste 7 (útil na demo)
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
from tqdm import tqdm

PASTA_PROJETO = os.path.dirname(os.path.abspath(__file__))

COLS_TO_DROP = ["fnlwgt", "education", "capital-gain", "capital-loss", "hours-per-week"]
TARGET_COL = "income"
TEST_SIZE = 1000
RADII = [0.5, 1.0, 1.5, 2.0, 6.0]
EPSILONS = [0.1, 0.5, 1.0, 5.0, 10.0, 100.0]

# Sensibilidade global da função de utilidade u(x, I) = c_{x,r}(I), o número de vizinhos da
# classe I no raio r de x: incluir ou remover uma tupla do treino muda a contagem de uma
# única classe em no máximo 1. Logo Δu = 1.
SENSITIVITY = 1.0


# 1. PRÉ-PROCESSAMENTO DO DATASET ADULT INCOME
def preprocess_data(data_path):
    # Carregar o arquivo CSV
    df = pd.read_csv(data_path, skipinitialspace=True)
    print(f"Base original: {len(df)} tuplas, {df.shape[1]} atributos")

    # Remover os atributos pedidos no enunciado (dá erro se algum nome não existir na base)
    df = df.drop(columns=COLS_TO_DROP)

    # Remover as tuplas com valores faltantes, representados por "?"
    num_rows = len(df)
    df = df.replace("?", np.nan).dropna().reset_index(drop=True)
    print(f"Tuplas removidas por conter '?': {num_rows - len(df)}")

    # Classe em ordem alfabética: <=50K -> 0 e >50K -> 1
    target = df[TARGET_COL].astype("category")
    class_names = np.array(target.cat.categories)
    y = target.cat.codes.to_numpy(dtype=np.int64)

    # Cada atributo vira um bloco de colunas com valores em [0, 1]
    attributes = [col for col in df.columns if col != TARGET_COL]
    blocks = []
    encoding = []  # (coluna codificada, atributo original, regra de codificação)
    num_numeric = num_categorical = 0
    for col in attributes:
        if pd.api.types.is_numeric_dtype(df[col]):
            # MinMax: v' = (v - min) / (max - min), que leva o atributo para [0, 1]
            values = df[col].to_numpy(dtype=float)
            low, high = values.min(), values.max()
            blocks.append(((values - low) / (high - low))[:, None])
            encoding.append((col, col, f"MinMax: (v - {low:g}) / ({high:g} - {low:g})"))
            num_numeric += 1
        else:
            # One-Hot: uma coluna 0/1 para cada valor distinto do atributo; cada tupla tem 1
            # só na coluna do seu valor
            column = df[col].astype("category")
            categories = column.cat.categories
            blocks.append(np.eye(len(categories))[column.cat.codes])
            encoding += [
                (f"{col}={value}", col, f"One-Hot: 1 se {col} = {value}, senão 0")
                for value in categories
            ]
            num_categorical += 1
    X = np.hstack(blocks)

    print(
        f"Pré-processada: {len(df)} tuplas | {len(attributes)} atributos viram "
        f"{X.shape[1]} colunas ({num_numeric} MinMax + One-Hot de {num_categorical} "
        "categóricos)"
    )
    # Dois valores diferentes de um atributo categórico somam 1² + 1² = 2 ao quadrado da
    # distância; um atributo MinMax soma no máximo 1. Daí a maior distância possível.
    max_distance = np.sqrt(2 * num_categorical + num_numeric)
    print(
        f"Maior distância possível entre duas tuplas: √(2·{num_categorical} + "
        f"{num_numeric}) = {max_distance:.2f}"
    )

    return X, y, class_names, df[attributes], encoding


# 2. SEPARAÇÃO TREINO/TESTE: 1000 TUPLAS DE TESTE, BALANCEADAS ENTRE AS CLASSES
def balanced_train_test_split(y, rng, test_size=1000):
    # Sorteia test_size / L tuplas de cada classe (sem reposição) para o teste; o resto é
    # treino. Devolve os índices, para a demo conseguir mostrar a tupla original.
    classes = np.unique(y)
    if test_size % len(classes):
        raise ValueError("o tamanho do teste precisa ser divisível pelo número de classes")
    per_class = test_size // len(classes)

    test_idx = np.concatenate(
        [rng.choice(np.flatnonzero(y == c), per_class, replace=False) for c in classes]
    )
    test_idx = rng.permutation(test_idx)  # mistura as classes no teste
    train_idx = np.setdiff1d(np.arange(len(y)), test_idx)

    return train_idx, test_idx


# 3. CONSULTA DE CONTAGEM NO RAIO r (c_{x,r}(I) DO ARTIGO) E r-N TRADICIONAL
def euclidean_distances(X_train, x):
    # Distância euclidiana de x a cada tupla de treino: raiz(Σ_j (x_j - t_j)²)
    diff = X_train - x
    return np.sqrt(np.sum(diff * diff, axis=1))


def count_within_radii(X_train, y_train, X_test, radii, num_classes):
    # counts[j, i, I] = c_{x_i, r_j}(I): quantas tuplas de treino da classe I estão a
    # distância <= r_j da tupla de teste x_i (Definição 8 do artigo). As distâncias de cada
    # tupla de teste são calculadas uma vez só e servem para todos os raios.
    counts = np.zeros((len(radii), len(X_test), num_classes), dtype=np.int64)
    for i, x_t in enumerate(tqdm(X_test, desc="Contagens nos raios")):
        distances = euclidean_distances(X_train, x_t)
        for j, r in enumerate(radii):
            counts[j, i] = np.bincount(y_train[distances <= r], minlength=num_classes)

    return counts


def rn_classify(counts, rng):
    # r-N tradicional (Definição 9 do artigo): a classe com mais vizinhos no raio.
    # Empates, inclusive o raio vazio (0 x 0), são decididos por sorteio entre as classes
    # empatadas, como o próprio artigo sugere.
    predictions = np.empty(len(counts), dtype=np.int64)
    for i, row in enumerate(counts):
        tied = np.flatnonzero(row == row.max())
        predictions[i] = tied[0] if len(tied) == 1 else rng.choice(tied)

    return predictions


# 4. MECANISMO EXPONENCIAL E r-N DIFERENCIALMENTE PRIVADO
def exponential_mechanism(utilities, epsilon, sensitivity, rng):
    # Mecanismo Exponencial (Definição 5 do artigo), uma chamada por linha de utilities:
    # devolve a saída I com probabilidade proporcional a exp(ε · u(I) / (2Δu)).
    scores = epsilon * utilities / (2 * sensitivity)
    # Subtrair o maior expoente da linha não muda as probabilidades (o fator cancela na
    # divisão) e evita overflow: com r = 6 e ε = 100 o expoente passaria de 1600.
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    probabilities = weights / weights.sum(axis=1, keepdims=True)

    # Sorteio pela CDF: com U ~ Uniforme[0, 1), escolhe o primeiro I tal que
    # P(0) + ... + P(I) > U, o que acontece com probabilidade exatamente P(I)
    cdf = np.cumsum(probabilities, axis=1)
    u = rng.random((len(probabilities), 1))
    choices = np.minimum((cdf <= u).sum(axis=1), probabilities.shape[1] - 1)

    return choices, probabilities


def algorithm2_private_rn(counts, epsilon_query, rng):
    # Algoritmo 2 de Gursoy et al. (2017): para cada tupla de teste (x, r), devolve a classe
    # I com probabilidade ∝ exp(ε' · c_{x,r}(I) / 2). A função de utilidade é a própria
    # contagem no raio, com Δc = 1. Cada linha de counts é uma chamada com orçamento ε'.
    return exponential_mechanism(counts, epsilon_query, SENSITIVITY, rng)


def run_rn_dp(counts, epsilon, rng):
    # As |X_test| tuplas de teste consultam a mesma base de treino, então dividem ε
    # (composição sequencial): ε' = ε / |X_test| em cada chamada. Diferente do trabalho 3,
    # não há divisão por L: cada chamada do mecanismo Exponencial escolhe a classe de uma
    # vez só, olhando as L contagens juntas.
    epsilon_query = epsilon / len(counts)
    return algorithm2_private_rn(counts, epsilon_query, rng)


# 5. AVALIAÇÃO E ARQUIVOS DE SAÍDA
def calculate_accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


def save_predictions(path, y_true, y_pred, class_names, counts=None):
    # Uma linha por tupla de teste: id, classe real e classe prevista. Só o arquivo do r-N
    # tradicional (que não é privado) traz as contagens exatas no raio. Os do r-N privado
    # trazem só a classe sorteada, que é a única saída do mecanismo: publicar as contagens
    # ou as probabilidades exatas junto com ela quebraria a privacidade diferencial.
    df = pd.DataFrame(
        {
            "id_teste": np.arange(len(y_true)),
            "classe_real": class_names[y_true],
            "classe_prevista": class_names[y_pred],
        }
    )
    if counts is not None:
        for c, name in enumerate(class_names):
            df[f"vizinhos_{name}"] = counts[:, c]
    df.to_csv(path, index=False)


def radius_statistics(counts):
    # Resumo da vizinhança de um raio: tamanho, raios vazios, empates e a diferença entre
    # as duas maiores contagens (a "margem" que o mecanismo Exponencial precisa enxergar)
    total = counts.sum(axis=1)
    sorted_counts = np.sort(counts, axis=1)
    margin = sorted_counts[:, -1] - sorted_counts[:, -2]
    return {
        "media_vizinhos": round(total.mean(), 1),
        "mediana_vizinhos": float(np.median(total)),
        "pct_raio_vazio": round(np.mean(total == 0) * 100, 1),
        "pct_empate": round(np.mean(margin == 0) * 100, 1),
        "mediana_margem": float(np.median(margin)),
    }


# 6. GRÁFICOS
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
AXIS_COLOR = "#c3c2b7"
GRID_COLOR = "#e1e0d9"
# Uma cor por raio (paleta categórica validada para daltonismo) e um marcador diferente
# por raio, para o gráfico continuar legível impresso em preto e branco
RADIUS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
RADIUS_MARKERS = ["o", "s", "D", "^", "v"]
REFERENCE_DASH = (0, (5, 3))


def format_pt(value, decimals=1):
    # Número com vírgula decimal, para os rótulos dos gráficos
    return f"{value:.{decimals}f}".replace(".", ",")


def format_eps(eps):
    return f"{eps:g}".replace(".", ",")


def style_axes(ax, epsilons, y_limits):
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.set_xticks(epsilons, labels=[format_eps(e) for e in epsilons])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(epsilons[0] / 1.4, epsilons[-1] * 1.4)
    ax.set_ylim(*y_limits)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_COLOR)
    ax.tick_params(colors=INK_SECONDARY, length=0, pad=6)


def accuracy_limits(results, rn_accuracies):
    # Eixo y em múltiplos de 5% que cobrem todas as acurácias, com uma folga para os marcadores
    values = [acc * 100 for acc in rn_accuracies.values()]
    values += [row["acuracia_rn_dp"] * 100 for row in results]
    return np.floor((min(values) - 1) / 5) * 5, np.ceil((max(values) + 1) / 5) * 5


def plot_accuracy(results, rn_accuracies, num_test, path):
    # Gráfico pedido no enunciado: acurácia por ε, uma linha por raio r, e a acurácia do r-N
    # sem privacidade de cada raio como referência (linha tracejada na mesma cor)
    fig, ax = plt.subplots(figsize=(10, 5.6), facecolor=SURFACE)
    style_axes(ax, EPSILONS, accuracy_limits(results, rn_accuracies))

    for j, r in enumerate(RADII):
        color, marker = RADIUS_COLORS[j], RADIUS_MARKERS[j]
        ax.axhline(
            rn_accuracies[r] * 100,
            color=color,
            linestyle=REFERENCE_DASH,
            linewidth=1.5,
            zorder=2,
        )
        dp_pct = [row["acuracia_rn_dp"] * 100 for row in results if row["raio"] == r]
        ax.plot(
            EPSILONS,
            dp_pct,
            color=color,
            linewidth=2,
            marker=marker,
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=3,
        )

    # Legenda em dois grupos: a cor/marcador identifica o raio; o traço, o classificador
    radius_handles = [
        Line2D(
            [],
            [],
            color=RADIUS_COLORS[j],
            marker=RADIUS_MARKERS[j],
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            linewidth=2,
            label=f"r = {format_eps(r)} (sem privacidade: {format_pt(rn_accuracies[r] * 100)}%)",
        )
        for j, r in enumerate(RADII)
    ]
    style_handles = [
        Line2D([], [], color=INK_SECONDARY, linewidth=2, label="r-N privado (Exponencial)"),
        Line2D(
            [],
            [],
            color=INK_SECONDARY,
            linewidth=1.5,
            linestyle=REFERENCE_DASH,
            label="r-N sem privacidade",
        ),
    ]
    legend_radius = ax.legend(
        handles=radius_handles,
        title="Raio",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        labelcolor=INK,
        alignment="left",
    )
    legend_radius.get_title().set_color(INK_SECONDARY)
    ax.add_artist(legend_radius)
    legend_style = ax.legend(
        handles=style_handles,
        title="Classificador",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.5),
        labelcolor=INK,
        alignment="left",
        handlelength=2.6,
    )
    legend_style.get_title().set_color(INK_SECONDARY)

    ax.set_title(
        "Acurácia do r-N com e sem privacidade diferencial",
        loc="left",
        fontsize=13,
        color=INK,
        pad=26,
    )
    num_test_pt = f"{num_test:,}".replace(",", ".")
    ax.text(
        0,
        1.03,
        f"Adult Income · mecanismo Exponencial · {num_test_pt} tuplas de teste "
        f"(metade de cada classe) · ε' = ε / {num_test_pt} por tupla",
        transform=ax.transAxes,
        fontsize=10,
        color=INK_SECONDARY,
    )
    ax.set_xlabel("Orçamento de privacidade (ε)", color=INK_SECONDARY)
    ax.set_ylabel("Acurácia no conjunto de teste (%)", color=INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)


def plot_accuracy_by_radius(results, rn_accuracies, num_test, path):
    # O mesmo resultado em um painel por raio, para ler cada curva sem sobreposição: a
    # acurácia realizada (pontos), a esperada (linha fina) e a referência sem privacidade
    y_limits = accuracy_limits(results, rn_accuracies)
    fig, axes = plt.subplots(
        1, len(RADII), figsize=(15, 4.2), sharey=True, facecolor=SURFACE
    )

    for j, (ax, r) in enumerate(zip(axes, RADII)):
        style_axes(ax, EPSILONS, y_limits)
        ax.tick_params(axis="x", labelsize=8)
        color, marker = RADIUS_COLORS[j], RADIUS_MARKERS[j]
        rows = [row for row in results if row["raio"] == r]
        dp_pct = [row["acuracia_rn_dp"] * 100 for row in rows]
        expected_pct = [row["acuracia_esperada_rn_dp"] * 100 for row in rows]
        rn_pct = rn_accuracies[r] * 100

        ax.axhline(rn_pct, color=INK_SECONDARY, linestyle=REFERENCE_DASH, linewidth=1.5)
        ax.plot(EPSILONS, expected_pct, color=color, linewidth=1, alpha=0.7)
        ax.plot(
            EPSILONS,
            dp_pct,
            color=color,
            linewidth=0,
            marker=marker,
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
        )
        ax.annotate(
            f"{format_pt(rn_pct)}%",
            (EPSILONS[0], rn_pct),
            textcoords="offset points",
            xytext=(-4, 6),
            ha="left",
            fontsize=9,
            color=INK_SECONDARY,
        )
        ax.annotate(
            f"{format_pt(dp_pct[-1])}%",
            (EPSILONS[-1], dp_pct[-1]),
            textcoords="offset points",
            xytext=(0, -16),
            ha="center",
            fontsize=9,
            color=INK,
        )
        ax.set_title(f"r = {format_eps(r)}", loc="left", fontsize=11, color=INK)
        ax.set_xlabel("ε", color=INK_SECONDARY)
    axes[0].set_ylabel("Acurácia no teste (%)", color=INK_SECONDARY)

    handles = [
        Line2D(
            [],
            [],
            color=INK_SECONDARY,
            marker="o",
            markersize=7,
            linewidth=0,
            label="r-N privado: acurácia obtida",
        ),
        Line2D([], [], color=INK_SECONDARY, linewidth=1, label="r-N privado: acurácia esperada"),
        Line2D(
            [],
            [],
            color=INK_SECONDARY,
            linewidth=1.5,
            linestyle=REFERENCE_DASH,
            label="r-N sem privacidade",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        ncol=3,
        frameon=False,
        labelcolor=INK,
        bbox_to_anchor=(1.0, 1.0),
    )
    num_test_pt = f"{num_test:,}".replace(",", ".")
    fig.suptitle(
        f"Acurácia do r-N privado por raio · mecanismo Exponencial · {num_test_pt} tuplas de teste",
        x=0.01,
        y=0.98,
        ha="left",
        fontsize=12,
        color=INK,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=200, facecolor=SURFACE)


# 7. PASSO A PASSO DE UMA TUPLA (DEMO)
def show_details(i, test_rows, y_test, class_names, counts, rn_predictions, dp_results):
    # Mostra, para uma tupla de teste: os atributos originais, as contagens no raio e a
    # predição do r-N para cada r, e as probabilidades do mecanismo Exponencial para cada ε.
    # As probabilidades usam as contagens exatas: servem só para explicar o mecanismo e não
    # fazem parte da saída privada.
    num_test = len(y_test)
    print(f"\n=== Passo a passo da tupla de teste {i} ===")
    for name, value in test_rows.iloc[i].items():
        print(f"  {name:<16}{value}")
    print(f"  classe real: {class_names[y_test[i]]}")

    for j, r in enumerate(RADII):
        row = counts[j, i]
        print(
            f"\nRaio r = {r:g}: vizinhos no raio "
            + ", ".join(f"{class_names[c]} = {n}" for c, n in enumerate(row))
            + f"  =>  r-N prevê {class_names[rn_predictions[r][i]]}"
        )
        for eps in EPSILONS:
            y_pred, probabilities = dp_results[(r, eps)]
            print(
                f"  ε = {eps:<4g} ε' = {eps / num_test:<7.4g} "
                + "  ".join(
                    f"P({class_names[c]}) = {p * 100:6.2f}%"
                    for c, p in enumerate(probabilities[i])
                )
                + f"  =>  sorteou {class_names[y_pred[i]]}"
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dados",
        default=os.path.join(PASTA_PROJETO, "adult.csv"),
        help="CSV do Adult Income Dataset (padrão: adult.csv na pasta do projeto)",
    )
    parser.add_argument(
        "--saida",
        default=os.path.join(PASTA_PROJETO, "resultados"),
        help="pasta onde as predições, as tabelas e os gráficos são salvos",
    )
    parser.add_argument(
        "--semente",
        type=int,
        default=42,
        help="semente da divisão treino/teste, dos desempates e dos sorteios (padrão: 42)",
    )
    parser.add_argument(
        "--detalhar",
        type=int,
        metavar="ID",
        help="mostra o passo a passo de uma tupla de teste (útil na demo)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    predictions_dir = os.path.join(args.saida, "predicoes")
    os.makedirs(predictions_dir, exist_ok=True)

    print("Carregando e pré-processando os dados...")
    X, y, class_names, attributes_df, encoding = preprocess_data(args.dados)
    pd.DataFrame(encoding, columns=["coluna", "atributo_original", "codificacao"]).to_csv(
        os.path.join(args.saida, "codificacao.csv"), index_label="indice"
    )
    num_classes = len(class_names)

    # Semente fixa para os resultados se repetirem na entrega e na demo (num sistema real os
    # sorteios não podem ser previsíveis). O RandomState é o gerador "legado" do NumPy, cuja
    # sequência é congelada: a mesma semente dá os mesmos números em qualquer versão.
    rng = np.random.RandomState(args.semente)
    train_idx, test_idx = balanced_train_test_split(y, rng, test_size=TEST_SIZE)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    test_rows = attributes_df.iloc[test_idx].reset_index(drop=True)
    print(
        f"Treino: {len(X_train)} tuplas | Teste: {len(X_test)} tuplas ("
        + ", ".join(f"{np.sum(y_test == c)} {name}" for c, name in enumerate(class_names))
        + f") | Classes (L = {num_classes})\n"
    )
    if args.detalhar is not None and not 0 <= args.detalhar < len(X_test):
        raise SystemExit(f"--detalhar precisa estar entre 0 e {len(X_test) - 1}")

    # --- 1. Contagens exatas no raio: c_{x,r}(I) para cada tupla de teste e cada raio ---
    counts = count_within_radii(X_train, y_train, X_test, RADII, num_classes)

    # --- 2. r-N tradicional (sem privacidade) para cada raio ---
    print("\nr-N tradicional (sem privacidade)")
    rn_predictions = {}
    rn_accuracies = {}
    radius_rows = []
    for j, r in enumerate(RADII):
        y_pred = rn_classify(counts[j], rng)
        rn_predictions[r] = y_pred
        rn_accuracies[r] = calculate_accuracy(y_test, y_pred)
        save_predictions(
            os.path.join(predictions_dir, f"predicoes_rn_tradicional_r_{r:g}.csv"),
            y_test,
            y_pred,
            class_names,
            counts[j],
        )
        stats = radius_statistics(counts[j])
        radius_rows.append(
            {"raio": r, **stats, "acuracia_rn": round(rn_accuracies[r], 4)}
        )
        print(
            f"  r = {r:<4g} vizinhos no raio: média {stats['media_vizinhos']:>8.1f}, "
            f"mediana {stats['mediana_vizinhos']:>7.0f} | raio vazio {stats['pct_raio_vazio']:4.1f}% "
            f"| empate {stats['pct_empate']:4.1f}% | mediana da margem "
            f"{stats['mediana_margem']:>6.0f} | acurácia {rn_accuracies[r] * 100:.2f}%"
        )

    # --- 3. r-N diferencialmente privado (Algoritmo 2) para cada raio e cada ε ---
    print(
        f"\nr-N privado (Algoritmo 2, mecanismo Exponencial): cada tupla de teste usa "
        f"ε' = ε / |X_test| = ε / {len(X_test)}, com sensibilidade Δu = {SENSITIVITY:g}"
    )
    results = []
    dp_results = {}
    for j, r in enumerate(RADII):
        print(f"  r = {r:g}")
        for eps in EPSILONS:
            y_pred, probabilities = run_rn_dp(counts[j], eps, rng)
            dp_results[(r, eps)] = (y_pred, probabilities)
            save_predictions(
                os.path.join(predictions_dir, f"predicoes_rn_dp_r_{r:g}_eps_{eps:g}.csv"),
                y_test,
                y_pred,
                class_names,
            )

            acc_dp = calculate_accuracy(y_test, y_pred)
            # Acurácia esperada: média, sobre as tuplas de teste, da probabilidade de o
            # mecanismo sortear a classe certa. É o valor em torno do qual a acurácia
            # obtida num sorteio oscila (desvio padrão de até 1,6 ponto com 1000 tuplas).
            expected = probabilities[np.arange(len(y_test)), y_test].mean()
            # Comparação entre a predição do r-N sem privacidade e a predição privada
            agreement = calculate_accuracy(rn_predictions[r], y_pred)
            results.append(
                {
                    "raio": r,
                    "epsilon": eps,
                    "epsilon_por_tupla": eps / len(X_test),
                    "acuracia_rn_dp": round(acc_dp, 4),
                    "acuracia_esperada_rn_dp": round(expected, 4),
                    "concordancia_com_rn": round(agreement, 4),
                    "acuracia_rn": round(rn_accuracies[r], 4),
                }
            )
            print(
                f"    ε = {eps:<4g} ε' = {eps / len(X_test):<7.4g} acurácia "
                f"{acc_dp * 100:6.2f}% (esperada {expected * 100:6.2f}%) | concordância "
                f"com o r-N {agreement * 100:6.2f}%"
            )

    # --- 4. Tabelas e gráficos ---
    pd.DataFrame(results).to_csv(os.path.join(args.saida, "acuracias.csv"), index=False)
    pd.DataFrame(radius_rows).to_csv(
        os.path.join(args.saida, "estatisticas_raio.csv"), index=False
    )
    plot_accuracy(
        results,
        rn_accuracies,
        len(X_test),
        os.path.join(args.saida, "grafico_acuracia_exponencial.png"),
    )
    plot_accuracy_by_radius(
        results,
        rn_accuracies,
        len(X_test),
        os.path.join(args.saida, "grafico_acuracia_por_raio.png"),
    )
    print(f"\nPredições, tabelas e gráficos salvos em {args.saida}")

    if args.detalhar is not None:
        show_details(
            args.detalhar,
            test_rows,
            y_test,
            class_names,
            counts,
            rn_predictions,
            dp_results,
        )

    plt.show()


if __name__ == "__main__":
    main()
