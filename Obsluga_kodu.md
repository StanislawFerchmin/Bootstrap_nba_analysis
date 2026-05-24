# Obsługa kodu — Analiza seryjności rzutów NBA (Bootstrap)

## Spis treści
1. [Instalacja Pythona i środowisko wirtualne](#1-instalacja-pythona-i-środowisko-wirtualne)
2. [Opis i obsługa każdego pliku .py](#2-opis-i-obsługa-każdego-pliku-py)
3. [Mapa tematyczna: Seminarium ↔ Kod Python](#3-mapa-tematyczna-seminarium--kod-python)

---

## 1. Instalacja Pythona i środowisko wirtualne

### 1.1 Instalacja Pythona

1. Przejdź na stronę [https://www.python.org/downloads/](https://www.python.org/downloads/) i pobierz **Python 3.10 lub nowszy**.
2. Podczas instalacji zaznacz opcję **„Add Python to PATH"** (krytyczne dla działania z terminala).
3. Zweryfikuj instalację — otwórz terminal i wpisz:

```bash
python --version
```

Oczekiwany wynik: `Python 3.10.x` lub nowszy.

### 1.2 Tworzenie środowiska wirtualnego

Otwórz terminal w katalogu projektu (`licencjat/`) i wykonaj poniższe polecenia:

```powershell
# Windows — PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
python -m venv venv
source venv/bin/activate
```

Po aktywacji na początku wiersza poleceń pojawi się `(venv)`.

### 1.3 Instalacja zależności

```bash
pip install -r requirements.txt
```

Zainstalowane zostaną: `flask`, `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn`, `nba_api`.

### 1.4 Deaktywacja środowiska

```bash
deactivate
```

### 1.5 Wymagane pliki danych

| Plik | Skąd pochodzi | Do czego służy |
|---|---|---|
| `nba_cleaned.csv` | dostarczone z projektem | wykresy rozdziału 4 (`generate_plots.py`) |
| `shots_cache.csv` | dostarczone z projektem lub pobrany przez `scraper.py` | aplikacja Flask (`app.py`) |

Oba pliki powinny znajdować się w katalogu `licencjat/`.

---

## 2. Opis i obsługa każdego pliku .py

---

### `scraper.py` — pobieranie danych z NBA API

**Przeznaczenie:** Pobiera sekwencje rzutów 5 najlepszych strzelców z 4 drużyn (SAS, OKC, NYK, CLE; finaliści konferencji w obecnym sezonie)  z sezonu 2025-26, oddzielnie dla sezonu zasadniczego i play-offów.

**Uruchomienie (force-refresh z API):**
```bash
python scraper.py
```
Ostrzeżenie: Mojemu laptopowi zajęło to ~ **20/30 minut** ze względu na limity zapytań API. Dane są zapisywane do `shots_cache.csv`.

**Użycie jako moduł:**
```python
from scraper import fetch_shot_sequences

df = fetch_shot_sequences()                    # wczytuje cache jeśli istnieje
df = fetch_shot_sequences(force_refresh=True)  # wymusza pobranie z API
```

**Schemat danych wyjściowych (`shots_cache.csv`):**

| Kolumna | Typ | Opis |
|---|---|---|
| `player_name` | str | Imię i nazwisko zawodnika |
| `player_id` | str | Identyfikator zawodnika |
| `team_abbr` | str | Skrót drużyny: `OKC`, `SAS`, `NYK`, `CLE` |
| `game_id` | str | 10-cyfrowy identyfikator meczu |
| `game_date` | str | Data meczu (`YYYY-MM-DD`) |
| `event_num` | int | Numer zdarzenia w meczu (porządek chronologiczny) |
| `shot_result` | int | `1` = trafiony, `0` = chybiony |
| `season_type` | str | `RS` = sezon zasadniczy, `PO` = play-off |

**Przykładowe wiersze `shots_cache.csv`:**

```
        player_name  player_id team_abbr       game_id   game_date  event_num  shot_result season_type
0  Donovan Mitchell  mitchdo01       CLE  202604180CLE  2026-04-18          6            0          PO
1  Donovan Mitchell  mitchdo01       CLE  202604180CLE  2026-04-18         14            0          PO
2  Donovan Mitchell  mitchdo01       CLE  202604180CLE  2026-04-18         35            1          PO
3  Donovan Mitchell  mitchdo01       CLE  202604180CLE  2026-04-18        108            1          PO
4  Donovan Mitchell  mitchdo01       CLE  202604180CLE  2026-04-18        135            1          PO
```

Każdy wiersz to jeden rzut. Rzuty z tego samego meczu (`game_id`) ułożone są chronologicznie według `event_num` — ta kolejność jest wymagana przez `bootstrap_engine.py` do poprawnego liczenia serii.

---

### `bootstrap_stats.py` — ogólny bootstrap i regresja OLS

**Przeznaczenie:** Klasy i funkcje do bootstrapowych przedziałów ufności, analizy Monte Carlo dla regresji liniowej oraz obliczania RSS. Odpowiada treści sekcji 4.1 i 4.2 seminarium.

**Uruchomienie (demo: regresja DRIBBLES → TOUCH_TIME):**
```bash
python bootstrap_stats.py
```
Generuje: `bootstrap_regression_coefs.png` i `bootstrap_rss_diff.png`.

**Kluczowe obiekty:**

| Obiekt | Typ | Opis |
|---|---|---|
| `Bootstrap(data)` | klasa | Inicjalizacja próby bootstrapowej |
| `Bootstrap.boot_sampling(B)` | metoda | Zwraca macierz `(B × n)` — B próbek ze zwróceniem |
| `Bootstrap.samples_statistics(B)` | metoda | Statystyki dla każdej próbki (mean, std, kwantyle) |
| `confidence_interval(data, alpha, side)` | funkcja | Percentylowy CI: `"2sided"` / `"Lsided"` / `"Rsided"` |
| `ci_width_analysis(data, sample_sizes)` | funkcja | Szerokość CI jako funkcja rozmiaru próbki bootstrapowej |
| `plot_ci_width(sizes, lowers, uppers)` | funkcja | Wykres CI bounds + szerokości |
| `reg_stats(X, Y)` | funkcja | Parametry OLS przez `scipy.stats.linregress` — zwraca `(intercept, slope)` |
| `mc_bootstrap(X, Y, B, alpha)` | funkcja | Monte Carlo bootstrap: B replik regresji → rozkład b0, b1, 95% CI |
| `rss(Y, X, intercept, slope)` | funkcja | RSS = Σ(Y − b0 − b1·X)², zgodnie z równaniami 4.1–4.2 |
| `bootstrap_regression_demo(csv_path)` | funkcja | Pełna analiza demonstracyjna dla DRIBBLES → TOUCH_TIME |

**Przykład użycia:**
```python
from bootstrap_stats import Bootstrap, confidence_interval, mc_bootstrap
import numpy as np

# Bootstrapowy CI dla średniej
bs = Bootstrap(data)
boot_means = bs.boot_sampling(1000).mean(axis=1)
print(confidence_interval(boot_means, alpha=0.05, side="2sided"))
# -> Przedział ufności na poziomie 0.95: (62.1234, 65.8901)

# Bootstrap regresji liniowej
result = mc_bootstrap(X, Y, B=10_000)
print(result["b0"]["conf_interval"])  # 95% CI dla wyrazu wolnego
print(result["b1"]["conf_interval"])  # 95% CI dla współczynnika kierunkowego
```

---

### `bootstrap_engine.py` — test Walda-Wolfowitza i bootstrap parametryczny

**Przeznaczenie:** Implementacja całego potoku statystycznego opisanego w seminarium: test serii Walda-Wolfowitza → agregacja Z → obserwowana statystyka ΔZ → bootstrap parametryczny → wartość p.

**Uruchomienie (self-test na danych syntetycznych):**
```bash
python bootstrap_engine.py
```

**Kluczowe obiekty:**

| Obiekt | Typ | Opis |
|---|---|---|
| `count_runs(shots)` | funkcja | Liczy liczbę serii R w binarnej sekwencji rzutów |
| `ww_z_score(shots)` | funkcja | Oblicza Z Walda-Wolfowitza dla jednego meczu; zwraca `(n, p, Z)` lub `None` |
| `aggregate_z(game_stats)` | funkcja | Z_s = Σ(n_i · Z_i) / Σ(n_i) — agregacja ważona objętością |
| `GameRecord` | dataclass | Dane meczu: `game_id`, `season_type`, `n`, `p`, `z` |
| `BootstrapResult` | dataclass | Wyniki analizy zawodnika: `delta_z_obs`, `z_rs`, `z_po`, `p_value`, `delta_z_boot` |
| `BootstrapResult.significant` | property | `True` jeśli `p_value < 0.05` |
| `BootstrapResult.summary()` | metoda | Słownik z kluczowymi wynikami (Z_RS, Z_PO, ΔZ, p) |
| `build_game_records(player_df)` | funkcja | Tworzy listę `GameRecord` z DataFrame zawodnika |
| `run_bootstrap(player_df, name, B)` | funkcja | Pełna analiza bootstrapowa jednego zawodnika |
| `analyse_all_players(df, B)` | funkcja | Uruchamia `run_bootstrap` dla każdego zawodnika w DataFrame |

**Przykład użycia:**
```python
from bootstrap_engine import run_bootstrap, analyse_all_players
import pandas as pd

df = pd.read_csv("shots_cache.csv")
results = analyse_all_players(df, B=5000)

for name, res in results.items():
    s = res.summary()
    print(f"{name}: ΔZ={s['delta_Z']:+.3f}, p={s['p_value']:.4f}, "
          f"istotny={s['significant']}")
# -> Donovan Mitchell: ΔZ=+0.142, p=0.3820, istotny=False
```

---

### `plots.py` — wykresy dla aplikacji Flask

**Przeznaczenie:** Fabryki wykresów Matplotlib/Seaborn zwracające ciągi base64 PNG do bezpośredniego osadzenia w szablonach HTML (`<img src="data:image/png;base64,...">`). Używany wyłącznie przez `app.py`.

| Funkcja | Wejście | Treść wykresu |
|---|---|---|
| `plot_bootstrap_distribution(result)` | `BootstrapResult` | Rozkład zerowy ΔZ\* z zaznaczonym ΔZ i obszarem p-value |
| `plot_z_comparison(result)` | `BootstrapResult` | Poziomy słupkowy: Z_RS vs Z_PO |
| `plot_game_timeline(records, name)` | lista słowników | Oś czasu Z_i według meczu, kolor = typ sezonu |
| `plot_fg_vs_z(records, name)` | lista słowników | FG% × 100 a Z_i, rozmiar bąbla = liczba rzutów |
| `plot_all_players_delta_z(results)` | dict `BootstrapResult` | ΔZ dla wszystkich zawodników (czerwony = p < 0.05) |

---

### `app.py` — aplikacja webowa Flask

**Przeznaczenie:** Interaktywny dashboard do przeglądania wyników analizy seryjności przez przeglądarkę.

**Uruchomienie:**
```bash
python app.py
```
Następnie otwórz `http://localhost:5050`.

**Zachowanie przy starcie:**
- Przy pierwszym żądaniu aplikacja automatycznie wczytuje `shots_cache.csv`.
- Bootstrap jest obliczany dla każdego zawodnika (B=5000) — może potrwać kilka sekund.
- Przycisk **Refresh Data** (trasa `POST /refresh`) pobiera dane na nowo z API NBA.

**Trasy:**

| Trasa | Metoda | Opis |
|---|---|---|
| `/` | GET | Strona główna — wyszukiwanie zawodnika i drużyny |
| `/player/<name>` | GET | Pełna analiza jednego zawodnika (4 wykresy + tabela meczów) |
| `/overview` | GET | Zestawienie ΔZ dla wszystkich zawodników |
| `/team/<abbr>` | GET | Zestawienie ΔZ przefiltrowane do jednej drużyny |
| `/refresh` | POST | Ponowne pobranie danych z NBA Stats API |
| `/api/players?q=` | GET | JSON z listą zawodników (autocomplete) |

---

### `generate_plots.py` — wykresy rozdziału 4

**Przeznaczenie:** Odtwarza 5 wykresów z rozdziału 4 pracy na tych samych danych i parametrach co dokument. Wymaga pliku `nba_cleaned.csv`.

**Uruchomienie:**
```bash
python generate_plots.py
```

**Schemat danych wejściowych (`nba_cleaned.csv`):**

| Kolumna | Typ | Opis |
|---|---|---|
| `GAME_ID` | int | Identyfikator meczu |
| `MATCHUP` | str | Opis meczu (data, drużyny) |
| `PERIOD` | int | Kwarta meczu (1–4) |
| `DRIBBLES` | int | Liczba dryblowań przed rzutem |
| `TOUCH_TIME` | float | Czas trzymania piłki przed rzutem (sekundy) |
| `SHOT_DIST` | float | Odległość rzutu (stopy) |
| `SHOT_RESULT` | str | `"made"` lub `"missed"` |
| `player_id` | int | Identyfikator zawodnika |

**Przykładowe wiersze `nba_cleaned.csv`** (wybrane kolumny używane przez `generate_plots.py`):

```
    GAME_ID            MATCHUP  PERIOD  DRIBBLES  TOUCH_TIME  SHOT_DIST SHOT_RESULT  player_id
0  21400899  MAR 04, 2015 CHA@BKN       1         2         1.9        7.7        made     203148
1  21400899  MAR 04, 2015 CHA@BKN       1         0         0.8       28.2      missed     203148
2  21400899  MAR 04, 2015 CHA@BKN       2         2         1.9       17.2      missed     203148
3  21400899  MAR 04, 2015 CHA@BKN       2         2         2.7        3.7      missed     203148

[119312 rows × 17 columns]
```

**Generowane pliki:**

| Plik wyjściowy | Sekcja | Treść |
|---|---|---|
| `ch4_fig1_ci_analysis.png` | 4.1 | Przedziały ufności i ich szerokość vs. n (n=894 meczów) |
| `ch4_fig2_params_correlated.png` | 4.2 | Rozkłady b0, b1 — dane skorelowane DRIBBLES→TOUCH\_TIME (r=0,94) |
| `ch4_fig3_rss_correlated.png` | 4.2 | Różnica błędów resztowych RSS — dane skorelowane |
| `ch4_fig4_params_uncorrelated.png` | 4.2 | Rozkłady b0, b1 — dane nieskorelowane Q4 (r≈−0,02) |
| `ch4_fig5_rss_uncorrelated.png` | 4.2 | Różnica błędów resztowych RSS — dane nieskorelowane |

---

## 3. Mapa tematyczna: Seminarium ↔ Kod Python

```
METODA BOOTSTRAP
│
├── Resampling ze zwróceniem
│   └── Bootstrap.boot_sampling(B)                          [bootstrap_stats.py]
│       • tworzy macierz (B × n) indeksów losowych
│
├── Statystyki próbek
│   └── Bootstrap.samples_statistics(B)                     [bootstrap_stats.py]
│       • mean, std, kwantyle dla każdej repliki
│
├── PRZEDZIAŁY UFNOŚCI (Sekcja 4.1)
│   ├── Metoda percentylowa
│   │   └── confidence_interval(data, alpha, side)           [bootstrap_stats.py]
│   │       • "2sided" → [q_α/2, q_{1-α/2}]
│   │       • "Lsided" → (-∞, q_α]
│   │       • "Rsided" → [q_{1-α}, +∞)
│   │
│   ├── Analiza szerokości CI vs. n
│   │   ├── ci_width_analysis(data, sample_sizes)            [bootstrap_stats.py]
│   │   └── plot_ci_width(sizes, lowers, uppers)             [bootstrap_stats.py]
│   │
│   └── Rysunek 1 (ch4_fig1_ci_analysis.png)                [generate_plots.py]
│       • n = 894 meczów (≥ 50 rzutów), seed = 15
│
├── REGRESJA LINIOWA OLS (Sekcja 4.2)
│   ├── Wyznaczenie parametrów b0, b1
│   │   └── reg_stats(X, Y)                                  [bootstrap_stats.py]
│   │       • scipy.stats.linregress → (intercept, slope)
│   │
│   ├── Bootstrap Monte Carlo dla współczynników
│   │   └── mc_bootstrap(X, Y, B, alpha)                     [bootstrap_stats.py]
│   │       • B = 10 000 replik z losowaniem par (X_i, Y_i)
│   │       • zwraca rozkład i CI dla b0 i b1
│   │
│   ├── Rysunek 2 (dane skorelowane, r = 0,942)              [generate_plots.py]
│   │   • DRIBBLES → TOUCH_TIME, n ≈ 27 000
│   │
│   └── Rysunek 4 (dane nieskorelowane, r ≈ −0,016)         [generate_plots.py]
│       • liczba rzutów vs. skuteczność w Q4, n = 281
│
├── RSS — BŁĘDY RESZTOWE (Równania 4.1–4.2, Twierdzenie 4.3)
│   ├── RSS = Σ(Y − b0 − b1·X)²
│   │   └── rss(Y, X, intercept, slope)                      [bootstrap_stats.py]
│   │       • np.sum (nie np.mean) — zgodnie z twierdzeniem
│   │
│   ├── RSS* − RSS → 0 p.w. (Twierdzenie 4.3)
│   │   └── diff_c / diff_u: [rss(..., b_b, s_b) - real_rss] [generate_plots.py]
│   │
│   ├── Rysunek 3 (dane skorelowane)                         [generate_plots.py]
│   │   └── ch4_fig3_rss_correlated.png
│   │
│   └── Rysunek 5 (dane nieskorelowane)                      [generate_plots.py]
│       └── ch4_fig5_rss_uncorrelated.png
│
└── ANALIZA SERYJNOŚCI NBA
    │
    ├── TEST SERII WALDA-WOLFOWITZA
    │   ├── Liczba serii R
    │   │   └── count_runs(shots)                            [bootstrap_engine.py]
    │   │       • R = suma miejsc gdzie shots[i] ≠ shots[i-1], + 1
    │   │
    │   ├── E[R] = 2·n·p·q + 1
    │   ├── Var[R] = 4·n·p·q·(1 − 3·p·q)
    │   ├── Z = (R − E[R]) / √Var[R]
    │   │   └── ww_z_score(shots)                            [bootstrap_engine.py]
    │   │       • zwraca (n, p, Z) lub None gdy degeneracja
    │   │       • degeneracja: n < 3, p = 0 lub p = 1
    │   │
    │   └── GameRecord                                       [bootstrap_engine.py]
    │       • dataclass: game_id, season_type, n, p, z
    │
    ├── AGREGACJA WAŻONA OBJĘTOŚCIĄ
    │   ├── Z_ctx = Σ(n_i · Z_i) / Σ(n_i)
    │   └── aggregate_z(game_stats)                          [bootstrap_engine.py]
    │
    ├── BOOTSTRAP PARAMETRYCZNY
    │   ├── ΔZ = Z_PO − Z_RS  (obserwowana statystyka testowa)
    │   │
    │   ├── B = 5000 replik:
    │   │   • dla każdego meczu i: losuj n_i rzutów z Bernoulli(p̂_i)
    │   │   • przelicz Z_RS*, Z_PO*, ΔZ* = Z_PO* − Z_RS*
    │   │   └── _simulate_context_z(game_params, rng)        [bootstrap_engine.py]
    │   │
    │   ├── Wartość p = (1/B) · Σ 𝟙{|ΔZ*| ≥ |ΔZ|}
    │   │   └── BootstrapResult.p_value                      [bootstrap_engine.py]
    │   │
    │   ├── run_bootstrap(player_df, name, B)                [bootstrap_engine.py]
    │   │   • pełna analiza jednego zawodnika
    │   │
    │   └── analyse_all_players(df, B)                       [bootstrap_engine.py]
    │       • uruchamia run_bootstrap dla wszystkich
    │
    ├── DANE NBA
    │   ├── Pobieranie z API                                 [scraper.py]
    │   │   • leaguedashplayerstats → top-5 strzelców
    │   │   • shotchartdetail → sekwencje rzutów
    │   │
    │   └── shots_cache.csv
    │       • jedna linia = jeden rzut, posortowane wg event_num
    │
    └── WIZUALIZACJA WYNIKÓW
        ├── Rozkład zerowy ΔZ* + p-value
        │   └── plot_bootstrap_distribution(result)          [plots.py]
        │
        ├── Porównanie Z_RS vs Z_PO
        │   └── plot_z_comparison(result)                    [plots.py]
        │
        ├── Oś czasu Z_i per mecz
        │   └── plot_game_timeline(records, name)            [plots.py]
        │
        ├── FG% vs seryjność
        │   └── plot_fg_vs_z(records, name)                  [plots.py]
        │
        └── Zestawienie ΔZ — wszyscy zawodnicy
            └── plot_all_players_delta_z(results)            [plots.py]
```
