#!/usr/bin/env python3
"""
WC 2026 Predictor — Elo + Poisson + Monte Carlo
Usage:  pip install -r requirements.txt
        python wc2026_predictor.py
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import poisson

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
DATA_DIR = Path(__file__).parent / "data"
OUT_DIR  = Path(__file__).parent / "output"
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

WC_MATCHES_CSV  = DATA_DIR / "wc_matches.csv"
WC_MATCHES_URL  = (
    "https://raw.githubusercontent.com/"
    "jfjelstul/worldcup/master/data-csv/matches.csv"
)
INTL_RESULTS_CSV  = DATA_DIR / "results.csv"        # optional: Kaggle martj42 dataset
PLAYERS_CSV       = DATA_DIR / "players" / "players.csv"  # optional: Kaggle player-scores
CUP26_CSV         = DATA_DIR / "cup26_predictions.csv"    # scraped from cup26matches.com

# ─── Name normalisation (WC2026 repo name → historical dataset name) ──────────
_NAME_MAP = {
    "Democratic Republic of the Congo": "DR Congo",
    "Bosnia and Herzegovina":           "Bosnia and Herzegovina",
    "South Korea":                      "South Korea",
    "United States":                    "United States",
    "Ivory Coast":                      "Ivory Coast",
    "Czech Republic":                   "Czech Republic",
    "Cape Verde":                       "Cape Verde",
    "Saudi Arabia":                     "Saudi Arabia",
    "New Zealand":                      "New Zealand",
    "Curaçao":                          "Curaçao",
}

def norm(name: str) -> str:
    return _NAME_MAP.get(name, name)

# ─── FIFA Rankings — scraped from Transfermarkt on 2026-06-11 ─────────────────
# Source: transfermarkt.es/statistik/weltrangliste (June 11, 2026)
FIFA_RANK: dict[str, int] = {
    "FRA": 1,   # France       (1877 pts)
    "ESP": 2,   # Spain        (1876 pts)
    "ARG": 3,   # Argentina    (1875 pts)
    "ENG": 4,   # England      (1826 pts)
    "POR": 5,   # Portugal     (1764 pts)
    "BRA": 6,   # Brazil       (1761 pts)
    "NED": 7,   # Netherlands  (1758 pts)
    "MAR": 8,   # Morocco      (1756 pts)
    "BEL": 9,   # Belgium      (1735 pts)
    "GER": 10,  # Germany      (1730 pts)
    "CRO": 11,  # Croatia      (1717 pts)
    "COL": 13,  # Colombia     (1693 pts)
    "SEN": 14,  # Senegal      (1689 pts)
    "MEX": 15,  # Mexico       (1681 pts)
    "USA": 16,  # United States(1673 pts)
    "URU": 17,  # Uruguay      (1673 pts)
    "JPN": 18,  # Japan        (1660 pts)
    "SUI": 19,  # Switzerland  (1649 pts)
    "IRN": 21,  # Iran         (1615 pts)
    "TUR": 22,  # Turkey       (1599 pts)
    "ECU": 23,  # Ecuador      (1595 pts)
    "AUT": 24,  # Austria      (1593 pts)
    "KOR": 25,  # South Korea  (1589 pts)
    "AUS": 27,  # Australia    (1581 pts)
    "ALG": 28,  # Algeria      (1564 pts)
    "EGY": 29,  # Egypt        (1563 pts)
    "CAN": 30,  # Canada       (1556 pts)
    "NOR": 31,  # Norway       (1551 pts)
    "PAN": 33,  # Panama       (1541 pts)
    "CIV": 34,  # Ivory Coast  (1533 pts)
    "SWE": 38,  # Sweden       (1515 pts)
    "PAR": 40,  # Paraguay     (1504 pts)
    "CZE": 41,  # Czech Rep.   (1501 pts)
    "SCO": 43,  # Scotland     (1498 pts)
    "TUN": 44,  # Tunisia      (1483 pts)
    "COD": 46,  # DR Congo     (1478 pts)
    "QAT": 55,  # Qatar        (1455 pts)
    "UZB": 50,  # Uzbekistan   (1465 pts)
    "IRQ": 57,  # Iraq         (1447 pts)
    "RSA": 60,  # South Africa (1430 pts)
    "KSA": 61,  # Saudi Arabia (1421 pts)
    "JOR": 63,  # Jordan       (1391 pts)
    "BIH": 65,  # Bosnia-Herz. (1386 pts)
    "CPV": 69,  # Cape Verde   (1366 pts)
    "GHA": 74,  # Ghana        (1346 pts)
    "CUW": 82,  # Curaçao      (1295 pts)
    "HAI": 83,  # Haiti        (1292 pts)
    "NZL": 85,  # New Zealand  (1282 pts)
}

# FIFA points per team — used to build a more granular Elo anchor
FIFA_PTS: dict[str, int] = {
    "FRA": 1877, "ESP": 1876, "ARG": 1875, "ENG": 1826, "POR": 1764,
    "BRA": 1761, "NED": 1758, "MAR": 1756, "BEL": 1735, "GER": 1730,
    "CRO": 1717, "COL": 1693, "SEN": 1689, "MEX": 1681, "USA": 1673,
    "URU": 1673, "JPN": 1660, "SUI": 1649, "IRN": 1615, "TUR": 1599,
    "ECU": 1595, "AUT": 1593, "KOR": 1589, "AUS": 1581, "ALG": 1564,
    "EGY": 1563, "CAN": 1556, "NOR": 1551, "PAN": 1541, "CIV": 1533,
    "SWE": 1515, "PAR": 1504, "CZE": 1501, "SCO": 1498, "TUN": 1483,
    "COD": 1478, "UZB": 1465, "QAT": 1455, "IRQ": 1447, "RSA": 1430,
    "KSA": 1421, "JOR": 1391, "BIH": 1386, "CPV": 1366, "GHA": 1346,
    "CUW": 1295, "HAI": 1292, "NZL": 1282,
}

REIGNING_CHAMP = "Argentina"  # 2022 World Cup winner

# ─── Data loading ─────────────────────────────────────────────────────────────
def load_historical() -> pd.DataFrame:
    """
    Load match data. Priority:
    1. Kaggle full dataset  → predictor/data/results.csv
       Download from: kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
       Place the file at predictor/data/results.csv
    2. WC-only dataset (auto-downloaded) → predictor/data/wc_matches.csv
    """
    if INTL_RESULTS_CSV.exists():
        print(f"Using Kaggle dataset: {INTL_RESULTS_CSV.name}")
        df = pd.read_csv(INTL_RESULTS_CSV, parse_dates=["date"])
        df = df.rename(columns={"home_team": "home_team", "away_team": "away_team"})
        df["home_team"] = df["home_team"].apply(norm)
        df["away_team"] = df["away_team"].apply(norm)
        df = df.dropna(subset=["home_score", "away_score"])
        df["home_score"] = df["home_score"].astype(int)
        df["away_score"] = df["away_score"].astype(int)
        if "tournament" not in df.columns:
            df["tournament"] = ""
        print(f"Loaded {len(df):,} international matches")
        return df

    # Fallback: WC-only data
    if not WC_MATCHES_CSV.exists():
        print("Downloading WC historical match data (1930-2022)...")
        r = requests.get(WC_MATCHES_URL, timeout=90)
        r.raise_for_status()
        WC_MATCHES_CSV.write_bytes(r.content)
    else:
        print(f"Using cached WC data: {WC_MATCHES_CSV.name}")

    raw = pd.read_csv(WC_MATCHES_CSV, parse_dates=["match_date"])
    if "tournament_name" not in raw.columns:
        raise ValueError("WC dataset must include tournament_name to filter men's matches")

    # The upstream file contains both men's and women's World Cups. Team names
    # alone are not enough to distinguish them (for example, United States), so
    # mixing both competitions severely inflates several ratings.
    mens_mask = raw["tournament_name"].str.contains(
        r"\bMen's World Cup\b", case=False, na=False, regex=True
    )
    excluded = int((~mens_mask).sum())
    raw = raw[mens_mask].copy()
    if raw.empty:
        raise ValueError("WC dataset contains no men's World Cup matches")

    df = pd.DataFrame({
        "date":       raw["match_date"],
        "home_team":  raw["home_team_name"].apply(norm),
        "away_team":  raw["away_team_name"].apply(norm),
        "home_score": raw["home_team_score"],
        "away_score": raw["away_team_score"],
        "tournament": "FIFA World Cup",
    })
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    print(f"Loaded {len(df):,} men's WC matches (1930–2022); excluded {excluded:,} women's matches")
    return df

# ─── cup26matches.com predictions ────────────────────────────────────────────
# Team name variants used on cup26matches.com → repo display names
_CUP26_TEAM_MAP: dict[str, str] = {
    "Usa":                      "United States",
    "Dr Congo":                 "Democratic Republic of the Congo",
    "Curacao":                  "Curaçao",
    "Bosnia And Herzegovina":   "Bosnia and Herzegovina",
}

def load_cup26_predictions() -> dict[tuple[str, str], dict]:
    """
    Load cup26matches.com match predictions from CSV.
    Returns dict keyed by (home_repo_name, away_repo_name) →
        {lh: float, la: float, win_pct: float | None}
    Only group-stage matches (named teams) are included.
    """
    if not CUP26_CSV.exists():
        print("  cup26 predictions: file not found — skipping.")
        return {}

    df = pd.read_csv(CUP26_CSV)
    result: dict[tuple[str, str], dict] = {}

    for _, row in df.iterrows():
        home_raw = str(row.get("home", ""))
        away_raw = str(row.get("away", ""))
        score    = str(row.get("score", ""))
        win_pct  = row.get("win_pct")

        # Skip knockout placeholder rows
        if not home_raw or not away_raw or home_raw.startswith(("W", "L")):
            continue

        # Map to repo display names
        home = _CUP26_TEAM_MAP.get(home_raw, home_raw)
        away = _CUP26_TEAM_MAP.get(away_raw, away_raw)

        # Parse predicted score → implied goal lambdas
        m = re.match(r"(\d+)-(\d+)", score)
        if not m:
            continue
        lh = float(m.group(1))
        la = float(m.group(2))

        # Guard against degenerate 0-0 predictions
        lh = max(lh, 0.3)
        la = max(la, 0.3)

        result[(home, away)] = {
            "lh": lh,
            "la": la,
            "win_pct": float(win_pct) if pd.notna(win_pct) else None,
        }

    print(f"  cup26 predictions loaded: {len(result)} group matches")
    return result

# ─── Squad quality (player-scores dataset) ───────────────────────────────────
# Country name variants used in Transfermarkt → FIFA code
_PLAYER_COUNTRY_MAP: dict[str, str] = {
    "France": "FRA", "Spain": "ESP", "Argentina": "ARG", "England": "ENG",
    "Portugal": "POR", "Brazil": "BRA", "Netherlands": "NED", "Morocco": "MAR",
    "Belgium": "BEL", "Germany": "GER", "Croatia": "CRO", "Colombia": "COL",
    "Senegal": "SEN", "Mexico": "MEX", "United States": "USA", "Uruguay": "URU",
    "Japan": "JPN", "Switzerland": "SUI", "Iran": "IRN", "Turkey": "TUR",
    "Ecuador": "ECU", "Austria": "AUT", "South Korea": "KOR", "Korea, South": "KOR",
    "Australia": "AUS", "Algeria": "ALG", "Egypt": "EGY", "Canada": "CAN",
    "Norway": "NOR", "Panama": "PAN", "Ivory Coast": "CIV", "Côte d'Ivoire": "CIV",
    "Sweden": "SWE", "Paraguay": "PAR", "Czech Republic": "CZE", "Czechia": "CZE",
    "Scotland": "SCO", "Tunisia": "TUN", "Uzbekistan": "UZB", "Qatar": "QAT",
    "Iraq": "IRQ", "South Africa": "RSA", "Saudi Arabia": "KSA", "Jordan": "JOR",
    "Bosnia and Herzegovina": "BIH", "Cape Verde": "CPV", "Ghana": "GHA",
    "Curaçao": "CUW", "Curacao": "CUW", "Haiti": "HAI", "New Zealand": "NZL",
    "Democratic Republic of the Congo": "COD", "Congo DR": "COD", "DR Congo": "COD",
}

def load_squad_quality() -> dict[str, float]:
    """
    Calculate squad quality score per WC2026 team from Transfermarkt player valuations.

    Requires: predictor/data/players/players.csv  (from Kaggle davidcariboo/player-scores)
    Download: kaggle.com/datasets/davidcariboo/player-scores
              → unzip and place players.csv in predictor/data/players/

    Returns: dict FIFA code → quality multiplier (1.0 = average WC team).
             Empty dict if file is not present (model runs without this feature).
    """
    if not PLAYERS_CSV.exists():
        print("  Squad quality: players.csv not found — skipping player value feature.")
        print(f"  To enable: place Kaggle player-scores/players.csv at {PLAYERS_CSV}")
        return {}

    print(f"Loading player valuations: {PLAYERS_CSV} ...")
    df = pd.read_csv(PLAYERS_CSV, low_memory=False)

    # Detect nationality and market-value columns
    nat_col = next((c for c in df.columns if "citizen" in c.lower()), None) \
           or next((c for c in df.columns if "nation" in c.lower()), None)
    val_col = next(
        (c for c in df.columns if "market_value" in c.lower() and "highest" not in c.lower()),
        None
    )

    if nat_col is None or val_col is None:
        print(f"  Unexpected columns: {list(df.columns)[:10]} — skipping.")
        return {}

    df["_code"] = df[nat_col].map(_PLAYER_COUNTRY_MAP)
    df = df[df["_code"].notna()].copy()
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df[df[val_col] > 0]

    # Sum of top-23 market values per national team (approximate squad selection)
    top23_value = (
        df.sort_values(val_col, ascending=False)
          .groupby("_code")
          .head(23)
          .groupby("_code")[val_col]
          .sum()
    )

    wc_codes = set(FIFA_RANK.keys())
    wc_vals  = {k: v for k, v in top23_value.items() if k in wc_codes}

    if not wc_vals:
        print("  No WC teams found in player data — skipping.")
        return {}

    avg = sum(wc_vals.values()) / len(wc_vals)
    quality = {k: v / avg for k, v in wc_vals.items()}

    print(f"  Squad quality loaded for {len(quality)}/{len(wc_codes)} WC teams")
    top5 = sorted(quality.items(), key=lambda x: -x[1])[:5]
    print("  Top 5 by squad value:", [(k, f"{v:.2f}x") for k, v in top5])
    return quality

# ─── Elo model ────────────────────────────────────────────────────────────────
class Elo:
    BASE = 1500.0

    def __init__(self):
        self.r: dict[str, float] = defaultdict(lambda: self.BASE)

    @staticmethod
    def k_factor(tournament: str) -> float:
        t = tournament.lower()
        if "world cup" in t:
            return 60
        if any(x in t for x in ("copa", "euro", "africa cup", "asian cup", "nations league")):
            return 40
        if "friendly" in t:
            return 20
        return 30

    def update(self, home: str, away: str, hg: int, ag: int, tournament: str = "") -> None:
        ra, rb = self.r[home], self.r[away]
        ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        sa = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        k  = self.k_factor(tournament)
        self.r[home] = ra + k * (sa - ea)
        self.r[away] = rb + k * ((1.0 - sa) - (1.0 - ea))

    def train(self, df: pd.DataFrame) -> None:
        print("Training Elo model...")
        df = df.sort_values("date")
        for row in df.itertuples(index=False):
            self.update(row.home_team, row.away_team,
                        int(row.home_score), int(row.away_score),
                        getattr(row, "tournament", ""))
        top = sorted(self.r.items(), key=lambda x: -x[1])[:5]
        print("  Top 5 by Elo:", [(t, round(v)) for t, v in top])

    def get(self, name: str, fifa_code: str | None = None) -> float:
        if name in self.r:
            return self.r[name]
        rank = FIFA_RANK.get(fifa_code or "", 80)
        return self.BASE - (rank - 1) * 7.0

# ─── Poisson model ────────────────────────────────────────────────────────────
AVG_GOALS = 1.15  # avg goals per team per match, international football

class PoissonModel:
    HOME_ADV = 0.0  # neutral venues for WC 2026
    PRIOR_MATCHES = 8.0

    def __init__(self, elo: Elo):
        self.elo   = elo
        self.att:           dict[str, float] = {}
        self.dfn:           dict[str, float] = {}
        self.sample_size:    dict[str, int] = {}
        self.squad_quality: dict[str, float] = {}  # FIFA code → multiplier

    def train(self, df: pd.DataFrame, teams: list[str], years: int = 10) -> None:
        if df.empty:
            raise ValueError("Cannot train Poisson model with an empty dataset")

        # Anchor the window to the latest observation, not the wall clock. This
        # keeps a cached historical dataset stable when the script is rerun.
        cutoff = df["date"].max() - pd.DateOffset(years=years)
        recent = df[df["date"] >= cutoff]
        for team in teams:
            hm = recent[recent["home_team"] == team]
            am = recent[recent["away_team"] == team]
            scored   = list(hm["home_score"]) + list(am["away_score"])
            conceded = list(hm["away_score"]) + list(am["home_score"])
            n = len(scored)
            self.sample_size[team] = n

            # Bayesian shrinkage prevents a handful of tournament matches from
            # producing extreme attack/defence rates. Eight prior matches at
            # the international average still let sustained form stand out.
            denom = n + self.PRIOR_MATCHES
            prior_goals = self.PRIOR_MATCHES * AVG_GOALS
            self.att[team] = (sum(scored) + prior_goals) / denom
            self.dfn[team] = (sum(conceded) + prior_goals) / denom

    def lambdas(self, home: str, away: str, elo_h: float, elo_a: float,
                code_h: str = "", code_a: str = "") -> tuple[float, float]:
        att_h = self.att.get(home, AVG_GOALS)
        dfn_h = self.dfn.get(home, AVG_GOALS)
        att_a = self.att.get(away, AVG_GOALS)
        dfn_a = self.dfn.get(away, AVG_GOALS)

        elo_factor = 10.0 ** ((elo_h - elo_a) / 400.0 * 0.25)

        # Squad quality: gentle multiplier based on market value ratio.
        # Exponent 0.15 keeps it as a tiebreaker — Elo/FIFA pts carry the main weight.
        sq = self.squad_quality
        sq_factor = ((sq.get(code_h, 1.0) / sq.get(code_a, 1.0)) ** 0.15) if sq else 1.0

        lh = att_h * (dfn_a / AVG_GOALS) * elo_factor * sq_factor
        la = att_a * (dfn_h / AVG_GOALS) / elo_factor / sq_factor
        return float(np.clip(lh, 0.2, 4.5)), float(np.clip(la, 0.2, 4.5))

    def win_probs(self, lh: float, la: float, max_g: int = 8) -> tuple[float, float, float]:
        g   = np.arange(max_g + 1)
        mat = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
        return float(np.tril(mat, -1).sum()), float(np.trace(mat)), float(np.triu(mat, 1).sum())

    def most_likely_score(self, lh: float, la: float, max_g: int = 8) -> tuple[int, int]:
        g   = np.arange(max_g + 1)
        mat = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
        i, j = np.unravel_index(np.argmax(mat), mat.shape)
        return int(i), int(j)

    def sample_score(self, lh: float, la: float) -> tuple[int, int]:
        return int(np.random.poisson(lh)), int(np.random.poisson(la))

# ─── Team registry ────────────────────────────────────────────────────────────
class TeamRegistry:
    def __init__(self, teams_data: list, elo: Elo, model: PoissonModel):
        self.by_id   = {t["id"]: t for t in teams_data}
        self.by_name = {t["name_en"]: t for t in teams_data}
        self.elo     = elo
        self.model   = model

    def name(self, team_id: str) -> str:
        t = self.by_id.get(team_id)
        return t["name_en"] if t else f"Team#{team_id}"

    def elo_for(self, display_name: str) -> float:
        t = self.by_name.get(display_name)
        code = t["fifa_code"] if t else None
        return self.elo.get(norm(display_name), code)

    def code(self, display_name: str) -> str:
        t = self.by_name.get(display_name)
        return t["fifa_code"] if t else ""

    def lambdas(self, home: str, away: str) -> tuple[float, float]:
        return self.model.lambdas(
            norm(home), norm(away),
            self.elo_for(home), self.elo_for(away),
            self.code(home), self.code(away),
        )

# ─── Group stage ──────────────────────────────────────────────────────────────
_CUP26_BLEND = 0.20  # external score forecasts are a secondary signal

def simulate_groups(matches: list, reg: TeamRegistry, random_mode: bool = True,
                    cup26: dict | None = None):
    standings: dict = defaultdict(lambda: defaultdict(
        lambda: dict(pts=0, gf=0, ga=0, gd=0, w=0, d=0, l=0)
    ))
    results: dict = {}

    for m in matches:
        if m["type"] != "group":
            continue
        home = reg.name(m["home_team_id"])
        away = reg.name(m["away_team_id"])
        grp  = m["group"]
        lh, la = reg.lambdas(home, away)

        # Blend with cup26matches.com prediction when available
        if cup26:
            c26 = cup26.get((home, away))
            if c26:
                lh = (1 - _CUP26_BLEND) * lh + _CUP26_BLEND * c26["lh"]
                la = (1 - _CUP26_BLEND) * la + _CUP26_BLEND * c26["la"]

        hg, ag = reg.model.sample_score(lh, la) if random_mode else reg.model.most_likely_score(lh, la)

        pw, pd_, pa = reg.model.win_probs(lh, la)
        results[m["id"]] = dict(
            home=home, away=away, hg=hg, ag=ag,
            group=grp, matchday=m["matchday"], date=m["local_date"],
            p_hw=pw, p_dr=pd_, p_aw=pa
        )

        hs = standings[grp][home]
        as_ = standings[grp][away]
        hs["gf"] += hg;  hs["ga"] += ag;  hs["gd"] += hg - ag
        as_["gf"] += ag; as_["ga"] += hg; as_["gd"] += ag - hg
        if hg > ag:
            hs["pts"]  += 3; hs["w"] += 1; as_["l"] += 1
        elif hg == ag:
            hs["pts"]  += 1; as_["pts"] += 1; hs["d"] += 1; as_["d"] += 1
        else:
            as_["pts"] += 3; as_["w"] += 1; hs["l"] += 1

    return standings, results

def get_qualified(standings: dict):
    qualified: dict[str, str] = {}
    thirds: list = []
    for grp, teams in standings.items():
        ranked = sorted(teams.items(),
                        key=lambda x: (x[1]["pts"], x[1]["gd"], x[1]["gf"]),
                        reverse=True)
        for pos, (name, stats) in enumerate(ranked, 1):
            qualified[f"{grp}{pos}"] = name
        if len(ranked) >= 3:
            nm, st = ranked[2]
            thirds.append(dict(team=nm, group=grp, **st))
    thirds.sort(key=lambda x: (x["pts"], x["gd"], x["gf"]), reverse=True)
    return qualified, thirds

# ─── R32 bracket resolver ─────────────────────────────────────────────────────
def resolve_r32(ko_matches: list, qualified: dict, thirds: list) -> dict:
    """Map each R32 match label to actual team names."""
    taken: set[str] = set()

    def pick_3rd(pool_str: str) -> str:
        allowed = set(pool_str.split("/"))
        for t in thirds:
            if t["group"] in allowed and t["team"] not in taken:
                taken.add(t["team"])
                return t["team"]
        for t in thirds:
            if t["team"] not in taken:
                taken.add(t["team"])
                return t["team"]
        return "TBD"

    def resolve(label: str) -> str:
        label = label.strip()
        if label.startswith("Winner Group "):
            g = label.split()[-1]
            return qualified.get(f"{g}1", "TBD")
        if label.startswith("Runner-up Group "):
            g = label.split()[-1]
            return qualified.get(f"{g}2", "TBD")
        if label.startswith("3rd Group "):
            pool = label.replace("3rd Group ", "")
            return pick_3rd(pool)
        return "TBD"

    bracket: dict = {}
    r32 = [m for m in ko_matches if m["type"] == "r32"]
    for m in r32:
        bracket[m["id"]] = dict(
            home=resolve(m.get("home_team_label", "")),
            away=resolve(m.get("away_team_label", "")),
            date=m["local_date"],
            home_label=m.get("home_team_label", ""),
            away_label=m.get("away_team_label", ""),
        )
    return bracket

# ─── Knockout simulation ──────────────────────────────────────────────────────
def ko_match(home: str, away: str, reg: TeamRegistry, random_mode: bool):
    lh, la = reg.lambdas(home, away)
    hg, ag = reg.model.sample_score(lh, la) if random_mode else reg.model.most_likely_score(lh, la)
    penalties = False
    if hg == ag:
        penalties = True
        eh, ea = reg.elo_for(home), reg.elo_for(away)
        p_home = 1.0 / (1.0 + 10.0 ** ((ea - eh) / 400.0))
        winner = home if (random.random() < p_home if random_mode else eh >= ea) else away
    else:
        winner = home if hg > ag else away
    return winner, hg, ag, penalties

def simulate_knockout(ko_matches: list, r32_bracket: dict, reg: TeamRegistry, random_mode: bool):
    ko_by_id = {m["id"]: m for m in ko_matches}
    results: dict[str, dict] = {}
    winner_of: dict[str, str] = {}
    loser_of:  dict[str, str] = {}

    # R32
    for mid, info in sorted(r32_bracket.items()):
        home, away = info["home"], info["away"]
        if "TBD" in (home, away):
            winner = home if away == "TBD" else away
            results[mid] = dict(home=home, away=away, hg=0, ag=0, winner=winner, penalties=False, round="r32")
        else:
            w, hg, ag, pen = ko_match(home, away, reg, random_mode)
            results[mid] = dict(home=home, away=away, hg=hg, ag=ag, winner=w, penalties=pen, round="r32")
        winner_of[mid] = results[mid]["winner"]
        loser_of[mid]  = results[mid]["away"] if results[mid]["winner"] == results[mid]["home"] else results[mid]["home"]

    # R16 → QF → SF → third → final
    for rnd in ("r16", "qf", "sf", "third", "final"):
        for m in sorted([x for x in ko_matches if x["type"] == rnd], key=lambda x: x["id"]):
            mid = m["id"]

            def resolve_label(label: str) -> str:
                wm = re.match(r"Winner Match (\d+)", label)
                if wm:
                    return winner_of.get(wm.group(1), "TBD")
                lm = re.match(r"Loser Match (\d+)", label)
                if lm:
                    return loser_of.get(lm.group(1), "TBD")
                return "TBD"

            home = resolve_label(m.get("home_team_label", ""))
            away = resolve_label(m.get("away_team_label", ""))

            if "TBD" in (home, away):
                winner = home if away == "TBD" else away
                results[mid] = dict(home=home, away=away, hg=0, ag=0, winner=winner, penalties=False, round=rnd)
            else:
                w, hg, ag, pen = ko_match(home, away, reg, random_mode)
                results[mid] = dict(home=home, away=away, hg=hg, ag=ag, winner=w, penalties=pen, round=rnd)

            winner_of[mid] = results[mid]["winner"]
            loser_of[mid]  = results[mid]["away"] if results[mid]["winner"] == results[mid]["home"] else results[mid]["home"]

    return results

# ─── Monte Carlo ──────────────────────────────────────────────────────────────
def monte_carlo(matches: list, ko_matches: list, reg: TeamRegistry, n: int = 20000,
                cup26: dict | None = None) -> dict:
    print(f"\nRunning {n:,} Monte Carlo simulations...")
    ko_type = {m["id"]: m["type"] for m in ko_matches}
    champ   = defaultdict(int)
    r_final = defaultdict(int)
    r_semi  = defaultdict(int)
    r_qf    = defaultdict(int)

    for i in range(n):
        if i % 5000 == 0 and i > 0:
            print(f"  {i:>6}/{n}...")
        standings, _ = simulate_groups(matches, reg, random_mode=True, cup26=cup26)
        qualified, thirds = get_qualified(standings)
        r32_bracket      = resolve_r32(ko_matches, qualified, thirds)
        ko_results       = simulate_knockout(ko_matches, r32_bracket, reg, random_mode=True)

        for mid, r in ko_results.items():
            rnd = ko_type.get(mid, r.get("round", ""))
            if rnd == "final":
                champ[r["winner"]] += 1
                r_final[r["home"]] += 1
                r_final[r["away"]] += 1
            elif rnd == "sf":
                r_semi[r["home"]] += 1
                r_semi[r["away"]] += 1
            elif rnd == "qf":
                r_qf[r["home"]] += 1
                r_qf[r["away"]] += 1

    def to_pct(d):
        return {k: round(v / n * 100, 1) for k, v in sorted(d.items(), key=lambda x: -x[1])}

    return {
        "champion":    to_pct(champ),
        "reach_final": to_pct(r_final),
        "reach_semi":  to_pct(r_semi),
        "reach_qf":    to_pct(r_qf),
    }

# ─── Output helpers ───────────────────────────────────────────────────────────
def _bar(pct: float, width: int = 30) -> str:
    filled = int(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)

def print_group_stage(group_results: dict) -> None:
    print("\n" + "=" * 72)
    print("  GROUP STAGE — predicted results")
    print("=" * 72)
    for grp in sorted(set(v["group"] for v in group_results.values())):
        print(f"\n  ── Group {grp} " + "─" * 55)
        gms = {k: v for k, v in group_results.items() if v["group"] == grp}
        for _, m in sorted(gms.items(), key=lambda x: (x[1]["matchday"], x[0])):
            pw, pd_, pa = m["p_hw"] * 100, m["p_dr"] * 100, m["p_aw"] * 100
            print(f"  MD{m['matchday']}  {m['home']:>26} {m['hg']}-{m['ag']} {m['away']:<26}"
                  f"  {pw:.0f}/{pd_:.0f}/{pa:.0f}%")

def print_knockout(ko_results: dict) -> None:
    labels = {"r32": "ROUND OF 32", "r16": "ROUND OF 16",
              "qf": "QUARTER-FINALS", "sf": "SEMI-FINALS",
              "final": "FINAL", "third": "THIRD PLACE"}
    print()
    for rnd in ("r32", "r16", "qf", "sf", "final", "third"):
        rnd_matches = {k: v for k, v in ko_results.items() if v.get("round") == rnd}
        if not rnd_matches:
            continue
        print("\n" + "=" * 72)
        print(f"  {labels.get(rnd, rnd.upper())}")
        print("=" * 72)
        for _, r in sorted(rnd_matches.items()):
            pen_str = " (pens)" if r.get("penalties") else ""
            print(f"  {r['home']:>26} {r['hg']}-{r['ag']} {r['away']:<26}  → {r['winner']}{pen_str}")

def print_mc(mc: dict) -> None:
    print("\n" + "=" * 72)
    print("  CHAMPION PROBABILITIES  (Monte Carlo)")
    print("=" * 72)
    for team, pct in list(mc["champion"].items())[:16]:
        print(f"  {team:<28}  {pct:5.1f}%  {_bar(pct)}")

    print("\n  ── Top 10: reach Final ──────────────────────────────")
    for team, pct in list(mc["reach_final"].items())[:10]:
        print(f"  {team:<28}  {pct:5.1f}%")

    print("\n  ── Top 10: reach Semis ──────────────────────────────")
    for team, pct in list(mc["reach_semi"].items())[:10]:
        print(f"  {team:<28}  {pct:5.1f}%")

def save_outputs(group_results: dict, ko_results: dict, mc: dict) -> None:
    # Group stage CSV
    rows = []
    for _, m in group_results.items():
        rows.append({
            "Group": m["group"], "Matchday": m["matchday"], "Date": m["date"],
            "Home": m["home"], "Score": f"{m['hg']}-{m['ag']}", "Away": m["away"],
            "P(Home Win)": f"{m['p_hw']*100:.0f}%",
            "P(Draw)":     f"{m['p_dr']*100:.0f}%",
            "P(Away Win)": f"{m['p_aw']*100:.0f}%",
        })
    gdf = pd.DataFrame(rows).sort_values(["Group", "Matchday"])
    gdf.to_csv(OUT_DIR / "group_stage.csv", index=False)

    # Knockout CSV
    ko_rows = []
    for mid, r in ko_results.items():
        ko_rows.append({
            "Round": r.get("round", "").upper(), "Match": mid,
            "Home": r["home"], "Score": f"{r['hg']}-{r['ag']}", "Away": r["away"],
            "Winner": r["winner"], "Penalties": r.get("penalties", False),
        })
    order = {"R32": 0, "R16": 1, "QF": 2, "SF": 3, "FINAL": 4, "THIRD": 5}
    ko_df = pd.DataFrame(ko_rows)
    ko_df["_ord"] = ko_df["Round"].map(order)
    ko_df.sort_values(["_ord", "Match"]).drop(columns="_ord").to_csv(OUT_DIR / "knockout.csv", index=False)

    # Probabilities JSON
    with open(OUT_DIR / "probabilities.json", "w") as f:
        json.dump(mc, f, indent=2, ensure_ascii=False)

    print(f"\nFiles saved to {OUT_DIR}/")
    print("  group_stage.csv  |  knockout.csv  |  probabilities.json")

# ─── Main ─────────────────────────────────────────────────────────────────────
def main(simulations: int = 50000, seed: int = 2026, use_cup26: bool = True) -> None:
    if simulations <= 0:
        raise ValueError("simulations must be greater than zero")

    random.seed(seed)
    np.random.seed(seed)

    # Load repo data
    teams   = json.loads((ROOT / "football.teams.json").read_text())
    matches = json.loads((ROOT / "football.matches.json").read_text())
    ko_matches = [m for m in matches if m["type"] != "group"]

    # Historical data + models
    df = load_historical()

    # Train Elo only on recent WCs (2010–2022) so old performances (USA 1930s,
    # etc.) don't inflate ratings. Teams absent from recent WCs start at BASE.
    df_recent = df[df["date"] >= "2010-01-01"]
    elo = Elo()
    elo.train(df_recent)

    # Blend recent-WC Elo (20%) with real FIFA points (80%).
    # FIFA points are used directly as the anchor — no rank conversion needed.
    # This ensures current form dominates over WC history.
    for t in teams:
        name   = norm(t["name_en"])
        anchor = float(FIFA_PTS.get(t["fifa_code"], 1300))
        elo.r[name] = elo.r[name] * 0.20 + anchor * 0.80

    # Reigning champion bonus (+25 pts, ~equivalent to half a good WC game)
    elo.r[norm(REIGNING_CHAMP)] += 25

    wc_norm_names = [norm(t["name_en"]) for t in teams]
    model = PoissonModel(elo)
    model.train(df, wc_norm_names, years=12)  # use WC 2014–2022 (3 recent tournaments)
    model.squad_quality = load_squad_quality()

    reg  = TeamRegistry(teams, elo, model)
    cup26 = load_cup26_predictions() if use_cup26 else {}

    # Elo ranking of WC teams
    print("\n  ── Elo ratings for WC 2026 teams ─────────────────────")
    elo_ranked = sorted(
        [(t["name_en"], round(reg.elo_for(t["name_en"]))) for t in teams],
        key=lambda x: -x[1]
    )
    for i, (name, rating) in enumerate(elo_ranked, 1):
        print(f"  {i:>2}. {name:<30} {rating}")

    # Deterministic prediction
    print("\nGenerating deterministic predictions...")
    standings, group_results = simulate_groups(matches, reg, random_mode=False, cup26=cup26)
    qualified, thirds = get_qualified(standings)
    r32_bracket = resolve_r32(ko_matches, qualified, thirds)
    ko_results  = simulate_knockout(ko_matches, r32_bracket, reg, random_mode=False)

    print_group_stage(group_results)
    print_knockout(ko_results)

    # Monte Carlo
    mc = monte_carlo(matches, ko_matches, reg, n=simulations, cup26=cup26)
    print_mc(mc)

    save_outputs(group_results, ko_results, mc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the 2026 FIFA Men's World Cup")
    parser.add_argument(
        "--simulations", type=int, default=50000,
        help="number of Monte Carlo tournament simulations (default: 50000)",
    )
    parser.add_argument(
        "--seed", type=int, default=2026,
        help="random seed for reproducible results (default: 2026)",
    )
    parser.add_argument(
        "--no-cup26", action="store_true",
        help="disable the external cup26matches.com score forecasts",
    )
    args = parser.parse_args()
    main(
        simulations=args.simulations,
        seed=args.seed,
        use_cup26=not args.no_cup26,
    )
