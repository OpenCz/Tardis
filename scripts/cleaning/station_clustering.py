import re
from collections import Counter

import pandas as pd

_CONFUSABLES = str.maketrans(
    {
        "1": "I",
        "3": "E",
        "4": "A",
        "5": "S",
        "8": "B",
        "@": "A",
        "'": "",
        "`": "",
        "-": "",
    }
)
_ABBREV = {"ST ": "SAINT "}
_INVALID = {"", "0", "NAN", "NONE", "NULL", "N/A", "NA"}
DEFAULT_THRESHOLD = 80


class StationClusterer:
    def __init__(self, threshold: int = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self.clusters: dict[str, list[str]] = {}

    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str], dict]:
        station_columns = self._detect_station_columns(df)
        print(f"Station-like columns detected: {station_columns}")

        all_values: list[str] = []
        for col in station_columns:
            all_values.extend(df[col].astype(str).tolist())

        cleaned_of_raw: dict[str, str] = {}
        valid_cleaned: list[str] = []
        for raw in all_values:
            cleaned = self._preprocess(raw)
            cleaned_of_raw[raw] = cleaned
            if cleaned not in _INVALID:
                valid_cleaned.append(cleaned)

        counts = Counter(valid_cleaned)
        unique = list(counts.keys())
        parent = {v: v for v in unique}

        for i, a in enumerate(unique):
            for b in unique[i + 1 :]:
                if self._tok_sort_ratio(a, b) < self.threshold:
                    continue
                ra = self._find_root(parent, a)
                rb = self._find_root(parent, b)
                if ra == rb:
                    continue
                if counts[ra] >= counts[rb]:
                    parent[rb] = ra
                else:
                    parent[ra] = rb

        root_to_members: dict[str, list[str]] = {}
        for v in unique:
            r = self._find_root(parent, v)
            root_to_members.setdefault(r, []).append(v)

        canonical_of: dict[str, str] = {}
        for members in root_to_members.values():
            canonical = max(members, key=lambda x: counts[x])
            for member in members:
                canonical_of[member] = canonical

        mapping = {
            raw: canonical_of.get(cleaned_of_raw.get(raw, ""), raw)
            for raw in all_values
        }

        for col in station_columns:
            df[col] = df[col].astype(str).map(mapping)

        self.clusters = {}
        for raw, canon in mapping.items():
            self.clusters.setdefault(canon, []).append(raw)

        stats = {
            "total": len(root_to_members),
            "merged": sum(1 for m in root_to_members.values() if len(m) > 1),
            "variants": sum(
                len(m) - 1 for m in root_to_members.values() if len(m) > 1
            ),
            "threshold": self.threshold,
            "columns": len(station_columns),
        }
        return df, station_columns, stats

    def print_merges(self) -> None:
        for canon, variants in sorted(
            self.clusters.items(), key=lambda x: str(x[0])
        ):
            if len(variants) > 1:
                print(f"\n{canon}")
                for v in sorted(variants, key=str):
                    prefix = "  " if v == canon else "↳ "
                    print(f"    {prefix}{v!r}")

    @staticmethod
    def _detect_station_columns(df: pd.DataFrame) -> list[str]:
        keywords = ("station", "departure", "arrival", "gare", "origine", "destination")
        station_columns = []
        for col in df.columns:
            if not any(kw in col.lower() for kw in keywords):
                continue
            if not pd.api.types.is_string_dtype(df[col]):
                continue
            sample = df[col].dropna().astype(str)
            if sample.empty:
                continue
            avg_len = sample.str.len().mean()
            digit_ratio = sample.str.count(r"\d").sum() / max(
                sample.str.len().sum(), 1
            )
            unique_ratio = df[col].nunique() / max(len(df[col]), 1)
            if (
                avg_len < 50
                and digit_ratio < 0.05
                and unique_ratio < 0.1
                and df[col].nunique() >= 5
            ):
                station_columns.append(col)
        return station_columns

    @staticmethod
    def _preprocess(raw: str) -> str:
        cleaned = str(raw).strip().upper().translate(_CONFUSABLES)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        for abbr, full in _ABBREV.items():
            cleaned = cleaned.replace(abbr, full)
        return cleaned

    @staticmethod
    def _tok_sort_ratio(a: str, b: str) -> int:
        a_s = " ".join(sorted(a.split()))
        b_s = " ".join(sorted(b.split()))
        t = len(a_s) + len(b_s)
        if t == 0:
            return 100
        common = sum((Counter(a_s) & Counter(b_s)).values())
        return int(200 * common / t)

    @staticmethod
    def _find_root(parent: dict, v: str) -> str:
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v
