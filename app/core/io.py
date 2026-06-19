import json
from pathlib import Path
from typing import Any

import filelock
import pandas as pd


def write_csv_frame(
    filepath: str | Path,
    df: pd.DataFrame,
    append: bool = True,
    sep: str = ";",
) -> None:
    path = Path(filepath)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path,
        mode="a" if append else "w",
        header=not append or not path.exists(),
        sep=sep,
        index=False,
    )


def read_csv_frame(
    filepath: str | Path,
    sep: str = ";",
    **kwargs: Any,
) -> pd.DataFrame:
    return pd.read_csv(Path(filepath), sep=sep, **kwargs)


def save_raw_run_result(filepath: str | Path, record: dict[str, Any]) -> None:
    path = Path(filepath)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    lock = filelock.FileLock(str(path) + ".lock")
    with lock:
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")


def read_jsonl(filepath: str | Path) -> list[dict[str, Any]]:
    with Path(filepath).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def read_jsonl_case(filepath: str | Path, case_id: str) -> list[dict[str, Any]]:
    with Path(filepath).open(encoding="utf-8") as file:
        records = [
            record
            for line in file
            if line.strip()
            for record in [json.loads(line)]
            if str(record.get("config", {}).get("case_id", "")) == str(case_id)
        ]
    if not records:
        return records
    df = pd.json_normalize(records, sep="_")
    subset = [column for column in df.columns if column != "history"]
    if subset:
        df = df.drop_duplicates(subset=subset)
    return df.to_dict(orient="records")


def normalize_w_column(df: pd.DataFrame) -> pd.DataFrame:
    if "w" not in df.columns:
        return df

    df = df.copy()
    if "w_max" not in df.columns:
        df["w_max"] = 0.0
    if "w_min" not in df.columns:
        df["w_min"] = 0.0

    is_dict = df["w"].map(lambda value: isinstance(value, dict))
    if is_dict.any():
        w_values = pd.DataFrame(df.loc[is_dict, "w"].tolist(), index=df.index[is_dict]).fillna(0.0)
        if "w_max" in w_values.columns:
            df.loc[is_dict, "w_max"] = w_values["w_max"]
        if "w_min" in w_values.columns:
            df.loc[is_dict, "w_min"] = w_values["w_min"]
        df.loc[is_dict, "w"] = 0.0
    return df
