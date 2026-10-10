"""INE table 2078 (hotel occupancy by tourist point): download and pivot onto the contract."""

import json
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from horizon.data import TimeSeriesFrame

BASE = "https://servicios.ine.es/wstempus/js/ES/DATOS_TABLA"
TABLE = 2078
CONCEPTS = {"Viajero": "travellers", "Pernoctaciones": "nights"}
KEYS = ["point", "residence", "period"]


def fetch_table(table_id: int, dest: Path, nult: int | None = None) -> Path:
    """Download one INE table as raw JSON, cached on disk."""
    if dest.exists():
        return dest

    params = {"tip": "AM"} | ({"nult": str(nult)} if nult else {})
    response = httpx.get(f"{BASE}/{table_id}", params=params, timeout=180.0, follow_redirects=True)
    response.raise_for_status()

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(response.json()), encoding="utf-8")
    return dest


def tidy(table: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per (point, residence, period), with `travellers` and `nights` as columns.

    Omitted months become null rows; `provisional` is true when either figure is provisional.
    """
    rows = []
    for series in table:
        meta = {m["T3_Variable"]: m["Nombre"] for m in series["MetaData"]}
        for obs in series["Data"]:
            rows.append(
                {
                    "point": meta["PUNTOS TURISTÍCOS"],  # misspelled upstream
                    "residence": meta["RESIDENCIA/ORIGEN"],
                    "concept": CONCEPTS[meta["Concepto turístico"]],
                    # Not `Fecha`: its timezone offset shifts months across DST.
                    "period": pd.Timestamp(
                        year=obs["Anyo"], month=int(obs["T3_Periodo"][1:]), day=1
                    ),
                    "value": obs["Valor"],
                    "provisional": obs["T3_TipoDato"] == "Provisional",
                }
            )
    long = pd.DataFrame(rows)

    wide = long.pivot(index=KEYS, columns="concept", values="value").astype("float64")
    wide.columns.name = None
    wide["provisional"] = long.groupby(KEYS)["provisional"].any()
    wide = (
        wide.reset_index("period")
        .groupby(level=["point", "residence"])
        .apply(lambda g: g.set_index("period").sort_index().asfreq("MS"))
        .reset_index()
    )
    wide["provisional"] = wide["provisional"].eq(True)
    return wide


def load(dest: Path = Path(f"data/ine/{TABLE}.json")) -> TimeSeriesFrame:
    """Table 2078 as a TimeSeriesFrame: monthly hotel nights per tourist point and residence."""
    table = json.loads(fetch_table(TABLE, dest).read_text(encoding="utf-8"))
    return TimeSeriesFrame(
        tidy(table),
        target="nights",
        time_index="period",
        freq="MS",
        group_key=("point", "residence"),
        past_only_covariates=("travellers",),
    )
