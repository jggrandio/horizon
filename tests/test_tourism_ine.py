import json
from pathlib import Path
from typing import Any

import pandas as pd

from horizon.domains.tourism import ine


def series(
    point: str, concept: str, data: list[tuple[int, int, float | None, str]]
) -> dict[str, Any]:
    return {
        "MetaData": [
            {"T3_Variable": "Concepto turístico", "Nombre": concept},
            {"T3_Variable": "PUNTOS TURISTÍCOS", "Nombre": point},
            {"T3_Variable": "RESIDENCIA/ORIGEN", "Nombre": "Residentes en España"},
        ],
        "Data": [
            {"Fecha": "ignored", "Anyo": y, "T3_Periodo": f"M{m:02d}", "Valor": v, "T3_TipoDato": t}
            for y, m, v, t in data
        ],
    }


def table() -> list[dict[str, Any]]:
    # Newest first, as the INE serves it. Altea omits 2024-02 and has a null in 2024-01 travellers.
    return [
        series(
            "Altea",
            "Pernoctaciones",
            [(2024, 3, 30.0, "Provisional"), (2024, 1, 10.0, "Definitivo")],
        ),
        series("Altea", "Viajero", [(2024, 3, 3.0, "Provisional"), (2024, 1, None, "Definitivo")]),
        series("Carboneras", "Pernoctaciones", [(2024, 2, 5.0, "Definitivo")]),
        series("Carboneras", "Viajero", [(2024, 2, 1.0, "Definitivo")]),
    ]


def test_tidy_pivots_sorts_and_fills_omitted_months() -> None:
    out = ine.tidy(table())
    altea = out[out["point"] == "Altea"]
    assert list(altea["period"]) == list(pd.date_range("2024-01-01", periods=3, freq="MS"))
    assert altea["nights"].tolist()[0::2] == [10.0, 30.0]
    assert altea["nights"].isna().tolist() == [False, True, False]
    assert altea["travellers"].isna().tolist() == [True, True, False]
    assert altea["provisional"].tolist() == [False, False, True]
    assert len(out) == 4


def test_load_builds_a_valid_frame(tmp_path: Path) -> None:
    dest = tmp_path / "2078.json"
    dest.write_text(json.dumps(table()), encoding="utf-8")  # cached, so no download
    frame = ine.load(dest)
    assert frame.target == "nights"
    assert frame.group_key == ("point", "residence")
