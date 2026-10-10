import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from horizon.domains.tourism import ine

# Real INE output (Mojácar, 2020-03 to 2020-07), cut from table 2078. In 2020-05 residents
# come as `Valor: null` and foreign residents omit the month: both forms of missing.
# Elaboración propia con datos extraídos del sitio web del INE: www.ine.es
SAMPLE = Path(__file__).parent / "fixtures" / "ine_2078_sample.json"


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


def test_unknown_concept_names_it() -> None:
    with pytest.raises(KeyError, match="Pernoctación"):
        ine.tidy([series("Altea", "Pernoctación", [(2024, 1, 1.0, "Definitivo")])])


def test_real_sample_normalises_both_forms_of_missing() -> None:
    out = ine.tidy(json.loads(SAMPLE.read_text(encoding="utf-8")))
    may = out[out["period"] == "2020-05-01"]
    assert may["residence"].tolist() == ["Residentes en España", "Residentes en el Extranjero"]
    assert may[["nights", "travellers"]].isna().all().all()
    assert out.groupby("residence").size().tolist() == [5, 5]


def test_load_builds_a_valid_frame() -> None:
    frame = ine.load(SAMPLE)  # exists, so no download
    assert frame.data.shape == (10, 6)
    assert frame.data["period"].min() == pd.Timestamp("2020-03-01")
    assert frame.data["nights"].notna().sum() == 6
