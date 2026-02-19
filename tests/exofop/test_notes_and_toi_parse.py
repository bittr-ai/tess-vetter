from __future__ import annotations

from bittr_tess_vetter.exofop.client import ExoFopClient


def test_parse_obsnotes_csv_with_quoted_text(tmp_path):
    client = ExoFopClient(cache_dir=tmp_path)
    text = (
        'TIC,Author,Date,Data Tag,Note\n'
        '365938305,alice,2024-01-02,imaging,"Detected companion, needs follow-up"\n'
    )

    rows = client._parse_obsnotes_text(text, fallback_tic_id=365938305)

    assert len(rows) == 1
    assert rows[0].tic_id == 365938305
    assert rows[0].author == "alice"
    assert rows[0].date_utc == "2024-01-02"
    assert rows[0].data_tag == "imaging"
    assert rows[0].text == "Detected companion, needs follow-up"


def test_parse_obsnotes_pipe_fallback(tmp_path):
    client = ExoFopClient(cache_dir=tmp_path)
    text = (
        "TIC ID|User|Obs Date|Tag|Comments\n"
        "365938305|bob|2024-02-10|spectrum|RV trend present\n"
    )

    rows = client._parse_obsnotes_text(text, fallback_tic_id=365938305)

    assert len(rows) == 1
    assert rows[0].tic_id == 365938305
    assert rows[0].author == "bob"
    assert rows[0].date_utc == "2024-02-10"
    assert rows[0].data_tag == "spectrum"
    assert rows[0].text == "RV trend present"


def test_extract_toi_row_best_effort(tmp_path):
    client = ExoFopClient(cache_dir=tmp_path)
    text = (
        "toi|tic_id|tfopwg_disp|disp|comments\n"
        "1262.01|365938305|PC|CP|known candidate\n"
        "1262.02|365938305|FP|FP|false positive\n"
    )

    row = client._extract_toi_row(text=text, tic_id=365938305, toi="1262.01")

    assert row is not None
    assert row.tic_id == 365938305
    assert row.toi == "1262.01"
    assert row.tfopwg_disposition == "PC"
    assert row.planet_disposition == "CP"
    assert row.comments == "known candidate"
