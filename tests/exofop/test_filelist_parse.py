from __future__ import annotations

from bittr_tess_vetter.exofop.client import ExoFopClient
from bittr_tess_vetter.exofop.types import ExoFopSelectors


def test_parse_filelist_csv_basic(tmp_path):
    client = ExoFopClient(cache_dir=tmp_path)
    csv_text = (
        "Type,File Name,File ID,TIC,TOI,Date,User,Group,Tag,Description\n"
        "Image,TOI1262_sensitivity.dat,562,365938305,1262.01,2020-08-04,howell,tfopwg,123,Speckle curve\n"
        "Spectrum,TIC0365938305S-foo.fits,451732,365938305,1262.01,2020-03-07,bar,tfopwg,999,TRES extracted spectra\n"
    )
    rows = client._parse_filelist_csv(csv_text)
    assert len(rows) == 2
    assert rows[0].file_id == 562
    assert rows[0].tic_id == 365938305
    assert rows[0].type == "Image"
    assert rows[0].filename == "TOI1262_sensitivity.dat"


def test_filter_rows(tmp_path):
    client = ExoFopClient(cache_dir=tmp_path)
    rows = client._parse_filelist_csv(
        "Type,File Name,File ID,TIC,TOI,Date,User,Group,Tag,Description\n"
        "Image,a.pdf,1,365938305,1262.01,2020-01-01,u,g,10,desc\n"
        "Spectrum,b.fits,2,365938305,1262.01,2020-01-02,u,g,11,desc\n"
        "Other,c.txt,3,365938305,1262.01,2020-01-03,u,g,12,desc\n"
    )
    sel = ExoFopSelectors(types={"Spectrum"}, filename_regex=r"\.fits$", tag_ids={11})
    out = client._filter_rows(rows, sel)
    assert [r.filename for r in out] == ["b.fits"]
