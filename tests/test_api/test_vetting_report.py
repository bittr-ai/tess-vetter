import tess_vetter.api as btv
from tess_vetter.api.types import error_result, ok_result, skipped_result


def test_format_vetting_table_smoke() -> None:
    bundle = btv.VettingBundleResult(
        results=[
            ok_result(id="V01", name="odd_even", metrics={"delta_ppm": 12.3}, confidence=0.9),
            skipped_result(id="V06", name="nearby_eb", reason_flag="NETWORK_DISABLED"),
        ],
        warnings=[],
        provenance={"duration_ms": 12.3},
        inputs_summary={"network": False},
    )

    out = btv.format_vetting_table(bundle)
    assert "Vetting Results" in out
    assert "V01" in out and "V06" in out


def test_summarize_bundle_contains_counts_and_results() -> None:
    bundle = btv.VettingBundleResult(
        results=[
            ok_result(id="V01", name="odd_even", metrics={"delta_ppm": 12.3}, confidence=0.9),
            error_result(id="V99", name="boom", error="RuntimeError"),
        ],
        warnings=["x"],
        provenance={"pipeline_version": "x"},
        inputs_summary={"has_tpf": False},
    )

    s = btv.summarize_bundle(bundle)
    assert s["counts"]["checks"] == 2
    assert s["counts"]["ok"] == 1
    assert s["counts"]["error"] == 1
    assert "V01" in s["results_by_id"]


def test_render_validation_report_markdown_smoke() -> None:
    bundle = btv.VettingBundleResult(
        results=[ok_result(id="V01", name="odd_even", metrics={"delta_ppm": 12.3})],
        warnings=[],
        provenance={},
        inputs_summary={},
    )
    md = btv.render_validation_report_markdown(title="Test", bundle=bundle)
    assert md.startswith("# Test")
    assert "```" in md


def test_summarize_bundle_preserves_status_literals() -> None:
    bundle = btv.VettingBundleResult(
        results=[
            ok_result(id="V01", name="odd_even", metrics={}),
            skipped_result(id="V06", name="nearby_eb", reason_flag="NETWORK_DISABLED"),
            error_result(id="V99", name="boom", error="RuntimeError"),
        ],
        warnings=[],
        provenance={},
        inputs_summary={},
    )

    s = btv.summarize_bundle(bundle)
    assert s["results_by_id"]["V01"]["status"] == "ok"
    assert s["results_by_id"]["V06"]["status"] == "skipped"
    assert s["results_by_id"]["V99"]["status"] == "error"
