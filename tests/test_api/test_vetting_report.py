import tess_vetter.api as btv
from tess_vetter.api.types import error_result, ok_result, skipped_result
from tess_vetter.api.vetting_report import (
    FORMAT_CHECK_RESULT_CALL_SCHEMA,
    FORMAT_VETTING_TABLE_CALL_SCHEMA,
    RENDER_VALIDATION_REPORT_MARKDOWN_CALL_SCHEMA,
    SUMMARIZE_BUNDLE_CALL_SCHEMA,
    VETTING_REPORT_BUNDLE_KEYS,
    VETTING_REPORT_COUNTS_KEYS,
    VETTING_REPORT_RESULT_KEYS,
    VETTING_REPORT_SCHEMA_VERSION,
)


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


def test_vetting_report_contract_constants_are_stable() -> None:
    assert VETTING_REPORT_SCHEMA_VERSION == 1
    assert VETTING_REPORT_COUNTS_KEYS == ("checks", "ok", "error", "skipped")
    assert VETTING_REPORT_RESULT_KEYS == (
        "id",
        "name",
        "status",
        "confidence",
        "metrics",
        "flags",
        "notes",
    )
    assert VETTING_REPORT_BUNDLE_KEYS == ("counts", "results_by_id", "inputs_summary", "provenance")
    assert FORMAT_VETTING_TABLE_CALL_SCHEMA["type"] == "object"
    assert FORMAT_CHECK_RESULT_CALL_SCHEMA["type"] == "object"
    assert SUMMARIZE_BUNDLE_CALL_SCHEMA["type"] == "object"
    assert RENDER_VALIDATION_REPORT_MARKDOWN_CALL_SCHEMA["type"] == "object"
