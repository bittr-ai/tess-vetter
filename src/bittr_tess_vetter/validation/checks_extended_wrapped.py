"""Extended metrics-only vetting checks (opt-in).

These checks are intentionally *not* part of the default `register_all_defaults()`
set. They surface additional diagnostics that are already implemented in the
codebase (or have stable compute primitives), without embedding policy decisions
in the library.

Check IDs (proposed):
- V16: model competition (AIC/BIC comparisons)
- V17: ephemeris reliability regime diagnostics
- V18: ephemeris sensitivity sweep diagnostics
- V19: alias/harmonic diagnostics
- V20: ghost/scattered-light pixel features
- V21: sector consistency (host-provided measurements via `context`)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bittr_tess_vetter.validation.registry import (
    CheckConfig,
    CheckInputs,
    CheckRegistry,
    CheckRequirements,
    CheckTier,
)
from bittr_tess_vetter.validation.result_schema import (
    CheckResult,
    error_result,
    ok_result,
    skipped_result,
)


def _valid_lc_arrays(inputs: CheckInputs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lc = inputs.lc
    mask = getattr(lc, "valid_mask", None)
    if mask is None:
        return (np.asarray(lc.time, dtype=np.float64), np.asarray(lc.flux, dtype=np.float64), np.asarray(lc.flux_err, dtype=np.float64))
    mask_arr = np.asarray(mask, dtype=bool)
    return (
        np.asarray(lc.time, dtype=np.float64)[mask_arr],
        np.asarray(lc.flux, dtype=np.float64)[mask_arr],
        np.asarray(lc.flux_err, dtype=np.float64)[mask_arr],
    )


def _smooth_template_config(extra: dict[str, Any]) -> Any:
    from bittr_tess_vetter.validation.ephemeris_specificity import SmoothTemplateConfig

    return SmoothTemplateConfig(
        ingress_egress_fraction=float(extra.get("ingress_egress_fraction", 0.2)),
        sharpness=float(extra.get("sharpness", 30.0)),
    )


class ModelCompetitionCheck:
    id = "V16"
    name = "Model Competition"
    tier = CheckTier.AUX
    requirements = CheckRequirements()
    citations = [
        "Coughlin et al. 2016, ApJS 224, 12",
        "Thompson et al. 2018, ApJS 235, 38",
    ]

    def run(self, inputs: CheckInputs, config: CheckConfig) -> CheckResult:
        try:
            time, flux, flux_err = _valid_lc_arrays(inputs)
            extra = config.extra_params

            from bittr_tess_vetter.compute.model_competition import (
                compute_artifact_prior,
                run_model_competition,
            )

            res = run_model_competition(
                time=time,
                flux=flux,
                flux_err=flux_err,
                period=float(inputs.candidate.period),
                t0=float(inputs.candidate.t0),
                duration_hours=float(inputs.candidate.duration_hours),
                bic_threshold=float(extra.get("bic_threshold", 10.0)),
                n_harmonics=int(extra.get("n_harmonics", 2)),
            )

            prior = compute_artifact_prior(
                period=float(inputs.candidate.period),
                sector=int(getattr(inputs.lc, "sector", 0)) or None,
                quality_flags=inputs.context.get("quality_flags"),
                alias_tolerance=float(extra.get("alias_tolerance", 0.01)),
            )

            fits = res.fits
            metrics: dict[str, float | int | str | bool | None] = {
                "winner": str(res.winner),
                "winner_margin_bic": float(res.winner_margin),
                "model_competition_label": str(res.model_competition_label),
                "artifact_risk": float(res.artifact_risk),
                "bic_transit_only": float(fits["transit_only"].bic),
                "bic_transit_sinusoid": float(fits["transit_sinusoid"].bic),
                "bic_eb_like": float(fits["eb_like"].bic),
                "aic_transit_only": float(fits["transit_only"].aic),
                "aic_transit_sinusoid": float(fits["transit_sinusoid"].aic),
                "aic_eb_like": float(fits["eb_like"].aic),
                "artifact_prior_combined_risk": float(prior.combined_risk),
                "artifact_prior_period_alias_risk": float(prior.period_alias_risk),
                "artifact_prior_sector_quality_risk": float(prior.sector_quality_risk),
                "artifact_prior_scattered_light_risk": float(prior.scattered_light_risk),
            }

            flags: list[str] = []
            if str(res.winner) != "transit_only":
                flags.append("MODEL_PREFERS_NON_TRANSIT")

            notes = list(res.warnings)
            provenance = {
                "bic_threshold": float(extra.get("bic_threshold", 10.0)),
                "n_harmonics": int(extra.get("n_harmonics", 2)),
                "alias_tolerance": float(extra.get("alias_tolerance", 0.01)),
            }

            raw = {
                "fits": {k: v.to_dict() for k, v in fits.items()},
                "artifact_prior": prior.to_dict(),
                "warnings": list(res.warnings),
            }

            return ok_result(self.id, self.name, metrics=metrics, flags=flags, notes=notes, provenance=provenance, raw=raw)
        except Exception as e:
            return error_result(self.id, self.name, error=type(e).__name__, notes=[str(e)])


class EphemerisReliabilityRegimeCheck:
    id = "V17"
    name = "Ephemeris Reliability Regime"
    tier = CheckTier.AUX
    requirements = CheckRequirements()
    citations = []

    def run(self, inputs: CheckInputs, config: CheckConfig) -> CheckResult:
        try:
            time, flux, flux_err = _valid_lc_arrays(inputs)
            extra = config.extra_params

            from bittr_tess_vetter.validation.ephemeris_reliability import (
                compute_reliability_regime_numpy,
            )

            stc = _smooth_template_config(extra)

            # Avoid embedding policy thresholds: set "warn thresholds" to extremes so
            # reliability warnings/labels are not driven by defaults inside the library.
            res = compute_reliability_regime_numpy(
                time=time,
                flux=flux,
                flux_err=flux_err,
                period_days=float(inputs.candidate.period),
                t0_btjd=float(inputs.candidate.t0),
                duration_hours=float(inputs.candidate.duration_hours),
                config=stc,
                n_phase_shifts=int(extra.get("n_phase_shifts", 200)),
                phase_shift_strategy=str(extra.get("phase_shift_strategy", "grid")),
                random_seed=int(config.random_seed or 0),
                period_jitter_frac=float(extra.get("period_jitter_frac", 0.002)),
                period_jitter_n=int(extra.get("period_jitter_n", 21)),
                include_harmonics=bool(extra.get("include_harmonics", True)),
                ablation_top_ns=tuple(int(x) for x in extra.get("ablation_top_ns", (1, 3, 5))),
                contribution_top_n=int(extra.get("contribution_top_n", 10)),
                t0_scan_n=int(extra.get("t0_scan_n", 81)),
                t0_scan_half_span_minutes=extra.get("t0_scan_half_span_minutes"),
                p_value_warn_threshold=float("inf"),
                peak_ratio_warn_threshold=float("-inf"),
                ablation_score_drop_warn_threshold=float("inf"),
                top_contribution_warn_fraction=float("inf"),
            )

            metrics: dict[str, float | int | str | bool | None] = {
                "score": float(res.base.score),
                "depth_hat_ppm": float(res.base.depth_hat * 1e6),
                "depth_sigma_ppm": float(res.base.depth_sigma * 1e6),
                "phase_shift_null_p_value": float(res.phase_shift_null.p_value_one_sided),
                "phase_shift_null_z": float(res.phase_shift_null.z_score),
                "null_percentile": float(res.null_percentile),
                "period_neighborhood_best_period_days": float(res.period_neighborhood.best_period_days),
                "period_neighborhood_best_score": float(res.period_neighborhood.best_score),
                "period_neighborhood_second_best_score": float(res.period_neighborhood.second_best_score),
                "period_peak_to_next_ratio": float(res.period_neighborhood.peak_to_next),
                "max_ablation_score_drop_fraction": float(res.max_ablation_score_drop_fraction),
                "top_5_fraction_abs": float(res.top_contribution_fractions.get("top_5_fraction_abs", 0.0)),
                "n_in_transit": int(res.concentration.n_in_transit),
                "effective_n_points": float(res.concentration.effective_n_points),
            }

            raw = res.to_dict()
            provenance = {
                "ingress_egress_fraction": float(stc.ingress_egress_fraction),
                "sharpness": float(stc.sharpness),
            }
            return ok_result(self.id, self.name, metrics=metrics, provenance=provenance, raw=raw)
        except Exception as e:
            return error_result(self.id, self.name, error=type(e).__name__, notes=[str(e)])


class EphemerisSensitivitySweepCheck:
    id = "V18"
    name = "Ephemeris Sensitivity Sweep"
    tier = CheckTier.AUX
    requirements = CheckRequirements(optional_deps=("celerite2",))
    citations: list[str] = []

    def run(self, inputs: CheckInputs, config: CheckConfig) -> CheckResult:
        try:
            time, flux, flux_err = _valid_lc_arrays(inputs)
            extra = config.extra_params

            from bittr_tess_vetter.validation.ephemeris_sensitivity_sweep import (
                compute_sensitivity_sweep_numpy,
            )

            stc = _smooth_template_config(extra)

            res = compute_sensitivity_sweep_numpy(
                time=time,
                flux=flux,
                flux_err=flux_err,
                period_days=float(inputs.candidate.period),
                t0_btjd=float(inputs.candidate.t0),
                duration_hours=float(inputs.candidate.duration_hours),
                config=stc,
                downsample_levels=list(extra.get("downsample_levels", [1, 2, 5])),
                outlier_policies=list(extra.get("outlier_policies", ["none", "sigma_clip_4"])),
                detrenders=list(extra.get("detrenders", ["none", "running_median_0.5d"])),
                include_celerite2_sho=bool(extra.get("include_celerite2_sho", False)),
                stability_threshold=float(extra.get("stability_threshold", 0.20)),
                random_seed=int(config.random_seed or 0),
                gp_max_iterations=int(extra.get("gp_max_iterations", 100)),
                gp_timeout_seconds=float(extra.get("gp_timeout_seconds", 30.0)),
            )

            metrics: dict[str, float | int | str | bool | None] = {
                "metric_variance": res.metric_variance,
                "score_spread_iqr_over_median": res.score_spread_iqr_over_median,
                "depth_spread_iqr_over_median": res.depth_spread_iqr_over_median,
                "n_variants_total": int(res.n_variants_total),
                "n_variants_ok": int(res.n_variants_ok),
                "n_variants_failed": int(res.n_variants_failed),
                "best_variant_id": res.best_variant_id,
                "worst_variant_id": res.worst_variant_id,
            }

            # Do not surface stable=True/False as a verdict; keep in raw.
            raw = res.to_dict()
            provenance = {
                "ingress_egress_fraction": float(stc.ingress_egress_fraction),
                "sharpness": float(stc.sharpness),
            }
            notes = list(res.notes)
            return ok_result(self.id, self.name, metrics=metrics, notes=notes, provenance=provenance, raw=raw)
        except Exception as e:
            return error_result(self.id, self.name, error=type(e).__name__, notes=[str(e)])


class AliasDiagnosticsCheck:
    id = "V19"
    name = "Alias Diagnostics"
    tier = CheckTier.AUX
    requirements = CheckRequirements()
    citations: list[str] = []

    def run(self, inputs: CheckInputs, config: CheckConfig) -> CheckResult:
        try:
            time, flux, flux_err = _valid_lc_arrays(inputs)
            extra = config.extra_params

            from bittr_tess_vetter.validation.alias_diagnostics import (
                compute_harmonic_scores,
                compute_secondary_significance,
                detect_phase_shift_events,
            )

            duration_hours = float(inputs.candidate.duration_hours)
            scores = compute_harmonic_scores(
                time=time,
                flux=flux,
                flux_err=flux_err,
                base_period=float(inputs.candidate.period),
                base_t0=float(inputs.candidate.t0),
                duration_hours=duration_hours,
            )

            base = next((s for s in scores if s.harmonic == "P"), None)
            base_score = float(base.score) if base is not None else 0.0
            base_depth_ppm = float(base.depth_ppm) if base is not None else 0.0

            best_other = None
            for s in scores:
                if s.harmonic == "P":
                    continue
                if best_other is None or s.score > best_other.score:
                    best_other = s

            best_other_harmonic = str(best_other.harmonic) if best_other is not None else None
            best_other_score = float(best_other.score) if best_other is not None else 0.0
            best_other_depth_ppm = float(best_other.depth_ppm) if best_other is not None else 0.0
            ratio = float(best_other_score / base_score) if base_score > 0 else float("inf")

            sec_sig = compute_secondary_significance(
                time=time,
                flux=flux,
                flux_err=flux_err,
                period=float(inputs.candidate.period),
                t0=float(inputs.candidate.t0),
                duration_hours=duration_hours,
            )

            events = detect_phase_shift_events(
                time=time,
                flux=flux,
                flux_err=flux_err,
                period=float(inputs.candidate.period),
                t0=float(inputs.candidate.t0),
                n_phase_bins=int(extra.get("n_phase_bins", 10)),
                significance_threshold=float(extra.get("event_sigma_threshold", 3.0)),
            )
            max_event_sigma = float(max((e.significance for e in events), default=0.0))

            metrics: dict[str, float | int | str | bool | None] = {
                "base_score_P": float(base_score),
                "base_depth_ppm_P": float(base_depth_ppm),
                "best_other_harmonic": best_other_harmonic,
                "best_other_score": float(best_other_score),
                "best_other_depth_ppm": float(best_other_depth_ppm),
                "best_other_over_base_score_ratio": float(ratio),
                "secondary_significance_sigma": float(sec_sig),
                "n_phase_shift_events": int(len(events)),
                "max_phase_shift_event_sigma": float(max_event_sigma),
            }

            raw = {
                "harmonic_scores": [s.__dict__ for s in scores],
                "phase_shift_events": [e.__dict__ for e in events],
            }
            provenance = {
                "n_phase_bins": int(extra.get("n_phase_bins", 10)),
                "event_sigma_threshold": float(extra.get("event_sigma_threshold", 3.0)),
            }
            return ok_result(self.id, self.name, metrics=metrics, provenance=provenance, raw=raw)
        except Exception as e:
            return error_result(self.id, self.name, error=type(e).__name__, notes=[str(e)])


class GhostFeaturesCheck:
    id = "V20"
    name = "Ghost Features"
    tier = CheckTier.PIXEL
    requirements = CheckRequirements(needs_tpf=True)
    citations: list[str] = []

    def run(self, inputs: CheckInputs, config: CheckConfig) -> CheckResult:
        if inputs.tpf is None:
            return skipped_result(self.id, self.name, reason_flag="NO_TPF", notes=["TPF data not provided"])
        try:
            extra = config.extra_params

            from bittr_tess_vetter.validation.ghost_features import compute_ghost_features

            tpf_data = np.asarray(inputs.tpf.flux, dtype=np.float64)
            time = np.asarray(inputs.tpf.time, dtype=np.float64)
            aperture_mask = np.asarray(getattr(inputs.tpf, "aperture_mask", None), dtype=bool)

            if aperture_mask.ndim != 2:
                return skipped_result(
                    self.id,
                    self.name,
                    reason_flag="NO_APERTURE_MASK",
                    notes=["TPF aperture_mask missing; cannot compute ghost features"],
                )

            features = compute_ghost_features(
                tpf_data=tpf_data,
                time=time,
                aperture_mask=aperture_mask,
                period=float(inputs.candidate.period),
                t0=float(inputs.candidate.t0),
                duration_hours=float(inputs.candidate.duration_hours),
                tic_id=int(inputs.tic_id or 0),
                sector=int(getattr(inputs.lc, "sector", 0)),
                background_annulus=tuple(extra.get("background_annulus", (3, 6))),
                prf_sigma=float(extra.get("prf_sigma", 1.0)),
            )

            metrics: dict[str, float | int | str | bool | None] = {
                "ghost_like_score": float(features.ghost_like_score),
                "scattered_light_risk": float(features.scattered_light_risk),
                "aperture_contrast": float(features.aperture_contrast),
                "spatial_uniformity": float(features.spatial_uniformity),
                "prf_likeness": float(features.prf_likeness),
                "edge_gradient_strength": float(features.edge_gradient_strength),
                "background_trend": float(features.background_trend),
                "in_aperture_depth": float(features.in_aperture_depth),
                "out_aperture_depth": float(features.out_aperture_depth),
                "aperture_pixels_used": int(features.aperture_pixels_used),
            }

            return ok_result(self.id, self.name, metrics=metrics, raw=features.to_dict())
        except Exception as e:
            return error_result(self.id, self.name, error=type(e).__name__, notes=[str(e)])


class SectorConsistencyCheck:
    id = "V21"
    name = "Sector Consistency"
    tier = CheckTier.AUX
    requirements = CheckRequirements()
    citations: list[str] = []

    def run(self, inputs: CheckInputs, config: CheckConfig) -> CheckResult:
        measurements = inputs.context.get("sector_measurements")
        if not measurements:
            return skipped_result(
                self.id,
                self.name,
                reason_flag="NO_SECTOR_MEASUREMENTS",
                notes=["Provide context['sector_measurements'] to enable this check"],
            )
        try:
            from bittr_tess_vetter.validation.sector_consistency import (
                SectorMeasurement,
                compute_sector_consistency,
            )

            rows: list[SectorMeasurement] = []
            for m in list(measurements):
                if not isinstance(m, dict):
                    continue
                if "sector" not in m or "depth_ppm" not in m or "depth_err_ppm" not in m:
                    continue
                rows.append(
                    SectorMeasurement(
                        sector=int(m["sector"]),
                        depth_ppm=float(m["depth_ppm"]),
                        depth_err_ppm=float(m["depth_err_ppm"]),
                        duration_hours=float(m.get("duration_hours", 0.0)),
                        duration_err_hours=float(m.get("duration_err_hours", 0.0)),
                        n_transits=int(m.get("n_transits", 0)),
                        shape_metric=float(m.get("shape_metric", 0.0)),
                        quality_weight=float(m.get("quality_weight", 1.0)),
                    )
                )

            if len(rows) < 2:
                return skipped_result(
                    self.id,
                    self.name,
                    reason_flag="INSUFFICIENT_SECTORS",
                    notes=["Need >=2 sector measurements with depth_ppm and depth_err_ppm"],
                )

            extra = config.extra_params
            chi2_threshold = float(extra.get("chi2_threshold", 0.01))
            min_sectors = int(extra.get("min_sectors", 2))
            cls, outliers, chi2_pval = compute_sector_consistency(
                rows, chi2_threshold=chi2_threshold, min_sectors=min_sectors
            )

            # Keep raw chi2_pval as the primary metric. The classification string
            # is included for convenience but should not be treated as a verdict.
            metrics: dict[str, float | int | str | bool | None] = {
                "chi2_p_value": float(chi2_pval),
                "n_sectors_used": int(len(rows)),
                "consistency_class": str(cls),
                "n_outlier_sectors": int(len(outliers)),
            }
            raw = {
                "outlier_sectors": [int(s) for s in outliers],
                "measurements": [m.to_dict() for m in rows],
                "params": {"chi2_threshold": chi2_threshold, "min_sectors": min_sectors},
            }
            return ok_result(self.id, self.name, metrics=metrics, raw=raw)
        except Exception as e:
            return error_result(self.id, self.name, error=type(e).__name__, notes=[str(e)])


def register_extended_checks(registry: CheckRegistry) -> None:
    registry.register(ModelCompetitionCheck())
    registry.register(EphemerisReliabilityRegimeCheck())
    registry.register(EphemerisSensitivitySweepCheck())
    registry.register(AliasDiagnosticsCheck())
    registry.register(GhostFeaturesCheck())
    registry.register(SectorConsistencyCheck())


__all__ = [
    "register_extended_checks",
    "ModelCompetitionCheck",
    "EphemerisReliabilityRegimeCheck",
    "EphemerisSensitivitySweepCheck",
    "AliasDiagnosticsCheck",
    "GhostFeaturesCheck",
    "SectorConsistencyCheck",
]
