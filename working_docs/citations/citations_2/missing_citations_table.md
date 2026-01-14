| Module | Callable | Triage |
|---|---|---|
| `bittr_tess_vetter.api.ttv_track_search` | `estimate_search_cost` | algorithmic (TTV search heuristics) |
| `bittr_tess_vetter.api.ttv_track_search` | `identify_observing_windows` | algorithmic (TTV search heuristics) |
| `bittr_tess_vetter.api.ttv_track_search` | `run_ttv_track_search` | algorithmic (TTV search heuristics) |
| `bittr_tess_vetter.api.ttv_track_search` | `run_ttv_track_search_for_candidate` | algorithmic (TTV search heuristics) |
| `bittr_tess_vetter.api.ttv_track_search` | `should_run_ttv_search` | algorithmic (TTV search heuristics) |
| `bittr_tess_vetter.api.catalog` | `vet_catalog` | algorithmic (catalog vetting orchestration) |
| `bittr_tess_vetter.api.exovetter` | `vet_exovetter` | algorithmic (exovetter orchestration) |
| `bittr_tess_vetter.api.pixel_localize` | `localize_transit_host_multi_sector` | algorithmic (pixel localization orchestrator) |
| `bittr_tess_vetter.api.pixel_localize` | `localize_transit_host_single_sector` | algorithmic (pixel localization orchestrator) |
| `bittr_tess_vetter.api.pixel_localize` | `localize_transit_host_single_sector_with_baseline_check` | algorithmic (pixel localization orchestrator) |
| `bittr_tess_vetter.api.recovery` | `prepare_recovery_inputs` | algorithmic (recovery helpers) |
| `bittr_tess_vetter.api.recovery` | `stack_transits` | algorithmic (recovery helpers) |
| `bittr_tess_vetter.api.stitch` | `stitch_lightcurve_data` | algorithmic (stitching) |
| `bittr_tess_vetter.api.stitch` | `stitch_lightcurves` | algorithmic (stitching) |
| `bittr_tess_vetter.api.lc_only` | `vet_lc_only` | algorithmic (vetting orchestration) |
| `bittr_tess_vetter.api.pixel` | `vet_pixel` | algorithmic (vetting orchestration) |
| `bittr_tess_vetter.api.vet` | `vet_many` | algorithmic (vetting orchestration) |
| `bittr_tess_vetter.api.aperture` | `create_circular_aperture_mask` | unknown |
| `bittr_tess_vetter.api.triceratops_cache` | `get_disposition` | utility (cache helper) |
| `bittr_tess_vetter.api.references` | `cite` | utility (citation system) |
| `bittr_tess_vetter.api.references` | `cites` | utility (citation system) |
| `bittr_tess_vetter.api.references` | `collect_module_citations` | utility (citation system) |
| `bittr_tess_vetter.api.references` | `generate_bibliography_markdown` | utility (citation system) |
| `bittr_tess_vetter.api.references` | `generate_bibtex` | utility (citation system) |
| `bittr_tess_vetter.api.references` | `get_all_references` | utility (citation system) |
| `bittr_tess_vetter.api.references` | `get_function_references` | utility (citation system) |
| `bittr_tess_vetter.api.references` | `get_reference` | utility (citation system) |
| `bittr_tess_vetter.api.references` | `reference` | utility (citation system) |
| `bittr_tess_vetter.api.primitives_catalog` | `list_primitives` | utility (introspection) |
| `bittr_tess_vetter.api.transit_masks` | `get_out_of_transit_mask_windowed` | utility (masking) |
| `bittr_tess_vetter.api.pipeline` | `describe_checks` | utility (pipeline introspection) |
| `bittr_tess_vetter.api.pipeline` | `list_checks` | utility (pipeline introspection) |
| `bittr_tess_vetter.api.canonical` | `canonical_hash` | utility (serialization/hash/evidence) |
| `bittr_tess_vetter.api.canonical` | `canonical_hash_prefix` | utility (serialization/hash/evidence) |
| `bittr_tess_vetter.api.canonical` | `canonical_json` | utility (serialization/hash/evidence) |
| `bittr_tess_vetter.api.evidence` | `checks_to_evidence_items` | utility (serialization/hash/evidence) |
| `bittr_tess_vetter.api.evidence_contracts` | `compute_code_hash` | utility (serialization/hash/evidence) |
| `bittr_tess_vetter.api.evidence_contracts` | `load_evidence` | utility (serialization/hash/evidence) |
| `bittr_tess_vetter.api.evidence_contracts` | `save_evidence` | utility (serialization/hash/evidence) |
