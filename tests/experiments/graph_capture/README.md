# rtcd_exp_graph_capture (Track G0)

Feasibility experiment behind `CheckerConfig::graph_prepare` / `Backend::prepareCapturable`: can the detector's
per-batch sequence (zero hits, staging copy, `prepare()` incl. `optixAccelBuild`, `optixLaunch`, result copy) be
captured into a CUDA graph, and what does it save? Findings: `docs/HANDOFF.md`, section 5.

Build only explicitly (not part of AllNew): `agent_build.ps1 -WT <root> -Targets rtcd_exp_graph_capture`; run from
the repository root: `build/bin/experiments/rtcd_exp_graph_capture.exe --help`.

**Warning:** every run that puts an OptiX call into a stream capture (`--what launch`, `correctness`, `failures`,
and `fallback` / `fallback_update` for mesh and the curve modes) corrupts the OptiX device context for the rest of
that process (an internal OptiX event is left invalid: later `optixLaunch` calls fail with
`OPTIX_ERROR_INVALID_VALUE`; a captured quadratic-curve build crashes with an illegal address, which kills the CUDA
context). Run each of those in its own process, one mode at a time. `--what timing` and `--what threads` capture
no OptiX call and are safe.
