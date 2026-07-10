# Changelog

All notable changes to POSITMJ are documented here.

**Authors:** Mihir and Jiyan

## [0.1.0] - 2026-07-10

### Added
- Initial release: `DatasetLoader`, `MLPBuilder`, `PositConverter`,
  `PositExporter`, and the end-to-end `PositMLPPipeline` orchestrator.
- Support for built-in datasets (iris, wine, breast_cancer) and external CSVs.
- Configurable MLP architectures for classification and regression.
- Posit<n, es> encoder with correct round-to-nearest-even carry propagation
  through the exponent and regime fields.
- Verilog LUT/ROM and `.mem` hex file export, programmatic and interactive.
- pytest suite with exhaustive round-trip correctness tests and optional
  SoftPosit reference cross-validation.
- Input validation with clear error messages across all pipeline stages.

### Fixed (pre-release hardening)
- Posit encoder previously lost the rounding carry at exponent/regime field
  boundaries, occasionally producing bit patterns one step off from the
  correctly rounded value.
- Extreme underflow previously rounded tiny nonzero magnitudes to exact zero
  instead of the smallest representable nonzero value (minpos).
- Exact power-of-two / regime-boundary ties now correctly round to even,
  matching reference posit implementations.
