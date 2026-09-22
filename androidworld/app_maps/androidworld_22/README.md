# AndroidWorld 22-app map set

Canonical map set used by the full-suite Macro+Map evaluation attempt.

- Build parameters: `max_screens=50`, `max_depth=3`, `scroll_pages=3`
- Validation: 22 expected maps, 22 valid maps
- Model used for classification/enrichment: `qwen/qwen3.7-plus`
- Raw per-app crawler logs are local under `artifacts/map_logs/androidworld_22/`

Some apps expose very little stable accessibility structure (for example
Camera, Clipper, OpenTracks, and Pro Expense), so their valid YAML files contain
only one or two discovered screens. In those tasks the runner falls back to the
VLM when no matching macro is available.
