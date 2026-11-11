# Historical Hackathon Reports

> **Notice:** The reports and JSON snapshots in `docs/reports/historical/` were produced during the initial hackathon phase (October 2026) and are retained strictly for historical provenance.

## Known Caveats in Historical Reports

1. **Synthetic Fallback vs. Real Weights:**
   - The comparative audit (`comparative_audit_2026-10-28.md`) was executed on `FallbackQwen2ForCausalLM` (a standalone GQA architecture harness with structured synthetic weights), **not** pretrained `Qwen/Qwen2.5-1.5B-Instruct` weights.
   - The real 7B runs (`7b_gpu_validation/FINAL_REPORT_REAL_7B.md`) loaded pretrained `Qwen/Qwen2.5-7B-Instruct` with CPU offloading on an 8 GB RTX 4060 Laptop GPU, where reduced calibration sequences (`3 prompts x 64 tokens`) degraded subspace probe correlation (`Spearman rho = -0.022`).

2. **Logical vs. Physical Memory Accounting:**
   - Memory reduction percentages reported in these files reflect **logical tier byte telemetry** (`compressed_bytes / fp16_bytes`), not physical GPU VRAM reduction (`torch.cuda.max_memory_allocated()`), because the prototype cache materialized full-length rectangular tensors for standard SDPA attention and called `super().update(...)` on the base `DynamicCache`.
