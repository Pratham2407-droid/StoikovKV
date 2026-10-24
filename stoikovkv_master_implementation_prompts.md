# StoikovKV: Copy-Paste Master Coding Prompts (24-Hr Hackathon Build)

You can either paste the **Master End-to-End Prompt** into an agentic IDE (like Cursor Composer / Claude Code / Windsurf) to scaffold the whole repository at once, or distribute the **3 Modular Teammate Prompts** below so each of you can build your exact component in parallel without merge conflicts.

---

## Option A: The Master End-to-End Prompt (Scaffold Complete Repo)

*Copy and paste everything inside the block below into your AI coding assistant:*

```text
Act as a Principal ML Systems Engineer. Build a complete, runnable, modular Python prototype and FastAPI + Next.js/HTML live dashboard for our hackathon project: "StoikovKV: FlashAttention-Compatible Hybrid Evict-or-Encode KV Cache with Matryoshka Self-Speculative Decoding".

### 1. Hardware & Model Constraints
- Target Model: Default to `Qwen/Qwen2.5-1.5B-Instruct` or `Qwen/Qwen2.5-3B-Instruct` (configurable to `meta-llama/Llama-3.1-8B-Instruct`) using HuggingFace `transformers` and `torch` (FP16/BF16) on a single 16GB–24GB CUDA GPU (fallback gracefully to CPU/MPS for local UI testing).
- Must subclass `transformers.cache_utils.DynamicCache` so it hooks directly into `model.generate()` and `model.forward(..., past_key_values=cache)` WITHOUT requiring custom C++/CUDA compilation.
- Must work with `attn_implementation="sdpa"` or `"flash_attention_2"` during inference (DO NOT require `output_attentions=True` during live inference; only use `output_attentions=True` in the one-time offline calibration script).

### 2. Repository Structure to Generate
Create the following clean file structure with working, bug-free code (no `TODO` placeholders):
1. `stoikov_kv/calibration.py`
2. `stoikov_kv/elastic_lexico.py`
3. `stoikov_kv/controller.py`
4. `stoikov_kv/cache.py`
5. `stoikov_kv/matryoshka_spec.py`
6. `benchmarks/run_eval.py`
7. `server/app.py` & `server/index.html` (Single-file Tailwind + Chart.js live WebSocket/SSE dashboard)

### 3. Detailed Module Specifications

#### Module 1: `stoikov_kv/calibration.py` (Offline Calibration & Head Profiler)
Write a fast calibration function `calibrate_model(model, tokenizer, num_samples=16)` that runs a short calibration pass (`output_attentions=True`) on sample prompts + a synthetic Needle-in-a-Haystack prompt (`"The secret passkey is 94812"`):
- **GQA-Aware Head Indexing:** Account for Grouped-Query Attention (`num_key_value_heads` vs `num_attention_heads`, where `q_per_kv = num_q_heads // num_kv_heads`).
- **Query Subspace Basis (`U_Q`):** Collect Query states `Q`, reshape per KV head group, and run `torch.linalg.svd` to extract the top `r = 16` orthonormal principal components `U_Q` of shape `[num_layers, num_kv_heads, head_dim, 16]`.
- **Per-Head Entropy (`H_h`):** Compute average attention entropy `H = -(attn * (attn + 1e-9).log()).sum(dim=-1).mean()` aggregated per KV head.
- **Retrieval Head Mask:** Identify which KV heads attend most strongly to the needle token span when predicting the passkey. Tag the top 5% of KV heads as `ROLE_RETRIEVAL = 0`. Among remaining heads, tag the lower-entropy half as `ROLE_PEAKED_H2O = 1` and the higher-entropy half as `ROLE_DIFFUSE_LEXICO = 2`.
- **Universal Dictionary (`D`) & Preconditioned ElasticNet Matrix (`W_ridge`):**
  For each layer, fit a normalized dictionary `D` of shape `[head_dim, N_atoms]` (`N_atoms = 512` or `1024`) from collected KV states using fast SVD/K-SVD initialization + 50 steps of Adam dictionary updates. Precompute the ridge projection matrix:
  `W_ridge = torch.linalg.solve(D.T @ D + lambda2 * torch.eye(N_atoms), D.T)` of shape `[N_atoms, head_dim]`.
- Save all calibrated artifacts to `calibrated_artifacts.pt`.

#### Module 2: `stoikov_kv/elastic_lexico.py` (1-Step ElasticNet Sparse Coder + Online L* Adaptation)
- Implement `ElasticLexicoCodec`:
  - `encode(X, s_budget, lambda1)`: Given tensor `X` of shape `[batch, heads, seq, head_dim]`, compute `Z = X @ W_ridge.T`, apply soft-thresholding `Y = torch.sign(Z) * torch.relu(Z.abs() - lambda1)`, select the top `s_budget` atoms by magnitude `|Y|` (`torch.topk`), sort them in descending order of magnitude (`|y_(1)| >= |y_(2)| >= ... >= |y_(s)|`), and apply a 1-step norm-rescaling factor so `||D @ y_top||_2 == ||X||_2` (eliminating L1 shrinkage bias).
  - Store compressed representations compactly as `indices` (`int16`, shape `[..., s]`) and `values` (`float8_e4m3fn` or `int8` + scale, shape `[..., s]`), reporting exact byte footprint `(3 * s + 2)` bytes per vector.
  - `decode(indices, values, active_atoms=None)`: Reconstruct `X_hat` using only the first `m = active_atoms` (for Matryoshka drafting) or all `s` atoms (for verification) via `torch.gather` / einsum over `D`.
  - `adapt_online_dictionary(residuals, M_adapt=64)`: Compute the top-64 SVD singular vectors of prefill reconstruction residuals `R = X - X_hat` when sequence length exceeds the theoretical break-even length `L_star = (2 * head_dim * M_adapt) / (3 * (s_univ - s_adapt))`.

#### Module 3: `stoikov_kv/controller.py` (Write-Time Subspace Probe + HMM + Avellaneda-Stoikov)
- **Write-Time Importance Probe:** Without using attention weights, score incoming tokens via:
  `I_t = torch.norm(K_t @ U_Q, dim=-1) * torch.norm(V_t, dim=-1)` plus an explicit sink boost `+ 100.0` for token positions `0..3`.
- **3-State HMM Regime Tracker:** Track a running exponential moving average and variance of subspace entropy across steps. When a sudden regime shift (z-score > 1.8) is detected, trigger a decay multiplier `beta_t = 0.65` (otherwise `beta_t = 0.98`) on accumulated H2O importance scores: `V_i(t) = beta_t * V_i(t-1) + I_t`.
- **Avellaneda-Stoikov VRAM Pricer:** Given normalized memory pressure `q_t = current_kv_bytes / max_kv_budget_bytes` (in `[0, 1]`), risk aversion `gamma`, and attention volatility `sigma_sq`, compute:
  `r_t = mean_V + q_t * gamma * sigma_sq` and spread `delta_t`.
  Use `q_t` and `r_t` to dynamically set:
  - Peaked H2O head keep-ratio (between `0.15` and `0.35`)
  - Diffuse Lexico head sparsity `s` (e.g., `s=16` when `q_t < 0.5`, `s=8` when `0.5 <= q_t < 0.8`, `s=4` when `q_t >= 0.8`).

#### Module 4: `stoikov_kv/cache.py` (`StoikovKVCache(DynamicCache)`)
- Subclass HuggingFace `DynamicCache`.
- Maintain an uncompressed local sliding window of the most recent `n_b = 64` tokens in FP16 for all heads.
- Whenever the sequence grows past `n_b`, flush older tokens per head according to `head_role`:
  - `ROLE_RETRIEVAL (0)`: Keep in uncompressed FP16.
  - `ROLE_PEAKED_H2O (1)`: Evict tokens with lowest decayed write-time score `V_i(t)`, keeping only budget `k_h2o`.
  - `ROLE_DIFFUSE_LEXICO (2)`: Compress tokens via `ElasticLexicoCodec` with `s` atoms determined by the Avellaneda-Stoikov controller.
- Expose `cache.set_matryoshka_mode(active_atoms=m)` so decoding can toggle between reading `m=4` atoms (Draft) and `s=16` atoms (Verify).
- Track real-time telemetry: `fp16_equivalent_bytes`, `actual_compressed_bytes`, `compression_ratio`, `q_t`, `r_t`, `hmm_regime`, and `bytes_read_per_step`.

#### Module 5: `stoikov_kv/matryoshka_spec.py` (Nested-Atom Self-Speculative Decoding)
- Implement `matryoshka_generate(model, tokenizer, input_ids, cache, gamma_spec=4, draft_atoms=4, verify_atoms=16, max_new_tokens=64)`:
  1. **Draft loop (`gamma_spec` steps):** Set `cache.set_matryoshka_mode(active_atoms=draft_atoms)`. Autoregressively sample `gamma_spec` draft tokens and record draft probabilities `q(x)`.
  2. **Parallel Verify pass (1 step):** Set `cache.set_matryoshka_mode(active_atoms=verify_atoms)`. Pass all `gamma_spec` drafted tokens into `model(draft_ids, past_key_values=cache)` in a single forward pass to get target probabilities `p(x)`.
  3. **Rejection Sampling:** Accept token `j` with probability `min(1.0, p(x_j) / q(x_j))`. On first rejection, resample from the adjusted distribution `norm(max(0, p - q))`, rollback the KV cache to the accepted length, and repeat.
  4. Return generated text, acceptance rate `alpha`, and total KV bytes read vs standard decoding.

#### Module 6: `benchmarks/run_eval.py` & `server/app.py`
- In `benchmarks/run_eval.py`, write a self-contained script that runs in under 3 minutes and outputs JSON + matplotlib plots for:
  1. Write-time Subspace Probe Spearman correlation & Recall@20% vs exact H2O attention sums.
  2. ElasticNet 1-step encoding latency & reconstruction MSE vs sequential greedy OMP (`s=8, 16`).
  3. 4-way comparison on a multi-turn Needle-in-a-Haystack + topic-shift prompt (`Full FP16` vs `Vanilla H2O` vs `Vanilla Lexico` vs `StoikovKV`).
  4. Break-even context length `L*` plot on English vs Kannada/Code prompts.
- In `server/app.py` and `server/index.html`, create a FastAPI server and a sleek dark-mode FinOps dashboard displaying live token generation, a color-coded Head Router map (Retrieval FP16 / Peaked H2O / Diffuse Lexico), live Avellaneda-Stoikov VRAM pressure `q_t` slider, and Matryoshka acceptance rate `alpha`.
```

---

## Option B: 3 Parallel Teammate Prompts (Divide & Conquer)

If all three of you want to code simultaneously on your own laptops and merge in Hour 12, use these individual prompts:

### Teammate 1 Prompt: Write-Time Subspace Probe & 1-Step ElasticNet Lexico (`calibration.py` + `elastic_lexico.py`)
```text
Write a standalone PyTorch module (`elastic_lexico.py` and `calibration.py`) that implements two core algorithms for KV cache compression on HuggingFace Transformers (`Qwen2.5-1.5B-Instruct` / `Llama-3.1-8B`):

1. **Write-Time Query Subspace Probe (FlashAttention-Compatible H2O Proxy):**
   - Given calibration `(Q, K, V)` tensors of shape `[layers, batch, heads, seq, head_dim]`, account for Grouped-Query Attention (GQA) and compute the top `r=16` SVD principal components `U_Q` of the Query states per KV head.
   - Implement a write-time importance scorer `score_tokens_write_time(K, V, U_Q)` that computes `||K @ U_Q||_2 * ||V||_2` (plus an attention sink gate for positions 0..3) without using the attention matrix.
   - Include an evaluation function that compares this write-time score against ground-truth H2O cumulative attention scores (`attn_weights.sum(dim=-2)`) and prints the Spearman rank correlation and Recall@20%.

2. **Preconditioned ElasticNet Sparse Dictionary Encoder (Replacing Lexico OMP):**
   - Train or initialize a universal dictionary `D` of shape `[head_dim=128, N=1024]` with unit-norm columns.
   - Implement both standard sequential Orthogonal Matching Pursuit (`omp_encode(K, D, s=16)`) AND our fast 1-step Preconditioned ElasticNet encoder (`elasticnet_encode(K, D, W_ridge, s=16, lambda1=0.05)`), where `W_ridge = (D^T D + lambda2 I)^(-1) D^T` is precomputed offline, followed by soft-thresholding, top-`s` selection sorted by descending coefficient magnitude (`|y_(1)| >= ... >= |y_(s)|`), and norm-preserving debiasing.
   - Include a benchmark script comparing wall-clock encoding time (ms), reconstruction MSE, and Softmax logit KL-divergence of 1-Step ElasticNet vs sequential OMP, plus an online SVD residual atom adaptation test (`M_adapt = 64` atoms) showing the crossover length `L*` on Kannada text vs English text.
```

---

### Teammate 2 Prompt: Per-Head Evict/Encode Router, Retrieval Guard & HMM (`cache.py`)
```text
Write a PyTorch module (`cache.py`) that subclasses HuggingFace's `transformers.cache_utils.DynamicCache` to implement a hybrid per-head KV cache router combining H2O eviction and Lexico sparse coding:

1. **Offline Head Profiler (`profile_heads`):**
   - Given a HuggingFace CausalLM and a synthetic Needle-in-a-Haystack prompt (`"The secret passkey is 94812"`), run one forward pass with `output_attentions=True`.
   - Compute per-KV-head Attention Entropy `H_h` and Needle Retrieval Copy Score (attention paid from the final answer token to the needle token span).
   - Partition all `(layer, kv_head)` pairs into 3 roles:
     - Top 5% retrieval score -> `RETRIEVAL_PROTECTED` (100% uncompressed FP16)
     - Low entropy (`H_h < median`) -> `PEAKED_H2O` (Evict low-scoring tokens outside a recent window `n_b = 64`)
     - High entropy (`H_h >= median`) -> `DIFFUSE_LEXICO` (Compress tokens outside `n_b = 64` into sorted top-`s` dictionary atoms).

2. **3-State HMM Topic-Shift Decay:**
   - Track the mean and variance of write-time importance entropy at each decoding step. When a regime shift (entropy spike > 1.8 std) occurs in a multi-turn conversation, apply a decay factor `beta = 0.65` to historical H2O scores so stale early-prompt tokens are evicted in favor of the new topic.

3. **Benchmark Script:**
   - Include a runnable script that tests 4 policies at a ~20% KV byte budget on a multi-turn prompt containing a hidden passkey followed by a topic shift: (a) Full FP16, (b) Vanilla H2O only, (c) Vanilla Lexico only, and (d) Our Hybrid Router + Retrieval Guard + HMM. Output a clean table and matplotlib Pareto plot of Accuracy vs Bytes/Token.
```

---

### Teammate 3 Prompt: Matryoshka Self-Speculative Decoding & Avellaneda–Stoikov Live Dashboard (`matryoshka_spec.py` + `server/`)
```text
Build two connected components in Python (`matryoshka_spec.py` and `server/app.py` + `server/index.html`):

1. **Avellaneda-Stoikov VRAM Controller & Matryoshka Self-Speculative Decoder (`matryoshka_spec.py`):**
   - Implement `AvellanedaStoikovController(gamma=0.35, vram_cap_mb=4096)`: Given current KV cache memory utilization `q_t in [0, 1]` and attention volatility `sigma_sq`, compute reservation price `r_t = V_bar + q_t * gamma * sigma_sq * tau` and spread `delta_t`. Map `q_t` and `r_t` to active Lexico sparsity `s in {16, 12, 8, 4}` and H2O eviction budget.
   - Implement `matryoshka_speculative_decode`: Given a KV cache where each token's sparse code stores `s=16` dictionary atoms sorted by magnitude (`|y_1| >= ... >= |y_16|`), draft `K=4` candidate tokens by reconstructing KV states from ONLY the top `m=4` atoms (`3*4+2 = 14 bytes/vector`), and then verify all `K=4` drafted tokens in a single parallel forward pass using all `s=16` atoms (`50 bytes/vector`) via speculative rejection sampling.
   - Measure and return token acceptance rate `alpha`, effective tokens generated per forward pass, and KV memory bandwidth saved (bytes read per verified token).

2. **FastAPI + Single-File HTML/Tailwind/Chart.js FinOps Dashboard (`server/app.py` & `server/index.html`):**
   - Create a FastAPI backend with a streaming `/api/generate` endpoint and `/api/stress_test` endpoint.
   - Build a modern dark-themed frontend dashboard (`index.html`) with:
     - Interactive sliders for **VRAM Pressure (`q_t`)**, **Risk Aversion (`gamma`)**, and **Draft Atoms (`m` vs `s`)**.
     - Live side-by-side streaming generation comparing `Baseline H2O`, `Baseline Lexico`, and `StoikovKV` on a Needle-in-a-Haystack coding prompt.
     - Real-time charts showing: (1) KV Memory Footprint (MB) vs OOM threshold, (2) Per-Head Router Heatmap (Protected FP16 vs H2O Evict vs Lexico Sparse), (3) Avellaneda-Stoikov Bid/Ask Spread bands, and (4) Akamai Edge Batch Concurrency & $/1M Token Savings calculator.
```