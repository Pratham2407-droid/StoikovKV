# 02 — StoikovKV Benchmark Theory, Causal Mechanics & Interpretation Guide

> **Navigation:** [Folder Index (`docs/benchmarks/`)](./README.md) | **Previous:** [01 — StoikovKV Benchmark Numbers Reference](./01_BENCHMARK_NUMBERS.md)

This document is the companion theory and interpretation reference for [01 — StoikovKV Benchmark Numbers Reference](./01_BENCHMARK_NUMBERS.md). For every benchmark suite and unit test in the repository, it explains three things in strict order:
1. **What the Test Means (Definition & Objective):** What exact property, bottleneck, or failure mode is being measured.
2. **How the Result Became the Result (Causal Mechanism & Math):** Why the underlying equations and systems architecture produced the exact numbers recorded in [`01_BENCHMARK_NUMBERS.md`](./01_BENCHMARK_NUMBERS.md).
3. **How to Infer & Interpret It (Diagnostic Guide):** How to read the numbers, compare baselines, and diagnose regressions.

---

## Table of Contents
1. [Section 1: Offline Calibration & Write-Time Query-Subspace Probe](#1-section-1-offline-calibration--write-time-query-subspace-probe)
2. [Section 2: 1-Step Ridge-Preconditioned ElasticNet vs. OMP & FISTA](#2-section-2-1-step-ridge-preconditioned-elasticnet-vs-omp--fista)
3. [Section 3: 4-Way Equal-Budget (~20% KV Bytes) Live Cache Ablation](#3-section-3-4-way-equal-budget-20-kv-bytes-live-cache-ablation)
4. [Section 4: Matryoshka Self-Speculative Decoding Sweep](#4-section-4-matryoshka-self-speculative-decoding-sweep)
5. [Section 5: Multi-Length Needle-in-a-Haystack & Latency Audit](#5-section-5-multi-length-needle-in-a-haystack--latency-audit)
6. [Section 6: WikiText-103 Throughput & 7B Hardware Validation](#6-section-6-wikitext-103-throughput--7b-hardware-validation)
7. [Section 7: Real Pretrained `Qwen/Qwen2.5-1.5B-Instruct` End-to-End GPU Validation](#7-section-7-real-pretrained-qwenqwen25-15b-instruct-end-to-end-gpu-validation)
8. [Section 8: Complete Unit Test Suite Reference (`20/20` Tests)](#8-section-8-complete-unit-test-suite-reference-2020-tests)

---

## 1. Section 1: Offline Calibration & Write-Time Query-Subspace Probe

### 1.1 Write-Time Subspace Probe Correlation (`Spearman rho` & `Recall@20%`)
* **What the Test Means:**
  Standard heavy-hitter eviction (H2O) requires materializing the explicit $O(T^2)$ attention matrix $\text{softmax}(Q K^\top / \sqrt{d})$ at every prefill/decode step to sum cumulative attention received by each token $t$. Modern FlashAttention and PyTorch `scaled_dot_product_attention` (SDPA) **never materialize** the $T \times T$ attention matrix in HBM. This test measures whether StoikovKV's $O(T \cdot r)$ write-time score $I_t$ — computed solely from the incoming token's key $K_t$, value $V_t$, and an offline-calibrated rank-$r$ ($r = 16$) query basis $U_Q \in \mathbb{R}^{d \times r}$ — accurately ranks tokens by their true future attention importance on **strictly held-out prompts**.
* **How the Result Became the Result:**
  1. During offline calibration (`calibrate_query_subspaces` in `src/stoikov_kv/calibration.py`), StoikovKV collects grouped post-RoPE query states across all $g = H_Q / H_{KV}$ query heads sharing each KV head.
  2. It computes both the uncentered SVD of queries $U_{\text{raw}}$ and the mean-query direction $\hat{\mu}_Q = \bar{Q} / \|\bar{Q}\|_2$, placing $\hat{\mu}_Q$ in column `0` of $U_Q$ ($u_0 = \hat{\mu}_Q$) and Gram-Schmidt orthogonalizing the remaining $r - 1$ principal directions against $u_0$.
  3. Because softmax attention $\exp(q \cdot K_t / \sqrt{d})$ is **sign-sensitive** along the mean query direction $\bar{Q}$ (a key anti-aligned with $\bar{Q}$ has $K_t \cdot u_0 < 0$ and receives exponentially suppressed attention even if $\|K_t U_Q\|_2$ is large), `score_write_time` evaluates:
     $$I_t = \phi\!\left((K_t U_Q)_0\right) \cdot \|K_t U_Q\|_2 \cdot \psi\!\left(\|V_t\|_2\right)$$
     where $\phi(\cdot)$ gates positive alignment with the dominant query direction $u_0$ ($\sigma((K_t U_Q)_0)$ at unit scale; $\text{softplus}((K_t U_Q)_0 / \|K_t\|_2)$ when RoPE/RMSNorm scales are large) and $\psi(\|V_t\|_2)$ weights value impact.
  4. Consequently, on the standalone GQA benchmark (`Section 1.1`), the raw directional subspace probe achieves **Spearman $\rho = 0.5776 \pm 0.2240$** (`52.27%` Recall@20% without any sink bonus), outperforming the value-norm-only baseline (`0.5612`). On real pretrained `Qwen/Qwen2.5-1.5B-Instruct` (`Section 7.1`), where value norms alone are virtually uncorrelated with attention (`-0.0354`), the directional subspace probe jumps to **$\rho = 0.4020$ (raw)** and **$\rho = 0.5733$ (sink-augmented)** — a **+0.6087 Spearman $\rho$ improvement** over $\|V_t\|_2$.
* **How to Infer & Interpret It:**
  * **$\rho > 0.40$ (Raw, $t \ge 4$) and $\rho > 0.48$ (Sink-Augmented):** Confirms that $U_Q$ captures the true query manifold on unseen text without needing `output_attentions=True` at inference time.
  * **Comparing Raw vs. Sink-Augmented vs. $\|V_t\|_2$:** If `Raw Spearman rho` drops near `0.0` or equals the $\|V_t\|_2$ baseline on a pretrained checkpoint, it means either RoPE was not applied before projecting onto $U_Q$ or the sign of $(K_t U_Q)_0$ was ignored.

### 1.2 Head Role Classification (`ROLE_RETRIEVAL_FP16`, `ROLE_PEAKED_H2O`, `ROLE_DIFFUSE_LEXICO`)
* **What the Test Means:**
  Not all KV heads in a GQA transformer perform the same computation. Some heads act as **induction/copy heads** (pinpointing exact passkeys or variable names thousands of tokens back), some act as **local/peaked syntax heads** (concentrating $>85\%$ of attention on a handful of heavy-hitter tokens), and others act as **high-entropy diffuse semantic heads** (averaging broadly across context). This test verifies how StoikovKV partitions KV heads into three disjoint compression policies.
* **How the Result Became the Result:**
  1. `classify_head_roles` measures two orthogonal statistics per KV head $(l, h_{kv})$ on calibration prompts containing planted passkey needles:
     - **Needle Attention Mass ($R_{l, h}$):** Mean attention weight placed directly on the planted passkey span when generating the answer token.
     - **Normalized Attention Entropy ($H_{l, h}$):** Shannon entropy $-\sum_j A_{i,j} \log A_{i,j}$ averaged over causal query positions.
  2. Heads in the top `5%` of needle mass ($R_{l,h} \ge \tau_{\text{needle}}$) are locked into **`ROLE_RETRIEVAL_FP16 (0)`** (`5.0%–5.4%` of heads, `2–3` KV heads).
  3. Among remaining heads, those below the 90th-percentile entropy threshold (`3.1755 nats`) have peaked distributions where dropping low-score tokens causes minimal drift; they are assigned **`ROLE_PEAKED_H2O (1)`** (`83.9%–85.0%` of heads).
  4. The highest-entropy `~10%` of heads (`> 3.1755 nats`) spread attention across all tokens; evicting 80% of tokens from these heads destroys their context mean, so they are routed to **`ROLE_DIFFUSE_LEXICO (2)`** (`10.0%–10.7%` of heads) for sparse dictionary coding of every token.
  5. Furthermore, when running on real HuggingFace SDPA models, `calibrate_model` co-locates `ROLE_RETRIEVAL_FP16` and `ROLE_DIFFUSE_LEXICO` onto the same layers so that the remaining `~82%` of layers are **100% pure `ROLE_PEAKED_H2O`** and execute directly on physically compacted `[B, H_kv, k_kept + n_b, d]` tensors.
* **How to Infer & Interpret It:**
  * **`Held-Out Needle Selectivity Score` (`0.9923 – 1.0000`):** Measures whether the `5%` of heads selected on calibration prompts are also the exact heads attending to needles on unseen prompts. A score near `1.0` proves that retrieval head identity is a structural property of the model weights, not an artifact of a specific prompt.

---

## 2. Section 2: 1-Step Ridge-Preconditioned ElasticNet vs. OMP & FISTA

* **What the Test Means:**
  Compressing a diffuse KV head vector $x \in \mathbb{R}^{128}$ into $s$ sparse dictionary atoms from $D \in \mathbb{R}^{128 \times N}$ ($D \hat{y} \approx x, \|\hat{y}\|_0 \le s$) traditionally requires iterative solvers: Orthogonal Matching Pursuit (**OMP**, $s$ sequential least-squares projections) or **FISTA** (tens of proximal gradient steps). Iterative solvers are too slow for live LLM decoding. This benchmark tests whether StoikovKV's **1-step ridge-preconditioned ElasticNet encoder** matches the cosine similarity of OMP/FISTA while executing an order of magnitude faster on GPU.
* **How the Result Became the Result:**
  1. During offline calibration (`train_universal_dictionaries`), StoikovKV learns a normalized dictionary $D$ in the **pre-RoPE frame** (un-rotating keys by $R_{-\theta, t}$ first so position-dependent rotation does not fragment dictionary atoms) and solves the ridge preconditioner in **FP64**:
     $$W_{\text{ridge}} = (D^\top D + \lambda_2 I)^{-1} D^\top, \quad \text{Max Residual} = 7.20 \times 10^{-16}$$
  2. At inference time (`encode_elastic_topk` in `src/stoikov_kv/elastic_lexico.py`), encoding a batch of vectors requires **one single GEMM** $z = W_{\text{ridge}} x$, soft-thresholding $\mathcal{S}_{\lambda_1}(z)$, selecting the top-$s$ magnitude entries, sorting them so $|y_{(1)}| \ge |y_{(2)}| \ge \dots \ge |y_{(s)}|$, performing a single closed-form scalar projection $\alpha^* = \frac{\langle x, D\hat{y}\rangle}{\|D\hat{y}\|_2^2}$, and quantizing to `uint8` indices (`10-bit` packed as `uint8` + `2-bit` high bits) and `int8` values + `fp16` scale (`3s + 2` bytes total).
  3. All `.item()` host-device synchronizations were removed from the hot path, keeping encoding latency flat at **`0.527 – 0.594 ms`** regardless of sparsity $s \in \{4, 8, 12, 16, 32\}$.
  4. In contrast, OMP's runtime scales linearly/quadratically with $s$ (`1.676 ms` at $s=4$ up to `18.776 ms` at $s=32$), and 30-step FISTA takes `8.336 – 8.979 ms`. This yields a **`3.17x – 31.60x` speedup over OMP** and a **`15.11x – 16.87x` speedup over FISTA** while staying within `0.001 – 0.012` cosine similarity of exact iterative solutions (`0.8103` vs. `0.8153` at $s=16$).
* **How to Infer & Interpret It:**
  * **Flat Encode Latency vs. $s$:** Because `encode_elastic_topk` is non-iterative, increasing $s$ from `4` to `32` increases encode latency by only `12%` (`0.529 ms` $\to$ `0.594 ms`), whereas OMP slows down by `11.2x`.
  * **Passkey Ablation (`100.0%` vs `0.0%`):** In Section 2's passkey test, `stoikov_hybrid_s16` (`100.0%` accuracy, `0.9998` logit cosine) and `stoikov_hybrid_draft_m4` (`100.0%` accuracy, `0.9997` logit cosine) preserve the exact needle because `ROLE_RETRIEVAL_FP16` heads are protected in FP16, whereas compressing retrieval heads to $s=1$ (`unprotected_ablation`) collapses accuracy to **`0.0%`** and logit cosine to `0.0786`.

---

## 3. Section 3: 4-Way Equal-Budget (~20% KV Bytes) Live Cache Ablation

* **What the Test Means:**
  This is the primary controlled multi-turn benchmark (`benchmarks/bench_cache_ablation.py`). All three compressed systems (`Standalone H2O`, `Standalone Lexico`, and `StoikovKV Hybrid`) are constrained by `solve_equal_budget_params` to the **exact same ~20% KV memory budget** (`~11.54 – 11.64 MB` vs. `58.40 MB` for `Full FP16`) on a `2,037`-token two-turn conversation:
  - **Turn 1 (Tokens `0..1169`):** Python backend service logs containing a passkey needle (`API_PORT = 8443`) planted at four early depths (`5%`, `12%`, `22%`, `35%`).
  - **Turn 2 (Tokens `1170..2036`):** An abrupt topic shift to CSS glassmorphism stylesheet design rules.
  Each system is tested on:
  1. **Needle `8443` Retrieval Accuracy & Logit Margin** (can it still recall an early Turn-1 detail?),
  2. **Turn-2 CSS Logit Cosine** (how closely do its next-token logits match `Full FP16` after the topic shift?),
  3. **Turn-2 H2O Freshness** (what fraction of its retained H2O heavy-hitter slots belong to the active Turn-2 CSS topic rather than stale Turn-1 Python logs?), and
  4. **Physical Memory & Wall-Clock Evaluation Time**.
* **How the Result Became the Result:**
  1. **Why `Standalone Lexico` fails Needle Retrieval (`50.0%` vs `100.0%`) and runs slowest (`5.83 s`):**
     Forcing all `56` KV heads into sparse dictionary compression at $s = 14$ (`19.79%` KV bytes, `22.93%` incl. dictionary) adds quantization/sparse approximation noise to the `3` sharp induction/retrieval heads. At deep needle positions, this noise blurs the exact token match, cutting needle retrieval to **`50.0%`** and Turn-2 CSS logit cosine to **`0.8569`**. Moreover, encoding/decoding all `56` heads across `28` layers costs `5.83 s` and `1.84 MB` of dictionary overhead.
  2. **Why `Standalone H2O` suffers Stale Cache Pollution (`44.62%` Turn-2 Freshness):**
     With static cumulative attention (`hmm_enabled=False`), Turn-1 Python tokens accumulate scores across `1,170` steps before Turn 2 even begins. Even after `867` tokens of Turn-2 CSS text, more than half (`55.38%`) of `Standalone H2O`'s heavy-hitter slots remain locked by obsolete Turn-1 tokens (`Turn-2 Freshness = 44.62%`). And as shown in Section 5.1, when needle prompts lack high write-time norm on non-retrieval heads, `Standalone H2O` evicts early needles (`0%` early needle recall in the comparative audit).
  3. **Why `StoikovKV Hybrid` achieves `100.0%` Needle Accuracy, `100.00%` Turn-2 Freshness, `0.9093` CSS Cosine, and `2.38 s` Runtime:**
     - **Head Routing:** Only the `3` retrieval heads (`5.4%`) use full FP16 (guaranteeing **`100.0%` needle accuracy** with the highest logit margin **`+2540.95`**), `47` peaked heads (`83.9%`) use H2O (`keep_ratio = 0.1211`), and only `6` diffuse heads (`10.7%`) use Lexico (`s = 10`, cutting dictionary memory by `4.7x` from `1.84 MB` to `393 KB`).
     - **HMM Topic-Shift Decay (`8` shifts detected, `100.00%` freshness):** `HMMTopicShiftController` monitors the cosine drift of the query-subspace projection profile $\|Q_t U_Q\|_2$. When the prompt switches from Python logs to CSS at token `1,170`, the log-likelihood ratio crosses the shift threshold (`8` shifts across the prompt), accelerating H2O score decay from $\beta = 0.995$ to $\beta_{\text{shift}} = 0.65$. Stale Turn-1 scores decay exponentially within ~15 steps, achieving **`100.00%` Turn-2 H2O Freshness**.
     - **1-Token Virtual Partition Sink (`0.9815` H2O / `0.9093` Hybrid CSS Cosine at `1.74 s` / `2.38 s`):** When compacting pure-H2O layers from `2,037` tokens down to `310` retained tokens, dropping `1,727` zero-mean evicted keys reduces the softmax denominator $Z = \sum_j \exp(q \cdot k_j / \sqrt{d})$ by $\approx n_{\text{evicted}} \cdot \exp(0) = 1,727$, which would otherwise over-sharpen attention on the kept tokens. StoikovKV appends **a single virtual partition-sink slot** (`k_sink = 4.05 * ln(n_evicted) * U_Q[..., 0]`, `v_sink = 0`), restoring exact softmax denominator calibration (`0.9093` CSS Cosine for Hybrid, `0.9815` for H2O) while keeping the physical tensor length at `311` tokens (`5.3x` faster than uncompacted execution).
* **How to Infer & Interpret It:**
  * **Logical == Physical KV Memory (`19.76%`):** Because `StoikovKVCache` stores only tier-specific buffers (`h2o_k`, `fp16_k`, or packed `lex_k_idx`/`lex_k_val`) without duplicating tensors in `DynamicCache.key_cache`, the reported `19.76%` is the **actual resident GPU memory footprint**.
  * **Composite Quality vs. Adaptation:** While `Standalone H2O` has a slightly higher static CSS cosine (`0.9815` vs `0.9093`) because it gives all `19.93%` of its budget to `keep_ratio = 0.168`, it fails multi-length early needle retrieval (`0%` in Section 5.1) and retains `55.4%` stale Turn-1 tokens. `StoikovKV Hybrid` is the only system that simultaneously achieves **`100%` needle retrieval**, **`100%` topic-shift freshness**, and **`>0.90` logit cosine** at `<20%` KV memory.

---

## 4. Section 4: Matryoshka Self-Speculative Decoding Sweep

* **What the Test Means:**
  Autoregressive LLM decoding is **memory-bandwidth bound**: generating each token requires reading the entire KV cache from GPU HBM. Standard speculative decoding requires loading and running a separate draft model. This benchmark tests whether **one single model and one single `StoikovKVCache`** can draft $\gamma_{\text{spec}} = 4$ tokens by reading only the first $m \in \{2, 4, 8, 12\}$ sorted dictionary atoms per `ROLE_DIFFUSE_LEXICO` vector (`3m + 2` bytes) and then verify all $\gamma_{\text{spec}}$ draft tokens in **a single batched forward pass** using all $s = 16$ atoms (`50` bytes).
* **How the Result Became the Result:**
  1. Because `encode_elastic_topk` sorts every compressed vector's atoms in strictly descending order of magnitude ($|y_{(1)}| \ge |y_{(2)}| \ge \dots \ge |y_{(s)}|$), any prefix $m < s$ is the optimal $m$-term truncated approximation **without re-encoding or duplicating storage**.
  2. During the $\gamma_{\text{spec}} = 4$ draft steps (`set_matryoshka_mode(active_atoms=m)`), `decode_matryoshka` slices `indices[..., :m]` and `values[..., :m]`, applying an energy-preserving Matryoshka rescale so the norm of the $m$-atom reconstruction matches the $s$-atom reconstruction.
  3. During verification (`set_matryoshka_mode(active_atoms=None)`), `matryoshka_speculative_generate` rolls the cache back to `n_prefix` via `cache.crop(n_prefix)` and passes all $\gamma_{\text{spec}} + 1 = 5$ tokens `[last_tok, d_1, d_2, d_3, d_4]` in **one single batched transformer call**.
  4. When all $\gamma_{\text{spec}} = 4$ draft tokens match the target distribution, the verification pass also emits the $(4+1)$-th **bonus token** from `verify_logits[:, 4, :]` and truncates the cache to `n_prefix + 4`.
  5. Across $m \in \{2, 4, 8, 12\}$, the empirical acceptance rate is **$\alpha = 0.9615$** (`3.33 / 4` accepted draft tokens per verify pass, **`4.33` tokens emitted per verify cycle**), reducing effective KV bytes read per generated token from `30,772.0 KB` (Full FP16) to **`7,422.9 KB` at $m=2$ (`75.88%` total KV bandwidth reduction)**.
* **How to Infer & Interpret It:**
  * **Why $\alpha$ is high (`0.9615` on standalone, `1.0000` on real `Qwen2.5-1.5B-Instruct`):** Because `ROLE_RETRIEVAL_FP16` and `ROLE_PEAKED_H2O` heads remain exact between draft and verify, and the top $m=2..4$ sorted atoms already capture the dominant principal components of `ROLE_DIFFUSE_LEXICO` heads (`Expected E[alpha] = 0.9445 – 0.9601`), draft logits closely track verify logits.
  * **Choosing $m$:** $m = 2$ or $m = 4$ maximizes bandwidth savings (`4.90%` and `4.20%` over standard Hybrid decode; `75.88%` and `75.70%` over FP16) with no drop in acceptance rate.

---

## 5. Section 5: Multi-Length Needle-in-a-Haystack & Latency Audit

* **What the Test Means:**
  Located in `benchmarks/comparative_audit/scripts/run_benchmarks.py`, this suite stress-tests passkey retrieval (`"94812"`) across **three context lengths (`512`, `2,048`, `4,096` tokens)** and **three needle depths (`Early` at 10%, `Middle` at 50%, `Late` at 90%)** over `15` trials (`5` runs $\times$ `3` random seeds) per cell (`135` evaluations per method), alongside multi-seed prefill + 32-token generation throughput.
* **How the Result Became the Result:**
  1. **Early Needle Failure in `Standalone H2O (B3)` (`0%` at `512`, `2,048`, and `4,096` tokens):**
     When the passkey `"94812"` is planted at the `Early` position (first 10% of the prompt), `90%` of the filler context arrives *after* the needle. Under pure H2O eviction (`Standalone H2O`), every KV head evicts tokens once the sequence exceeds the local window `n_b`. Because the query asking *"What is the secret passkey?"* only appears at the very end of the prompt (token `512`, `2,048`, or `4,096`), early needle tokens get evicted by intermediate heavy hitters before the final query ever arrives — resulting in **`0% / 100% / 100%`** (Early / Middle / Late) across all three context lengths.
  2. **100% Retrieval Across All Depths & Lengths in `StoikovKV Hybrid (B4)` (`100% / 100% / 100%`):**
     Because StoikovKV calibrated `ROLE_RETRIEVAL_FP16` offline on needle-copy tasks, the `5%` of heads responsible for long-range copy/induction never evict tokens regardless of whether the needle is at token `50` or token `4,000`. Thus `StoikovKV Hybrid` achieves **`100%` (`15/15`)** across every `(context_length, depth)` combination.
  3. **Generation Throughput (`30.9 tok/s` FP16, `23.1 tok/s` H2O, `18.5 tok/s` Hybrid):**
     Physical compaction (`_materialize_layer_kv`) keeps per-step decode overhead low (`1.73 s` for prefill + 32 decode steps on `StoikovKV Hybrid`).
* **How to Infer & Interpret It:**
  * This table is the clearest demonstration of **why homogeneous eviction (`Standalone H2O`) fails in production RAG/agent workloads**: any fact stated early in a system prompt or document is vulnerable to eviction before the user asks about it at the end of the context. Protecting just `5%` of KV heads (`ROLE_RETRIEVAL_FP16`) completely eliminates early-needle blindness.

---

## 6. Section 6: WikiText-103 Throughput & 7B Hardware Validation

* **What the Test Means:**
  `experiments/7b_gpu_validation/bench_wikitext.py` evaluates long-stream language modeling across `50` WikiText-103 passages (`600` tokens each) to measure sustained **decode throughput (`tok/s`)**, **peak GPU VRAM (`GB`)**, **physical KV memory savings (`%`)**, and **natural HMM regime shift detection** on real Wikipedia articles. Section 6.2 validates `StoikovKVCache` on real **`Qwen/Qwen2.5-7B-Instruct`** weights under an 8 GB VRAM ceiling using CPU/GPU layer offloading.
* **How the Result Became the Result:**
  1. **Physical Compaction Speedup (`1,910.8 tok/s` vs `687.6 tok/s`):**
     On WikiText-103, `Standalone H2O` reaches `3,109.3 tok/s` (`86.5%` memory savings, `0.26 GB` peak VRAM) because all layers are pure H2O. `Standalone Lexico` runs at `687.6 tok/s` (`58.8%` memory savings) because it decodes dictionary atoms on every layer and head. `StoikovKV Hybrid` achieves **`1,910.8 tok/s` (`2.78x` faster than Lexico)** with **`80.0%` physical KV memory savings (`20.0%` of FP16)** and **`0.29 GB` peak VRAM** while detecting **`15` natural topic shifts** across the 50 Wikipedia passages.
  2. **7B Offloaded Validation:**
     On `Qwen/Qwen2.5-7B-Instruct` (`28` layers, `4` KV heads, `head_dim = 128`), placing layers `0..9` on the 8 GB RTX 4060 GPU and layers `10..27` on CPU keeps peak VRAM at `6.94 GB` while `StoikovKVCache` reduces physical KV cache memory by **`60.1% – 67.3%`**.
* **How to Infer & Interpret It:**
  * The **`15` HMM shifts** on WikiText-103 show that `HMMTopicShiftController` is not triggered only by synthetic prompt splices — natural section boundaries in Wikipedia articles also produce statistically significant query-subspace shifts that automatically refresh H2O heavy hitters.

---

## 7. Section 7: Real Pretrained `Qwen/Qwen2.5-1.5B-Instruct` End-to-End GPU Validation

* **What the Test Means:**
  Sections 1–5 use a controlled standalone GQA architecture (`FallbackQwen2ForCausalLM`) so unit tests and ablations can run deterministically without downloading multi-gigabyte checkpoints. Section 7 (`benchmarks/results/real_qwen15b_gpu_metrics.json`) validates the **entire StoikovKV pipeline on real pretrained HuggingFace weights (`Qwen/Qwen2.5-1.5B-Instruct`, `Qwen2ForCausalLM` in `torch.float16` on `cuda:0`)**, testing:
  1. Streaming $O(B \cdot T)$ calibration with `output_attentions=False`,
  2. Write-time subspace probe correlation against true pretrained attention,
  3. Compressed-cache passkey retrieval (`s = 32`, `m = 8`),
  4. Native HuggingFace `model.generate(past_key_values=cache)` drop-in compatibility, and
  5. Live `matryoshka_speculative_generate` on a `412`-token security vault prompt.
* **How the Result Became the Result:**
  1. **Why $\|V_t\|_2$ fails on real `Qwen2.5-1.5B-Instruct` (`-0.0354`) while StoikovKV's Subspace Probe succeeds (`0.5733` / `0.4020`):**
     In real pretrained `Qwen2.5-1.5B-Instruct`, post-RoPE key norms are large (`~70–95`) while value norms (`~7.5`) vary smoothly across tokens and have a slightly *negative* correlation (`-0.0354`) with attention weights. Because `score_write_time` projects post-RoPE keys onto the calibrated query subspace $U_Q$ ($u_0 = \hat{\mu}_Q$), gates alignment via $\text{softplus}((K_t U_Q)_0 / \|K_t\|_2)$, and damps value-norm variance via $\sqrt{\|V_t\|_2}$, it achieves **$\rho = 0.4020 \pm 0.2887$ (raw, $t \ge 4$)** and **$\rho = 0.5733 \pm 0.2716$ (sink-augmented)** across all `336` held-out `(layer, prompt)` pairs.
  2. **Why `calibrate_model` finishes in `8.92 s` on Laptop GPU:**
     Instead of running $O(T^2)$ `eager` attention (`output_attentions=True`), `_collect_Layer_Calibration_Stats` hooks `q_proj`, `k_proj`, and `v_proj` while running Flash/SDPA (`output_attentions=False`) and streams causal key-importance scores in `64`-query chunks on GPU.
  3. **Why Real-Weights `model.generate` and `matryoshka_speculative_generate` Output Exact `'94812.'` at `60.50%` Physical Memory Savings:**
     - On `Qwen2ForCausalLM`, HuggingFace creates a 4D causal `attention_mask` of shape `[B, 1, q_len, total_seq_len]` before calling the attention layers. When pure-H2O layers compact their physical KV tensor from `total_seq_len = 412` to `k_actual = 152` tokens, StoikovKV's SDPA wrapper automatically slices `attention_mask[..., -k_actual:]` so PyTorch's C++ SDPA kernel executes seamlessly on the compacted tensor.
     - As a result, native `model.generate(..., past_key_values=cache)` outputs `'94812.'` while cutting physical KV cache memory from `11,956,224 B` to **`4,722,700 B` (`60.50%` physical memory reduction)**, and `matryoshka_speculative_generate` achieves **`100.0%` empirical draft acceptance ($\alpha = 1.0000$)**.
* **How to Infer & Interpret It:**
  * Section 7 confirms that **every novel mechanism in StoikovKV transfers directly to real pretrained weights**: streaming subspace calibration (`8.92 s`), positive held-out Spearman correlation (`+0.6087` over $\|V_t\|_2$), `0.9923` held-out needle head selectivity, `100%` passkey accuracy under hybrid compression (`0%` unprotected), native `model.generate()` compatibility, and $\alpha = 1.0000$ Matryoshka speculative decoding.

---

## 8. Section 8: Complete Unit Test Suite Reference (`20/20` Tests)

All 20 automated unit tests in `tests/test_calibration_and_codec.py` (11 tests) and `tests/test_cache_and_speculative.py` (9 tests) run via:
```bash
python3 -m unittest tests/test_calibration_and_codec.py tests/test_cache_and_speculative.py -v
```

### 8.1 Offline Calibration & Sparse Codec Tests (`tests/test_calibration_and_codec.py`)

| Test Name | What the Test Means | How the Result Became the Result | How to Infer / Diagnose |
|---|---|---|---|
| `test_1_gqa_grouping_and_subspace_orthonormality` | Verifies that $U_Q$ has shape `(L, H_kv, d, r)` (grouped by KV head, not Q head) and satisfies $U_Q^\top U_Q = I_r$. | Queries from all $g = H_Q / H_{KV}$ query heads sharing a KV head are concatenated before computing the mean-anchored SVD basis. | If $U_Q^\top U_Q \ne I_r$, Gram-Schmidt orthogonalization of $u_0 = \hat{\mu}_Q$ against SVD columns was broken. |
| `test_2_write_time_probe_and_attention_reduction` | Verifies `score_write_time` matches causal GQA attention without `output_attentions=True` and that sink tokens (`0..3`) receive the sink bonus. | Streaming causal reduction computes exact attention column sums in $O(T)$ memory chunks and compares against `score_write_time`. | Ensures calibration never OOMs on long sequences while maintaining positive Spearman $\rho$. |
| `test_3_head_role_assignment_and_determinism` | Verifies deterministic 3-tier classification (`ROLE_RETRIEVAL_FP16`, `ROLE_PEAKED_H2O`, `ROLE_DIFFUSE_LEXICO`) across repeated runs. | Fixed seed + exact quantile thresholds (`top 5%` needle score, `90th` percentile entropy) guarantee bit-identical `head_roles`. | If `head_roles` differs across runs, non-deterministic reduction or random sampling crept into calibration. |
| `test_4_dictionary_and_ridge_matrix_solve` | Verifies unit-norm dictionary columns $\|D_{l,:,k}\|_2 = 1$ and FP64 ridge preconditioner accuracy $\|(D^\top D + \lambda_2 I)W_{\text{ridge}} - D^\top\|_F / \|D^\top\|_F < 10^{-12}$. | `torch.linalg.solve` is executed in `torch.float64` before casting $W_{\text{ridge}}$ back to runtime dtype. | Residuals $>10^{-5}$ indicate FP16/FP32 inversion instability in `train_universal_dictionaries`. |
| `test_5_fast_encoder_dtypes_sorting_zero_safety_and_rescale` | Verifies `encode_elastic_topk` outputs packed `uint8`/`int8`/`fp16` tensors (`3s + 2` bytes), sorts magnitudes descending, handles zero vectors safely, and applies closed-form scalar rescaling. | Checks `torch.all(mags[..., :-1] >= mags[..., 1:])`, verifies zero input produces zero output without `NaN`, and checks $\alpha^*$ projection reduces MSE. | Descending magnitude order is the mathematical prerequisite for Matryoshka prefix slicing (`[:m]`). |
| `test_6_omp_and_iterative_fista_reference_solvers` | Verifies `encode_omp_reference` and `encode_iterative_elastic_net` baselines reconstruct vectors accurately and that `encode_elastic_topk` is faster. | Runs 1-step ElasticNet, OMP, and FISTA on identical held-out vectors and compares cosine similarity and latency. | Confirms the speed/accuracy claims in Section 2 of [`01_BENCHMARK_NUMBERS.md`](./01_BENCHMARK_NUMBERS.md). |
| `test_7_matryoshka_partial_decode_prefix_isolation_and_monotonicity` | Verifies `decode_matryoshka(active_atoms=m)` depends *only* on the first $m$ atoms and that reconstruction MSE decreases monotonically as $m$ grows ($m = 2 \to 4 \to 8 \to 16$). | Corrupts atoms at indices $\ge m$ to prove prefix isolation, then measures $\text{MSE}(m)$ across increasing $m$. | Monotonic MSE reduction guarantees that higher draft $m$ strictly increases speculative acceptance $\alpha$. |
| `test_8_online_adaptation_and_break_even_edge_cases` | Verifies `OnlineRank1DictionaryUpdater` adapts dictionary atoms to out-of-distribution shifts and `compute_domain_breakeven_length` computes $L^* = B_{\text{dict}} / (b_{\text{fp16}} - b_{\text{sparse}})$. | Updates $D$ and $W_{\text{ridge}}$ via Sherman-Morrison rank-1 updates on shifted vectors and checks MSE improvement. | Explains when shared dictionary overhead (`+Dict %`) amortizes across sequence length $L > L^*$. |
| `test_9_artifact_roundtrip_and_hook_cleanup_on_failure` | Verifies `save_calibrated_artifacts` / `load_calibrated_artifacts` round-trip integrity and confirms forward hooks are always removed even if the model raises an exception. | Injects a runtime exception mid-calibration and asserts `_forward_hooks` is empty afterward. | Prevents memory leaks when calibrating models in long-running server processes. |
| `test_10_formula_verification_and_split_disjointness` | Verifies the exact byte formula `3s + 2`, disjoint train/eval prompt splits (`set(calib) & set(eval) == set()`), and pre-RoPE key un-rotation invariance. | Rotates a key by $R_{\theta, t_1}$, un-rotates, encodes/decodes, and rotates to $R_{\theta, t_2}$, verifying position-independent reconstruction. | Proves calibration metrics are never inflated by train/test leakage and keys remain valid across positions. |
| `test_11_compressed_cache_passkey_inference` | End-to-end integration test running `evaluate_compressed_cache_passkey` across `full_fp16`, `stoikov_hybrid_s16`, `stoikov_hybrid_draft_m4`, and `unprotected_ablation`. | Asserts `100%` passkey accuracy on hybrid verify/draft modes and `< 100%` on unprotected ablation. | Directly validates the passkey table in Section 2 of [`01_BENCHMARK_NUMBERS.md`](./01_BENCHMARK_NUMBERS.md). |

### 8.2 Live Cache, Physical Compaction, Speculative Decoding & Server Tests (`tests/test_cache_and_speculative.py`)

| Test Name | What the Test Means | How the Result Became the Result | How to Infer / Diagnose |
|---|---|---|---|
| `test_01_create_mock_artifacts_standalone` | Verifies standalone artifact initialization without requiring internet access or `output_attentions=True`. | Builds valid `(U_Q, head_roles, D, W_ridge)` directly from model projection weights. | Enables fast offline testing and cold-start fallback. |
| `test_02_stoikov_cache_routing_and_telemetry` | Verifies tokens older than `n_b` are routed to FP16, H2O, or Lexico tiers according to `head_roles` and that `get_memory_stats()` matches exact byte counts. | Prefills `128` tokens with `n_b = 32`, triggering `_flush_expired_window_tokens`, and audits per-tier token-head counts. | Confirms conservation of token-heads (`local + fp16 + h2o + lexico + evicted == L * H_kv * T`). |
| `test_03_matryoshka_mode_and_crop_rollback` | Verifies `set_matryoshka_mode(active_atoms=m)` switches reconstruction bandwidth and `crop(max_length)` losslessly rolls back all three tiers and the local window. | Prefills, snapshots state at `T_1`, appends draft tokens to `T_2` (causing window flushes into tiers), calls `crop(T_1)`, and asserts exact state restoration. | Lossless `crop()` across tiers is critical for speculative rejection sampling; without it, rejected draft tokens corrupt the cache. |
| `test_04_matryoshka_speculative_generate` | Verifies end-to-end `matryoshka_speculative_generate` produces valid token sequences and telemetry (`acceptance_rate`, `effective_kv_bytes_read_per_token`). | Runs draft ($m=4$) and verify ($s=16$) cycles and checks output shape and telemetry bounds. | Validates Section 4's speculative decoding pipeline. |
| `test_05_h2o_context_window_and_legacy_rope_frame` | Verifies H2O contextual retention and backward compatibility with legacy post-RoPE vs. pre-RoPE artifacts. | Tests `dict_rope_frame='pre_rope'` and `'post_rope'` flags during live cache updates. | Ensures older saved `.pt` artifacts still load and decode properly. |
| `test_06_p0_physical_memory_compaction_and_no_duplicates` | **P0 Verification:** Asserts `StoikovKVCache` stores `0` duplicate bytes in `DynamicCache.key_cache`/`layers`, does not cache FP16 `lex_k_recon` buffers, and returns physically compact tensors on pure-H2O layers. | Checks `physical_resident_bytes == compressed_bytes < 0.45 * fp16_bytes` and verifies pure-H2O materialized sequence length is `< total_seq_len`. | Proves that StoikovKV's memory savings are **real physical GPU RAM reductions**, not just bookkeeping numbers. |
| `test_07_p1_native_generate_drop_in_compatibility` | **P1 Verification:** Asserts standard HuggingFace `model.generate(input_ids, past_key_values=cache)` works out-of-the-box and implements `get_mask_sizes` / `get_max_cache_shape`. | Invokes `model.generate()` directly with a `StoikovKVCache` instance and verifies generated token parity. | Confirms zero-code-change integration with standard HuggingFace pipelines. |
| `test_08_p3_single_pass_verification_and_strict_sampling` | **P3 Verification:** Verifies verification runs in **1 batched forward pass** of length $\gamma_{\text{spec}} + 1$ (instead of $\gamma_{\text{spec}}$ serial passes), emits the bonus token on full acceptance, and supports temperature sampling. | Counts exact `model.forward` calls (`1` prefill + $\gamma_{\text{spec}}$ draft + `1` verify = `4` calls for $\gamma_{\text{spec}}=2$) and tests stochastic acceptance $r < \min(1, p(x)/q(x))$. | Proves that verification is truly parallel on GPU. |
| `test_09_p4_openai_server_and_cli` | **P4 Verification:** Verifies `StoikovInferenceEngine` OpenAI-compatible `/v1/chat/completions` and `/v1/models` endpoints plus the `stoikov-kv --help` CLI. | Executes a live chat completion request against the engine and inspects the OpenAI response schema and `stoikov_telemetry` payload. | Confirms readiness for drop-in OpenAI client and CLI workflows. |
