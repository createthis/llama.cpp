#pragma once

struct llama_model;
class llama_kv_cache;

// DeepSeek V3.2: FP8 indexer KV cache helpers
// These functions encapsulate DeepSeek-specific FP8 indexer cache behavior
// so that llama_kv_cache remains mostly architecture-agnostic.

// Initialize per-layer FP8 indexer cache sidecars for a KV cache instance.
// This inspects the model and kv cache geometry and, when enabled via
// LLAMA_FP8_INDEXER_CACHE and for DeepSeek V3.2 models, allocates GGML_TYPE_I8
// tensors attached to kv_layer entries.
void llama_init_indexer_fp8_sidecar(const llama_model & model, llama_kv_cache & kv);

