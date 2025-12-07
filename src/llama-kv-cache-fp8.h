#pragma once

#include "llama-kv-cache.h"

// Experimental FP8 KV cache for DeepSeek V3.2.
//
// This class mirrors llama_kv_cache's public API closely but stores
// K/V tensors in FP8 E4M3 (GGML_TYPE_E4M3) plus per-row scale
// tensors. It is not yet wired into llama_model; only tests will
// instantiate it for now.

class llama_kv_cache_fp8 : public llama_memory_i {
public:
    llama_kv_cache_fp8(
            const llama_model & model,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_pad,
                     uint32_t   n_swa,
               llama_swa_type   swa_type,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse);

    ~llama_kv_cache_fp8() override = default;

    // llama_memory_i
    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    // FP8-specific API (mirrors llama_kv_cache where applicable)
    uint32_t get_size()     const;
    uint32_t get_n_stream() const;

    bool get_has_shift() const;

    uint32_t get_n_kv(const llama_kv_cache::slot_info & sinfo) const;

    ggml_tensor * get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const llama_kv_cache::slot_info & sinfo) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const llama_kv_cache::slot_info & sinfo) const;

    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const llama_kv_cache::slot_info & sinfo) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const llama_kv_cache::slot_info & sinfo) const;

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const llama_kv_cache::slot_info & sinfo) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const llama_kv_cache::slot_info & sinfo) const;

    void set_input_k_shift(ggml_tensor * dst) const;
    void set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

private:
    const llama_model & model;
    const llama_hparams & hparams;
    const ggml_type type_k;
    const ggml_type type_v;

    struct kv_layer_fp8 {
        uint32_t il;

        // For DeepSeek V3.2, K-side backing store is a single 656-byte record
        // per token (fp8_ds_mla-style layout):
        //   - 512 bytes FP8 NoPE latent
        //   - 16  bytes (4 x FP32 tile scales)
        //   - 128 bytes (64 x BF16 RoPE values)
        // We store this as a raw byte tensor with shape [656, kv_size, n_stream]
        // when model.arch == LLM_ARCH_DEEPSEEK3_2. For other architectures, we
        // fall back to the older separated FP8+scale layout.
        ggml_tensor * k_blob = nullptr; // type GGML_TYPE_I8, layout [656, kv_size, n_stream]

        // Legacy FP8 K/V layout (used for non-DeepSeek V3.2 experiments):
        ggml_tensor * k_fp8   = nullptr; // type GGML_TYPE_E4M3
        ggml_tensor * v_fp8   = nullptr; // type GGML_TYPE_E4M3
        ggml_tensor * k_scale = nullptr; // [kv_size, n_stream], F32
        ggml_tensor * v_scale = nullptr; // [kv_size, n_stream], F32

        // Per-stream views for convenience
        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;
    };

    bool v_trans = true;

    const uint32_t n_seq_max = 1;
    const uint32_t n_stream  = 1;
    const uint32_t n_pad     = 1;
    const uint32_t n_swa     = 0;

    const llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;

    std::vector<ggml_context_ptr>        ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    std::vector<uint32_t> v_heads;
    std::vector<llama_kv_cells> v_cells;
    std::vector<uint32_t> seq_to_stream;

    llama_kv_cache::stream_copy_info sc_info;

    std::vector<kv_layer_fp8> layers;
    std::unordered_map<int32_t, int32_t> map_layer_ids;

    size_t total_size() const;
    size_t size_k_bytes() const;
    size_t size_v_bytes() const;

    bool is_masked_swa(llama_pos p0, llama_pos p1) const;

    ggml_tensor * build_rope_shift(
            const llama_cparams & cparams,
                   ggml_context * ctx,
                    ggml_tensor * cur,
                    ggml_tensor * shift,
                    ggml_tensor * factors,
                          float   freq_base,
                          float   freq_scale) const;

    ggml_cgraph * build_graph_shift(
               llm_graph_result * res,
                  llama_context * lctx) const;

    // Serialization helpers intentionally omitted for now; the FP8 KV
    // cache is not yet wired into state save/restore.
    void state_write_meta(llama_io_write_i & io, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count);
};
