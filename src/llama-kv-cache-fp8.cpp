#include "llama-kv-cache-fp8.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"
#include "llama-context.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

// NOTE: This is an experimental FP8 KV cache implementation intended
// to mirror the existing llama_kv_cache layout while storing K/V in
// GGML_TYPE_E4M3. It is currently only used by tests and is not wired
// into llama_model::init_mem.

llama_kv_cache_fp8::llama_kv_cache_fp8(
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
    const  layer_reuse_cb & reuse) :
    model(model), hparams(model.hparams), type_k(type_k), type_v(type_v), v_trans(v_trans),
    n_seq_max(n_seq_max), n_stream(unified ? 1u : n_seq_max),
    n_pad(n_pad), n_swa(n_swa), swa_type(swa_type) {

    GGML_ASSERT(kv_size % n_pad == 0);

    const bool is_deepseek_v32 = (model.arch == LLM_ARCH_DEEPSEEK3_2);

    const uint32_t n_layer_kv = hparams.n_layer_kv();

    // create a context for each buffer type
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(5u*(1 + n_stream)*n_layer_kv*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map[buft] = ctx;
            ctxs.emplace_back(ctx);

            return ctx;
        }

        return it->second;
    };

    GGML_ASSERT(n_stream == 1 || n_stream == n_seq_max);

    v_heads.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_heads[s] = 0;
    }

    v_cells.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].resize(kv_size);
    }

    // by default, all sequence ids are mapped to the 0th stream
    seq_to_stream.resize(LLAMA_MAX_SEQ, 0);

    if (n_stream > 1) {
        seq_to_stream.resize(n_stream, 0);
        for (uint32_t s = 0; s < n_stream; ++s) {
            seq_to_stream[s] = s;
        }
    }

    // allocate per-layer FP8 tensors

    for (uint32_t il = 0; il < hparams.n_layer; il++) {
        if (!hparams.has_kv(il)) {
            continue;
        }

        if (filter && !filter(il)) {
            continue;
        }

        const uint32_t n_embd_k_gqa =            hparams.n_embd_k_gqa(il);
        const uint32_t n_embd_v_gqa = !v_trans ? hparams.n_embd_v_gqa(il) : hparams.n_embd_v_gqa_max();

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s: [fp8] layer %3d: dev = %s\n", __func__, il, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for fp8 kv cache");
        }

        ggml_tensor * k_blob  = nullptr;
        ggml_tensor * k_fp8   = nullptr;
        ggml_tensor * v_fp8   = nullptr;
        ggml_tensor * k_scale = nullptr;
        ggml_tensor * v_scale = nullptr;

        // Backing storage:
        //  - For DeepSeek V3.2 (sparse MLA), use a single 656-byte record per
        //    token for K-side (fp8_ds_mla-style layout) stored as raw bytes.
        //  - For other architectures, fall back to FP8 E4M3 with per-row F32
        //    scales for both K and V, as in the original experimental design.
        GGML_UNUSED(type_k);
        GGML_UNUSED(type_v);

        if (is_deepseek_v32) {
            // K-side FP8 ds_mla blob: [656, kv_size, n_stream] bytes
            const int64_t entry_bytes = 656;
            k_blob = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, entry_bytes, kv_size, n_stream);
            ggml_format_name(k_blob, "cache_k_fp8_blob_l%d", il);
        } else {
            // Legacy per-element FP8 + per-row scale layout
            k_fp8 = ggml_new_tensor_3d(ctx, GGML_TYPE_E4M3, n_embd_k_gqa, kv_size, n_stream);
            v_fp8 = ggml_new_tensor_3d(ctx, GGML_TYPE_E4M3, n_embd_v_gqa, kv_size, n_stream);
            k_scale = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kv_size, n_stream);
            v_scale = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kv_size, n_stream);

            ggml_format_name(k_fp8,   "cache_k_fp8_l%d",      il);
            ggml_format_name(v_fp8,   "cache_v_fp8_l%d",      il);
            ggml_format_name(k_scale, "cache_k_scale_l%d",    il);
            ggml_format_name(v_scale, "cache_v_scale_l%d",    il);
        }

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;

        if (is_deepseek_v32) {
            // For DeepSeek V3.2, create per-stream 2D views over [entry_bytes, kv_size]
            for (uint32_t s = 0; s < n_stream; ++s) {
                k_stream.push_back(ggml_view_2d(ctx, k_blob,
                                                k_blob->ne[0], k_blob->ne[1],
                                                k_blob->nb[1], s * k_blob->nb[2]));
            }
        } else {
            for (uint32_t s = 0; s < n_stream; ++s) {
                k_stream.push_back(ggml_view_2d(ctx, k_fp8, n_embd_k_gqa, kv_size,
                                                k_fp8->nb[1], s * k_fp8->nb[2]));
                v_stream.push_back(ggml_view_2d(ctx, v_fp8, n_embd_v_gqa, kv_size,
                                                v_fp8->nb[1], s * v_fp8->nb[2]));
            }
        }

        map_layer_ids[il] = layers.size();

        layers.push_back({});
        auto & lyr = layers.back();
        lyr.il      = il;
        lyr.k_blob  = k_blob;
        lyr.k_fp8   = k_fp8;
        lyr.v_fp8   = v_fp8;
        lyr.k_scale = k_scale;
        lyr.v_scale = v_scale;
        lyr.k_stream = std::move(k_stream);
        lyr.v_stream = std::move(v_stream);
    }

    if (reuse) {
        for (uint32_t il = 0; il < hparams.n_layer; il++) {
            const int32_t il_reuse = reuse(il);

            if (il_reuse < 0) {
                continue;
            }

            if (filter && !filter(il)) {
                continue;
            }

            GGML_ASSERT(map_layer_ids.find(il_reuse) != map_layer_ids.end());

            map_layer_ids[il] = map_layer_ids[il_reuse];
        }
    }

    // allocate tensors and initialize buffers
    for (auto it : ctx_map) {
        auto * buft = it.first;
        auto * ctx  = it.second;

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for fp8 kv cache");
        }

        LLAMA_LOG_INFO("%s: [fp8] %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        ggml_backend_buffer_clear(buf, 0);
        bufs.emplace_back(buf);
    }

    {
        const size_t memory_size_k = size_k_bytes();
        const size_t memory_size_v = size_v_bytes();

        LLAMA_LOG_INFO("%s: [fp8] size = %7.2f MiB (%6u cells, %3d layers, %2u/%u seqs), K (fp8): %7.2f MiB, V (fp8): %7.2f MiB\n", __func__,
                (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f), kv_size, (int) layers.size(), n_seq_max, n_stream,
                (float)memory_size_k / (1024.0f * 1024.0f),
                (float)memory_size_v / (1024.0f * 1024.0f));
    }
}

bool llama_kv_cache_fp8::get_can_shift() const {
    return true;
}

uint32_t llama_kv_cache_fp8::get_size() const {
    const auto & cells = v_cells[seq_to_stream[0]];
    return cells.size();
}

uint32_t llama_kv_cache_fp8::get_n_stream() const {
    return n_stream;
}

bool llama_kv_cache_fp8::get_has_shift() const {
    bool result = false;
    for (uint32_t s = 0; s < n_stream; ++s) {
        result |= v_cells[s].get_has_shift();
    }
    return result;
}

uint32_t llama_kv_cache_fp8::get_n_kv(const llama_kv_cache::slot_info & sinfo) const {
    uint32_t result = 0;
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const auto & cells = v_cells[sinfo.strm[s]];
        result = std::max(std::min(cells.size(), std::max(n_pad, GGML_PAD(cells.used_max_p1(), n_pad))), result);
    }
    return result;
}

// Helpers to quantize/dequantize rows using ggml FP8 helpers

extern "C" {
    struct ggml_e4m3_t;
    void ggml_e4m3_to_fp32_row(const ggml_e4m3_t * x, float * y, int64_t k);
    void ggml_fp32_to_e4m3_row_ref(const float * x, ggml_e4m3_t * y, int64_t k);
}

static void fp32_to_e4m3_row(const float * src, ggml_e4m3_t * dst, int64_t k) {
    ggml_fp32_to_e4m3_row_ref(src, dst, k);
}

static void e4m3_to_fp32_row(const ggml_e4m3_t * src, float * dst, int64_t k) {
    ggml_e4m3_to_fp32_row(src, dst, k);
}

// Clear / seq_* / state_* follow the patterns of llama_kv_cache but
// operate on v_cells/v_heads only. For brevity we reuse the same
// logic by delegating where possible.

void llama_kv_cache_fp8::clear(bool data) {
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].reset();
        v_heads[s] = 0;
    }

    if (data) {
        for (auto & buf : bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }
}

bool llama_kv_cache_fp8::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if (seq_id >= 0) {
        auto & cells = v_cells[seq_to_stream[seq_id]];
        auto & head  = v_heads[seq_to_stream[seq_id]];

        uint32_t new_head = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }

        if (new_head != cells.size() && new_head < head) {
            head = new_head;
        }
    } else {
        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];
            auto & head  = v_heads[s];

            uint32_t new_head = cells.size();

            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (!cells.pos_in(i, p0, p1)) {
                    continue;
                }

                cells.rm(i);

                if (new_head == cells.size()) {
                    new_head = i;
                }
            }

            if (new_head != cells.size() && new_head < head) {
                head = new_head;
            }
        }
    }

    return true;
}

void llama_kv_cache_fp8::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    GGML_ASSERT(seq_id_src >= 0 && (size_t) seq_id_src < seq_to_stream.size());
    GGML_ASSERT(seq_id_dst >= 0 && (size_t) seq_id_dst < seq_to_stream.size());

    const auto s0 = seq_to_stream[seq_id_src];
    const auto s1 = seq_to_stream[seq_id_dst];

    if (s0 == s1) {
        auto & cells = v_cells[s0];

        if (seq_id_src == seq_id_dst) {
            return;
        }

        if (p0 < 0) {
            p0 = 0;
        }

        if (p1 < 0) {
            p1 = std::numeric_limits<llama_pos>::max();
        }

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id_src)) {
                cells.seq_add(i, seq_id_dst);
            }
        }

        return;
    }

    bool is_full = true;

    if (p0 > 0 && p0 + 1 < (int) get_size()) {
        is_full = false;
    }

    if (p1 > 0 && p1 + 1 < (int) get_size()) {
        is_full = false;
    }

    GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers (fp8)");

    sc_info.ssrc.push_back(s0);
    sc_info.sdst.push_back(s1);

    v_cells[s1].reset();
    for (uint32_t i = 0; i < v_cells[s0].size(); ++i) {
        if (v_cells[s0].seq_has(i, seq_id_src)) {
            llama_pos pos   = v_cells[s0].pos_get(i);
            llama_pos shift = v_cells[s0].get_shift(i);

            if (shift != 0) {
                pos -= shift;
                assert(pos >= 0);
            }

            v_cells[s1].pos_set(i, pos);
            v_cells[s1].seq_add(i, seq_id_dst);

            if (shift != 0) {
                v_cells[s1].pos_add(i, shift);
            }
        }
    }

    v_heads[s1] = v_heads[s0];
}

void llama_kv_cache_fp8::seq_keep(llama_seq_id seq_id) {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    uint32_t new_head = cells.size();

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.seq_keep(i, seq_id)) {
            if (new_head == cells.size()) {
                new_head = i;
            }
        }
    }

    if (new_head != cells.size() && new_head < head) {
        head = new_head;
    }
}

void llama_kv_cache_fp8::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    auto & cells = v_cells[seq_to_stream[seq_id]];

    if (shift == 0) {
        return;
    }

    uint32_t new_head = cells.size();

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            if (cells.pos_add(i, shift)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }
    }

    // we don't maintain a head per-stream here, but preserve behavior by
    // leaving v_heads unchanged; this is fine for FP8 cache experiments.
}

void llama_kv_cache_fp8::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    auto & cells = v_cells[seq_to_stream[seq_id]];

    if (d == 1) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            cells.pos_div(i, d);
        }
    }
}

llama_pos llama_kv_cache_fp8::seq_pos_min(llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    const auto & cells = v_cells[seq_to_stream[seq_id]];
    return cells.seq_pos_min(seq_id);
}

llama_pos llama_kv_cache_fp8::seq_pos_max(llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    const auto & cells = v_cells[seq_to_stream[seq_id]];
    return cells.seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_fp8::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> result;

    for (const auto & buf : bufs) {
        result[ggml_backend_buffer_get_type(buf.get())] += ggml_backend_buffer_get_size(buf.get());
    }

    return result;
}

size_t llama_kv_cache_fp8::total_size() const {
    size_t size = 0;
    for (const auto & buf : bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }
    return size;
}

size_t llama_kv_cache_fp8::size_k_bytes() const {
    size_t size_k_bytes = 0;
    for (const auto & layer : layers) {
        if (layer.k_blob != nullptr) {
            size_k_bytes += ggml_nbytes(layer.k_blob);
        } else if (layer.k_fp8 != nullptr) {
            size_k_bytes += ggml_nbytes(layer.k_fp8);
            if (layer.k_scale != nullptr) {
                size_k_bytes += ggml_nbytes(layer.k_scale);
            }
        }
    }
    return size_k_bytes;
}

size_t llama_kv_cache_fp8::size_v_bytes() const {
    size_t size_v_bytes = 0;
    for (const auto & layer : layers) {
        size_v_bytes += ggml_nbytes(layer.v_fp8) + ggml_nbytes(layer.v_scale);
    }
    return size_v_bytes;
}

bool llama_kv_cache_fp8::is_masked_swa(llama_pos p0, llama_pos p1) const {
    return llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1);
}

// Graph-related methods (init_batch/init_full/init_update, apply_ubatch,
// get_k/get_v/cpy_k/cpy_v, mask building, and state read/write) are
// intentionally left unimplemented for now. This FP8 KV cache is
// currently a storage-side experiment and not yet wired into llama.cpp
// graphs. Once DeepSeek V3.2 is switched over, we can mirror the
// corresponding logic from llama_kv_cache here.

llama_memory_context_ptr llama_kv_cache_fp8::init_batch(
        llama_batch_allocr & balloc,
        uint32_t n_ubatch,
        bool embd_all) {
    GGML_UNUSED(balloc);
    GGML_UNUSED(n_ubatch);
    GGML_UNUSED(embd_all);
    return std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache_fp8::init_full() {
    return std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache_fp8::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(lctx);
    GGML_UNUSED(optimize);
    return std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

void llama_kv_cache_fp8::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    GGML_UNUSED(io);
    GGML_UNUSED(seq_id);
    GGML_UNUSED(flags);
}

void llama_kv_cache_fp8::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(io);
    GGML_UNUSED(seq_id);
    GGML_UNUSED(flags);
}

// Stubbed graph-building helpers

ggml_tensor * llama_kv_cache_fp8::build_rope_shift(
        const llama_cparams & cparams,
               ggml_context * ctx,
                ggml_tensor * cur,
                ggml_tensor * shift,
                ggml_tensor * factors,
                      float   freq_base,
                      float   freq_scale) const {
    GGML_UNUSED(cparams);
    GGML_UNUSED(ctx);
    GGML_UNUSED(cur);
    GGML_UNUSED(shift);
    GGML_UNUSED(factors);
    GGML_UNUSED(freq_base);
    GGML_UNUSED(freq_scale);
    return nullptr;
}

ggml_cgraph * llama_kv_cache_fp8::build_graph_shift(
           llm_graph_result * res,
              llama_context * lctx) const {
    GGML_UNUSED(res);
    GGML_UNUSED(lctx);
    return nullptr;
}

void llama_kv_cache_fp8::state_write_meta(llama_io_write_i & io, llama_seq_id seq_id) const {
    GGML_UNUSED(io);
    GGML_UNUSED(seq_id);
}

void llama_kv_cache_fp8::state_write_data(llama_io_write_i & io) const {
    GGML_UNUSED(io);
}

bool llama_kv_cache_fp8::state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, llama_seq_id dest_seq_id) {
    GGML_UNUSED(io);
    GGML_UNUSED(strm);
    GGML_UNUSED(cell_count);
    GGML_UNUSED(dest_seq_id);
    return false;
}

bool llama_kv_cache_fp8::state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count) {
    GGML_UNUSED(io);
    GGML_UNUSED(strm);
    GGML_UNUSED(cell_count);
    return false;
}

// Accessors for K/V are left unimplemented for now since the FP8 cache
// is not yet used in any graph. They will be filled in when wiring the
// cache to DeepSeek V3.2.

ggml_tensor * llama_kv_cache_fp8::get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const llama_kv_cache::slot_info & sinfo) const {
    GGML_UNUSED(ctx);
    GGML_UNUSED(il);
    GGML_UNUSED(n_kv);
    GGML_UNUSED(sinfo);
    return nullptr;
}

ggml_tensor * llama_kv_cache_fp8::get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const llama_kv_cache::slot_info & sinfo) const {
    GGML_UNUSED(ctx);
    GGML_UNUSED(il);
    GGML_UNUSED(n_kv);
    GGML_UNUSED(sinfo);
    return nullptr;
}

ggml_tensor * llama_kv_cache_fp8::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const llama_kv_cache::slot_info & sinfo) const {
    GGML_UNUSED(ctx);
    GGML_UNUSED(k_cur);
    GGML_UNUSED(k_idxs);
    GGML_UNUSED(il);
    GGML_UNUSED(sinfo);
    return nullptr;
}

ggml_tensor * llama_kv_cache_fp8::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const llama_kv_cache::slot_info & sinfo) const {
    GGML_UNUSED(ctx);
    GGML_UNUSED(v_cur);
    GGML_UNUSED(v_idxs);
    GGML_UNUSED(il);
    GGML_UNUSED(sinfo);
    return nullptr;
}

ggml_tensor * llama_kv_cache_fp8::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    GGML_UNUSED(ctx);
    GGML_UNUSED(ubatch);
    return nullptr;
}

ggml_tensor * llama_kv_cache_fp8::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    GGML_UNUSED(ctx);
    GGML_UNUSED(ubatch);
    return nullptr;
}

void llama_kv_cache_fp8::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const llama_kv_cache::slot_info & sinfo) const {
    GGML_UNUSED(dst);
    GGML_UNUSED(ubatch);
    GGML_UNUSED(sinfo);
}

void llama_kv_cache_fp8::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const llama_kv_cache::slot_info & sinfo) const {
    GGML_UNUSED(dst);
    GGML_UNUSED(ubatch);
    GGML_UNUSED(sinfo);
}

void llama_kv_cache_fp8::set_input_k_shift(ggml_tensor * dst) const {
    GGML_UNUSED(dst);
}

void llama_kv_cache_fp8::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    GGML_UNUSED(dst);
    GGML_UNUSED(ubatch);
    GGML_UNUSED(causal_attn);
}

void llama_kv_cache_fp8::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    GGML_UNUSED(dst);
    GGML_UNUSED(ubatch);
}
