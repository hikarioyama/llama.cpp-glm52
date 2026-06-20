#include "models.h"

void llama_model_glm_dsa::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,     hparams.n_ff_exp);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,    hparams.f_norm_rms_eps);
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, false);

    // MoE parameters
    ml.get_key(LLM_KV_EXPERT_COUNT,                hparams.n_expert);
    ml.get_key(LLM_KV_EXPERT_USED_COUNT,           hparams.n_expert_used);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);

    // deepseek MLA parameters
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,      hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,     hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   hparams.n_embd_head_k_mla_impl, false);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, hparams.n_embd_head_v_mla_impl, false);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,        hparams.n_expert_shared);

    // DSA parameters
    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);

    // Expert gating function (GLM-4.5 uses sigmoid)
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);
    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        hparams.expert_gating_func =  LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
    }

    // NextN/MTP parameters
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS,        hparams.nextn_predict_layers, false);
    GGML_ASSERT(hparams.nextn_predict_layers < hparams.n_layer && "nextn_predict_layers must be < n_layer");

    // TODO: when MTP is implemented, this should probably be updated if needed
    // when MTP loading is enabled, give the nextn layer(s) a KV slot so build_attn(il=78)
    // can resolve map_layer_ids; the MTP-off path stays numerically identical.
    hparams.n_layer_kv_from_start = params.enable_mtp ? hparams.n_layer
                                                      : hparams.n_layer - hparams.nextn_predict_layers;

    switch (hparams.n_layer) {
        case 79: type = LLM_TYPE_744B_A40B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_glm_dsa::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;
    const int64_t n_expert_shared = hparams.n_expert_shared;

    const bool is_mla = hparams.is_mla();
    if (!is_mla) {
        throw std::runtime_error("GLM_DSA architecture requires MLA");
    }

    // note: these are the actual head sizes you get when treating as MHA or after "decompression" using wv_b for MLA
    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();

    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;

    const int64_t q_lora_rank  = hparams.n_lora_q;
    const int64_t kv_lora_rank = hparams.n_lora_kv;

    const int64_t n_ff_exp        = hparams.n_ff_exp;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    // try to load output.weight, if not found, use token_embd (tied embeddings)
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (!output) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        // skip the NextN/MTP layer(s) unless MTP loading is explicitly enabled
        const bool skip_nextn = !params.enable_mtp && hparams.nextn_predict_layers > 0 &&
            static_cast<uint32_t>(i) >= n_layer - hparams.nextn_predict_layers;

        int flags = 0;
        if (skip_nextn) {
            // skip all tensors in the NextN layers
            // TODO @ngxson : TENSOR_NOT_REQUIRED was a hack, need to remove it later
            flags |= TENSOR_SKIP | TENSOR_NOT_REQUIRED;
        }

        auto & layer = layers[i];

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, flags);
        layer.attn_q_a_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, flags);
        layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, flags);

        layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", i), {n_embd, q_lora_rank}, flags);
        layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", i), {q_lora_rank, n_head * n_embd_head_k_mla}, flags);

        layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), {n_embd, kv_lora_rank + n_embd_head_qk_rope}, flags);

        // note: only old legacy GGUF files will have the unsplit wkv_b tensor in
        layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", i), {n_embd_head_qk_nope, kv_lora_rank, n_head}, flags);
        layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", i), {kv_lora_rank, n_embd_head_v_mla, n_head}, flags);

        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head * n_embd_head_v_mla, n_embd}, flags);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, flags);

        // DSA indexer
        layer.indexer_k_norm   = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "weight", i), {hparams.indexer_head_size}, flags);
        layer.indexer_k_norm_b = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "bias",   i), {hparams.indexer_head_size}, flags);
        layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, hparams.indexer_n_head}, flags);
        layer.indexer_attn_k   = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K,   "weight", i), {n_embd, hparams.indexer_head_size}, flags);
        layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, hparams.indexer_n_head * hparams.indexer_head_size}, flags);
        if (i < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, flags);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, flags);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, flags);
        } else {
            layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, flags);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED);

            if (n_expert == 0) {
                throw std::runtime_error("n_expert must be > 0");
            }
            if (n_expert_used == 0) {
                throw std::runtime_error("n_expert_used must be > 0");
            }

            // MoE branch
            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, flags);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, flags);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, flags);

            // Shared expert branch
            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, flags);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {        n_ff_exp * n_expert_shared, n_embd}, flags);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, flags);
        }

        // NextN/MTP tensors (preserved but unused) - conditionally load for last nextn_predict_layers
        if (hparams.nextn_predict_layers > 0 && static_cast<uint32_t>(i) >= n_layer - hparams.nextn_predict_layers) {
            layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", i), { 2 * n_embd, n_embd }, flags);
            layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM, "weight", i), { n_embd }, flags);
            layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM, "weight", i), { n_embd }, flags);

            // Optional tensors
            layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS, "weight", i), { n_embd, n_vocab }, flags | TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i), { n_embd, n_vocab }, flags | TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), { n_embd }, flags | TENSOR_NOT_REQUIRED);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_glm_dsa::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_glm_dsa::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    // glm_dsa is always MLA and never OCR (plain MLA path; the DSA indexer tensors are loaded
    // but unused by this graph).
    const bool is_mla = hparams.is_mla();
    GGML_ASSERT(is_mla);

    // note: these are the actual head sizes you get when treating as MHA or after "decompression" using wv_b for MLA
    const int64_t n_embd_head_k = hparams.n_embd_head_k_mla();

    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;

    const uint32_t kv_lora_rank = hparams.n_lora_kv;

    // We have to pre-scale kq_scale and attn_factor to make the YaRN RoPE work correctly.
    // See https://github.com/ggml-org/llama.cpp/discussions/7416 for detailed explanation.
    GGML_ASSERT(ext_factor >= 0.0f);
    const float attn_factor_org = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));

    const float mscale   = attn_factor_org * (1.0f + 0.1f * hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
    const float kq_scale = 1.0f * mscale * mscale / sqrtf(float(n_embd_head_k));

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // {n_embd, n_tokens}
    inpL = build_inp_embd(model.tok_embd);

    // keep a handle to the original input token embeddings for the nextn probe's shift-by-one view
    ggml_tensor * tok_emb_all = inpL;

    // (optional) temperature tuning - used by mistral-large
    ggml_tensor * inp_attn_scale = nullptr;
    if (hparams.f_attn_temp_scale != 0.0f) {
        inp_attn_scale = build_inp_attn_scale();
    }

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn_k = build_attn_inp_k();

    // MTP draft-only forward (selected via llama_mtp_set_draft_mode): skip the main layers and run
    // only the nextn block (see the early branch below). It operates on all positions, so it has no
    // out_ids reduction — and building an UNUSED out_ids input would leave its tensor unallocated and
    // crash in set_input (ggml_backend_buffer_is_host on a null buffer). So skip it in draft mode.
    const bool mtp_draft = llama_mtp_get_draft_mode() && model.layers[hparams.n_layer - 1].nextn.eh_proj;
    ggml_tensor * inp_out_ids = mtp_draft ? nullptr : build_inp_out_ids();

    // per-layer transformer body (deepseek2 MLA + MoE). Factored into a lambda so the nextn
    // (MTP) block can re-run it with the layer-78 weights. apply_out_ids is true ONLY for the
    // last main layer; the nextn call passes false so it operates on all positions.
    auto run_layer = [&](ggml_tensor * inpL_in, int il, bool apply_out_ids, bool skip_attn = false) -> ggml_tensor * {
        ggml_tensor * inpSA = inpL_in;
        ggml_tensor * cur   = inpSA;
        ggml_tensor * ffn_inp;

      if (skip_attn) {
        // MTP depth-2 chained draft (see the mtp_spec branch below): skip this block's MLA entirely
        // so it performs NO KV write. The layer-78 KV for committed tokens — written by the depth-1
        // nextn pass — therefore stays uncorrupted, preserving the depth-1 acceptance α exactly. The
        // FFN/MoE sublayer alone refines the injected hidden (= the depth-1 nextn block output) plus
        // emb(d1) into the depth-2 draft. Self-spec is lossless regardless of draft quality (the
        // verify forward's main logits are ground truth), so dropping attention here only trades
        // depth-2 draft α — never the emitted token stream.
        ffn_inp = inpSA;
      } else {
        // norm
        cur = build_norm(inpL_in, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self_attention (MLA with absorption)
        {
            ggml_tensor * q = NULL;

            // glm_dsa always has wq_a/wq_b (not lite)
            q = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
            cb(q, "q", il);

            q = build_norm(q, model.layers[il].attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(q, "q", il);

            q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
            cb(q, "q", il);

            // split into {n_embd_head_qk_nope, n_head, n_tokens}
            ggml_tensor * q_nope =
                ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
                             ggml_row_size(q->type, n_embd_head_k) * n_head, 0);
            cb(q_nope, "q_nope", il);

            // and {n_embd_head_qk_rope, n_head, n_tokens}
            ggml_tensor * q_pe = ggml_view_3d(
                ctx0, q, n_embd_head_qk_rope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
                ggml_row_size(q->type, n_embd_head_k) * n_head, ggml_row_size(q->type, n_embd_head_qk_nope));
            cb(q_pe, "q_pe", il);

            ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
            cb(kv_cmpr_pe, "kv_cmpr_pe", il);

            // split into {kv_lora_rank, n_tokens}
            ggml_tensor * kv_cmpr =
                ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                             ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
            cb(kv_cmpr, "kv_cmpr", il);

            // and {n_embd_head_qk_rope, 1, n_tokens}
            ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
            cb(k_pe, "k_pe", il);

            q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(q_pe, "q_pe", il);

            k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(k_pe, "k_pe", il);

            kv_cmpr = build_norm(kv_cmpr, model.layers[il].attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(kv_cmpr, "kv_cmpr", il);

            // {n_embd_head_qk_nope, n_tokens, n_head}
            q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
            cb(q_nope, "q_nope_perm", il);

            // {n_embd_head_qk_nope, kv_lora_rank, n_head} x {n_embd_head_qk_nope, n_tokens, n_head}
            ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, model.layers[il].wk_b, q_nope);
            cb(q_nope_absorbed, "q_nope_absorbed", il);

            // {kv_lora_rank, n_head, n_tokens}
            q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
            cb(q_nope_absorbed, "q_nope_absorbed_perm", il);

            // {n_embd_head_qk_rope + kv_lora_rank, n_head, n_tokens}
            // note: rope must go first for in-place context shifting in build_rope_shift()
            ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
            cb(Qcur, "Qcur", il);

            kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
            cb(kv_cmpr, "kv_cmpr_reshape", il);

            // {n_embd_head_qk_rope + kv_lora_rank, 1, n_tokens}
            ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
            cb(Kcur, "Kcur", il);

            // {kv_lora_rank, 1, n_tokens}
            ggml_tensor * Vcur = kv_cmpr;
            cb(Vcur, "Vcur", il);

            if (inp_attn_scale) {
                // apply llama 4 temperature scaling
                Qcur = ggml_mul(ctx0, Qcur, inp_attn_scale);
                cb(Qcur, "Qcur_attn_temp_scaled", il);
            }

            // note: MLA with the absorption optimization converts into MQA (ie: GQA with 1 group)
            cur = build_attn(inp_attn_k,
                    model.layers[il].wo, NULL, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, model.layers[il].wv_b, kq_scale, il);
        }

        if (apply_out_ids && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ffn_inp = ggml_add(ctx0, cur, inpSA);
      } // end else (skip_attn)
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                model.layers[il].ffn_up, NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE branch
            ggml_tensor * moe_out = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                model.layers[il].ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il,
                nullptr,
                model.layers[il].ffn_gate_up_exps);
            cb(moe_out, "ffn_moe_out", il);

            // FFN shared expert
            {
                ggml_tensor * ffn_shexp =
                    build_ffn(cur,
                        model.layers[il].ffn_up_shexp, NULL, NULL,
                        model.layers[il].ffn_gate_shexp, NULL, NULL,
                        model.layers[il].ffn_down_shexp, NULL, NULL,
                        NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(ffn_shexp, "ffn_shexp", il);

                cur = ggml_add(ctx0, moe_out, ffn_shexp);
                cb(cur, "ffn_out", il);
            }
        }
        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        return cur;
    };

    const int il_nextn = hparams.n_layer - 1; // 78 (the single nextn/MTP block)

    // ---- MTP draft-only forward (self-spec cheap draft) -----------------------------------------
    // Skip the main layers (0..77) entirely and run ONLY the nextn block on a host-injected
    // pre-norm hidden. The draft for "the token after t" = nextn(h_t, emb(t)); the caller submits
    // exactly the token t whose successor it wants drafted (so NO left-shift here) and injects h_t.
    // This reads only the layer-78 experts (~92 MB) instead of the full ~1.75 GB offloaded set, so
    // it is the cheap half of the K=1 self-spec loop. Selected by llama_mtp_set_draft_mode(true);
    // graph_variant keeps this topology out of the normal-forward reuse cache.
    if (mtp_draft) {
        ggml_tensor * emb_in = tok_emb_all;            // {n_embd, N} = emb(submitted tokens)
        ggml_tensor * h      = build_inp_mtp_hidden();  // {n_embd, N} = injected pre-norm hidden h_t
        cb(h, "nextn_h_prenorm", il_nextn);

        ggml_tensor * e  = build_norm(emb_in, model.layers[il_nextn].nextn.enorm, NULL, LLM_NORM_RMS, -1);
        ggml_tensor * hn = build_norm(h,      model.layers[il_nextn].nextn.hnorm, NULL, LLM_NORM_RMS, -1);
        ggml_tensor * x  = ggml_mul_mat(ctx0, model.layers[il_nextn].nextn.eh_proj, ggml_concat(ctx0, e, hn, 0)); // {n_embd, N}
        cb(x, "nextn_x", il_nextn);

        ggml_tensor * blk = run_layer(x, il_nextn, /*apply_out_ids=*/ false); // layer-78 MLA(+KV)+MoE
        ggml_tensor * o   = build_norm(blk, model.layers[il_nextn].nextn.shared_head_norm, NULL, LLM_NORM_RMS, -1);
        ggml_tensor * draft = ggml_mul_mat(ctx0, model.output, o); // tied head -> {n_vocab, N}
        cb(draft, "nextn_draft", il_nextn);

        res->t_embd   = o;     // non-null placeholder (unused; embeddings are not requested here)
        res->t_logits = draft; // the draft logits ARE the output of this forward
        ggml_build_forward_expand(gf, draft);
        return;
    }

    int effective_n_layers = hparams.n_layer - hparams.nextn_predict_layers;
    for (int il = 0; il < effective_n_layers; ++il) {
        inpL = run_layer(inpL, il, /*apply_out_ids=*/ il == effective_n_layers - 1);
    }

    // pre-output-norm hidden at every position (used by the nextn probe; do NOT use res->t_embd which is post-norm)
    ggml_tensor * h_prenorm = inpL;

    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = ggml_mul_mat(ctx0, model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    // nextn (MTP) head — pure measurement add, env-gated so the base graph is byte-identical
    // when neither LLAMA_MTP_PROBE nor LLAMA_MTP_SPEC is set. Requires --enable-mtp (nextn tensors
    // loaded and layer-78 KV allocated via the hparams gate above).
    //   * LLAMA_MTP_PROBE  : OVERWRITE res->t_logits with the draft logits (offline α-probe; the
    //                        target logits are not needed in that mode).
    //   * LLAMA_MTP_SPEC   : keep res->t_logits = the TARGET logits (verify needs them) and only
    //                        expose the draft logits as the named tensor "nextn_draft-78", which the
    //                        self-spec loop reads back via cb_eval. Used by the K=1 self-spec decode.
    const bool mtp_probe = getenv("LLAMA_MTP_PROBE") != nullptr;
    const bool mtp_spec  = getenv("LLAMA_MTP_SPEC")  != nullptr;
    // (il_nextn declared above)
    // Engage the nextn branch ONLY on all-positions-output forwards, where inp_out_ids is the
    // identity so h_prenorm keeps its full n_tokens width and matches emb_shift (built from
    // tok_emb_all = n_tokens wide). On a partial-output forward (e.g. the prompt prefill: seqlen=15,
    // only the last logits=true) inp_out_ids reduces h_prenorm to {n_embd, n_outputs}, and the
    // nextn ggml_concat(e,hn) would assert on the ne[1] mismatch. The prompt prefill needs no draft,
    // so skipping it there is correct. (The probe always runs an all-logits prefill, so the guard is
    // a no-op for it; it gates only the self-spec partial-output forwards.)
    if ((mtp_probe || mtp_spec) && model.layers[il_nextn].nextn.eh_proj
            && h_prenorm->ne[1] == tok_emb_all->ne[1]) {
        const int64_t N = tok_emb_all->ne[1];
        const size_t  s = tok_emb_all->nb[1]; // column (dim1) stride

        // The nextn input emb(t_{i+1}) — TWO sources depending on mode:
        ggml_tensor * emb_in;
        if (mtp_spec) {
            // IN-GRAPH DRAFT (the speedup): emb at column c = embedding of the model's OWN argmax of
            // the target logits at column c = emb(the true next token after position c). Then
            //   col0 nextn = nextn(h_a, emb(v))  = the MISS draft (guess succ(v), next processes v)
            //   col1 nextn = nextn(h_d, emb(w))  = the HIT  draft (guess succ(w), next processes w)
            // i.e. the fused VERIFY forward ALSO produces the next cycle's drafts -> NO separate
            // cheap-draft forward (saves its per-call fixed overhead) AND the graph never switches
            // topology (no reuse thrash). This is the faithful probe setup (emb of the real next
            // token), so acceptance should match the probe (~0.72), not the input-shift approximation.
            ggml_tensor * ids = ggml_argmax(ctx0, res->t_logits);  // {N} I32, argmax over vocab per col
            emb_in = ggml_get_rows(ctx0, model.tok_embd, ids);     // {n_embd, N}
        } else {
            // probe: emb input = input tokens shifted LEFT by one column (emb of the actual next
            // token in S). Keep width N (pad last col with col N-1; the harness ignores it).
            if (N <= 1) {
                emb_in = tok_emb_all; // {n_embd, 1}
            } else {
                ggml_tensor * drop0 = ggml_view_2d(ctx0, tok_emb_all, n_embd, N - 1, s, s);          // cols 1..N-1
                ggml_tensor * lastc = ggml_view_2d(ctx0, tok_emb_all, n_embd, 1,     s, (N - 1) * s); // col N-1 (junk pad)
                emb_in = ggml_concat(ctx0, drop0, lastc, 1);                                          // {n_embd, N}
            }
        }
        cb(emb_in, "nextn_emb_shift", il_nextn); // raw input (for offline math cross-check)

        ggml_tensor * h = h_prenorm; // {n_embd, N}, used as-is so ne[1]==N matches emb_in
        cb(h, "nextn_h_prenorm", il_nextn);         // raw input (for offline math cross-check)

        // 1. e  = enorm(emb(t_{i+1}))
        ggml_tensor * e = build_norm(emb_in, model.layers[il_nextn].nextn.enorm, NULL, LLM_NORM_RMS, -1);
        // 2. hn = hnorm(h_i)
        ggml_tensor * hn = build_norm(h, model.layers[il_nextn].nextn.hnorm, NULL, LLM_NORM_RMS, -1);
        // 3. x  = eh_proj · concat(e, hn)   (concat order: e THEN hn, along dim0)
        ggml_tensor * ehcat = ggml_concat(ctx0, e, hn, 0); // {2*n_embd, N}
        cb(ehcat, "nextn_ehcat", il_nextn); // eh_proj input (offline math cross-check: norms + concat order)
        ggml_tensor * x = ggml_mul_mat(ctx0, model.layers[il_nextn].nextn.eh_proj, ehcat); // {n_embd, N}
        cb(x, "nextn_x", il_nextn);

        // 4. one full layer-78 transformer block (MLA + MoE) with the nextn weights; all positions
        ggml_tensor * blk = run_layer(x, il_nextn, /*apply_out_ids=*/ false);
        cb(blk, "nextn_block", il_nextn);

        // 5. o = shared_head_norm(block_out)   (nextn's OWN norm, NOT output_norm)
        ggml_tensor * o = build_norm(blk, model.layers[il_nextn].nextn.shared_head_norm, NULL, LLM_NORM_RMS, -1);
        cb(o, "nextn_o", il_nextn);                 // head input (for offline math cross-check)

        // 6. draft_logits = output · o   (head tied to model.output)
        ggml_tensor * draft = ggml_mul_mat(ctx0, model.output, o); // {n_vocab, N}
        cb(draft, "nextn_draft", il_nextn);

        if (mtp_spec) {
            // self-spec: leave res->t_logits = the TARGET logits untouched (verify reads them via
            // llama_get_logits_ith). The draft logits are read back through cb_eval by name, so they
            // only need to be part of the compute graph — expand them explicitly.
            ggml_build_forward_expand(gf, draft); // depth-1 draft = "nextn_draft-78"

          // LLAMA_MTP_NO_STEP2: isolate the per-forward cost of the 3rd verify column from the cost
          // of the depth-2 nextn traversal. When set, skip building step2 entirely (graph keeps the
          // 3 columns + single nextn pass) so a spec run measures the 3-col, depth-1-only per-forward.
          if (getenv("LLAMA_MTP_NO_STEP2") == nullptr) {
            // ---- depth-2 chained draft (K=2): one more nextn step, weight-tied to the same layer-78
            // module, reusing the depth-1 block output `blk` as the hidden and emb(argmax(depth-1
            // draft)) as the token embedding. run_layer is called with skip_attn=true so this step
            // writes NO layer-78 KV — the depth-1 pass's committed-token KV stays clean (preserves
            // α1). Per column c the chain seeds from succ(token@c): col0→miss path, col1→1-hit path,
            // col2→2-hit path. Lossless regardless: verify catches every wrong draft.
            ggml_tensor * ids2  = ggml_argmax(ctx0, draft);                         // {N} I32
            ggml_tensor * emb2  = ggml_get_rows(ctx0, model.tok_embd, ids2);        // {n_embd, N}
            ggml_tensor * e2    = build_norm(emb2, model.layers[il_nextn].nextn.enorm, NULL, LLM_NORM_RMS, -1);
            ggml_tensor * hn2   = build_norm(blk,  model.layers[il_nextn].nextn.hnorm, NULL, LLM_NORM_RMS, -1);
            ggml_tensor * x2    = ggml_mul_mat(ctx0, model.layers[il_nextn].nextn.eh_proj, ggml_concat(ctx0, e2, hn2, 0)); // {n_embd, N}
            ggml_tensor * blk2  = run_layer(x2, il_nextn, /*apply_out_ids=*/ false, /*skip_attn=*/ true);
            ggml_tensor * o2    = build_norm(blk2, model.layers[il_nextn].nextn.shared_head_norm, NULL, LLM_NORM_RMS, -1);
            ggml_tensor * draft2 = ggml_mul_mat(ctx0, model.output, o2);            // {n_vocab, N}
            cb(draft2, "nextn_draft2", il_nextn);
            ggml_build_forward_expand(gf, draft2);
          } // end LLAMA_MTP_NO_STEP2 gate
        } else {
            // probe: overwrite the logits readback with the draft logits, shape {n_vocab, N} == n_outputs=N
            res->t_logits = draft;
            cur = draft;
        }
    }

    ggml_build_forward_expand(gf, cur);
}

