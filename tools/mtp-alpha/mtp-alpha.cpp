// GLM-5.2 nextn (MTP) acceptance-α probe.
//
// Two modes, selected by the LLAMA_MTP_PROBE env var (so the in-graph nextn branch — which is
// itself env-gated at graph-build time — is only ever engaged in a dedicated process):
//
//   * env UNSET   (generate):  greedily generate N tokens at temp 0 from a fixed prompt and write
//                              the full token sequence S to --alpha-seq-file.
//   * env SET     (probe):     read S, build ONE batch of all tokens with logits at every position,
//                              llama_decode (the graph overwrites t_logits with the nextn draft
//                              logits), then for i in [0, N-3] compare argmax(logits_i) to S[i+2]
//                              and report α = matches/(N-2), a rank histogram and example pairs.
//
// Both modes must be run with --enable-mtp so the nextn tensors are loaded and the layer-78 KV is
// allocated. The probe mode additionally requires LLAMA_MTP_PROBE=1.

#include "arg.h"
#include "common.h"
#include "chat.h"
#include "log.h"
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static int argmax_logits(const float * logits, int n_vocab) {
    int best = 0;
    float bestv = logits[0];
    for (int v = 1; v < n_vocab; ++v) {
        if (logits[v] > bestv) {
            bestv = logits[v];
            best  = v;
        }
    }
    return best;
}

// rank of token `tok` in the logits distribution (0 == top-1)
static int rank_of(const float * logits, int n_vocab, llama_token tok) {
    const float target = logits[tok];
    int rank = 0;
    for (int v = 0; v < n_vocab; ++v) {
        if (logits[v] > target) {
            ++rank;
        }
    }
    return rank;
}

// ---- offline math cross-check (gate 2): dump named nextn tensors for ONE column ----
// When LLAMA_MTP_DUMP is set, capture the column LLAMA_MTP_DUMP_COL of the tagged nextn
// tensors to /tmp/mtp_dump_<name>.bin (float32) so a numpy script can recompute steps 1-3
// (enorm/hnorm/eh_proj -> x) and the head (shared_head_norm -> o) and compare.
struct dump_cb_data {
    int col = 32;
    std::vector<uint8_t> tmp;
};

// the graph-build callback names nextn tensors with the layer suffix (il=78), e.g. "nextn_x-78"
static const char * g_dump_names[] = {
    "nextn_emb_shift-78", "nextn_h_prenorm-78", "nextn_ehcat-78",
    "nextn_x-78", "nextn_block-78", "nextn_o-78",
};

// ---- self-spec (gate 3): capture the nextn DRAFT logits each forward via cb_eval ----
// The in-graph nextn branch (LLAMA_MTP_SPEC) leaves res->t_logits = the TARGET logits and exposes
// the draft logits as the tensor "nextn_draft-78" of shape {n_vocab, N}. The nextn emb input is the
// batch shifted left by one column, so for a fused batch [a, d]:
//   col 0 = nextn(h_a, emb(d))  -> MTP predicts t_{a+2} = "the token after d" = the NEXT draft
//   col 1 = nextn(h_d, emb(pad)) -> junk (emb is the pad column)
// We therefore always read COLUMN 0. ggml_backend_tensor_get is synchronous and called at ask=false
// (after the node is computed), so the value is final for this forward.
// IN-GRAPH DRAFT capture: the fused forward's nextn branch produces draft logits {n_vocab, 2}:
//   col0 = nextn(h_a, emb(v))  -> the MISS draft (guess succ(v); next cycle processes v)
//   col1 = nextn(h_d, emb(w))  -> the HIT  draft (guess succ(w); next cycle processes w)
// We argmax BOTH columns in the callback (synchronous, post-compute) so the loop needs NO separate
// draft forward at all.
// K=2 chain: capture BOTH the depth-1 draft ("nextn_draft-78") and the depth-2 chained draft
// ("nextn_draft2-78"), argmaxed per column (up to 3 columns for the fused [a, d1, d2] verify batch).
// Per column c the in-graph chain seeds from succ(token@c):
//   col0 → miss path (next a = v),  col1 → 1-hit path (next a = w1),  col2 → 2-hit path (next a = w2)
// so d1[c] = depth-1 draft for that path, d2[c] = depth-2 draft for that path.
struct spec_cb_data {
    int         n_vocab = 0;
    llama_token d1[3]   = { -1, -1, -1 }; // depth-1 draft argmax per column (nextn_draft-78)
    llama_token d2[3]   = { -1, -1, -1 }; // depth-2 draft argmax per column (nextn_draft2-78)
    bool        got1    = false;
    bool        got2    = false;
};

static bool spec_eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * d = (spec_cb_data *) user_data;
    const bool want1 = (strcmp(t->name, "nextn_draft-78")  == 0);
    const bool want2 = (strcmp(t->name, "nextn_draft2-78") == 0);
    if (ask) {
        return want1 || want2;
    }
    if (!(want1 || want2)) {
        return true;
    }
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    const int64_t nv   = t->ne[0]; // n_vocab
    const int64_t cols = t->ne[1];
    static std::vector<float> buf;
    if ((int64_t) buf.size() != nv) { buf.resize(nv); }
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    auto argmax_col = [&](int c) -> llama_token {
        if (is_host) {
            const float * base = (const float *) ((const uint8_t *) t->data + c * t->nb[1]);
            return argmax_logits(base, (int) nv);
        }
        ggml_backend_tensor_get(t, buf.data(), c * t->nb[1], nv * sizeof(float));
        return argmax_logits(buf.data(), (int) nv);
    };
    llama_token * dst = want1 ? d->d1 : d->d2;
    for (int c = 0; c < 3; ++c) { dst[c] = (c < cols) ? argmax_col(c) : -1; }
    if (want1) { d->got1 = true; } else { d->got2 = true; }
    return true;
}

static bool dump_eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * d = (dump_cb_data *) user_data;
    const char * name = t->name;
    bool wanted = false;
    for (const char * n : g_dump_names) {
        if (strcmp(name, n) == 0) { wanted = true; break; }
    }
    if (ask) {
        return wanted; // request data only for the tensors we dump
    }
    if (!wanted) {
        return true;
    }
    // t is {n_embd, N, ...}; grab one column
    const int64_t ne0 = t->ne[0];
    const int64_t ne1 = t->ne[1];
    const int     col = std::min<int>(d->col, (int) ne1 - 1);
    GGML_ASSERT(t->type == GGML_TYPE_F32);

    std::vector<float> colbuf(ne0);
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    if (is_host) {
        const float * base = (const float *) ((const uint8_t *) t->data + col * t->nb[1]);
        for (int64_t i = 0; i < ne0; ++i) {
            colbuf[i] = base[i];
        }
    } else {
        ggml_backend_tensor_get(t, colbuf.data(), col * t->nb[1], ne0 * sizeof(float));
    }
    // strip the "-78" layer suffix for a clean dump filename
    char clean[128];
    snprintf(clean, sizeof(clean), "%s", name);
    char * dash = strrchr(clean, '-');
    if (dash) { *dash = '\0'; }
    char path[256];
    snprintf(path, sizeof(path), "/tmp/mtp_dump_%s.bin", clean);
    FILE * f = fopen(path, "wb");
    if (f) {
        fwrite(colbuf.data(), sizeof(float), ne0, f);
        fclose(f);
        LOG_INF("dump: %s col=%d ne0=%lld -> %s\n", name, col, (long long) ne0, path);
    }
    return true;
}

int main(int argc, char ** argv) {
    common_params params;

    // defaults; overridable on the command line
    params.prompt   = "The history of the Roman Empire is a long and complex story that begins with";
    params.n_predict = 256;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    const char * seq_file_env = getenv("LLAMA_MTP_ALPHA_SEQ_FILE");
    const std::string seq_file = seq_file_env ? seq_file_env : "/tmp/mtp_alpha_seq.txt";

    const bool probe_mode = getenv("LLAMA_MTP_PROBE") != nullptr;
    const bool spec_mode  = getenv("LLAMA_MTP_SPEC")  != nullptr;
    // interactive chat driven by the MTP self-spec loop (needs the spec graph, i.e. LLAMA_MTP_SPEC).
    const bool chat_mode  = getenv("LLAMA_MTP_CHAT")  != nullptr;

    if (!params.enable_mtp) {
        LOG_ERR("%s: this tool must be run with --enable-mtp\n", __func__);
        return 1;
    }
    if (probe_mode && spec_mode) {
        LOG_ERR("%s: set only ONE of LLAMA_MTP_PROBE / LLAMA_MTP_SPEC\n", __func__);
        return 1;
    }
    if (chat_mode && !spec_mode) {
        LOG_ERR("%s: LLAMA_MTP_CHAT requires LLAMA_MTP_SPEC=1 (the in-graph draft spec graph)\n", __func__);
        return 1;
    }

    // skip the warmup forward (this tool drives its own decodes; LLAMA_EXAMPLE_COMMON does not
    // expose --no-warmup, so set it here)
    params.warmup = false;

    llama_backend_init();
    llama_numa_init(params.numa);

    // gate-2 dump: capture named nextn tensors for one column (probe mode only)
    dump_cb_data dump_data;
    const bool dump_mode = probe_mode && getenv("LLAMA_MTP_DUMP") != nullptr;
    if (dump_mode) {
        const char * cs = getenv("LLAMA_MTP_DUMP_COL");
        if (cs) {
            dump_data.col = atoi(cs);
        }
        params.cb_eval = dump_eval_cb;
        params.cb_eval_user_data = &dump_data;
        params.warmup = false;
    }

    // gate-3 self-spec: capture the nextn draft logits each forward via cb_eval
    spec_cb_data spec_draft;
    if (spec_mode) {
        params.cb_eval = spec_eval_cb;
        params.cb_eval_user_data = &spec_draft;
        params.warmup = false;
    }

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to init model/context\n", __func__);
        return 1;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    if (chat_mode) {
        // ===================================================================
        // INTERACTIVE CHAT driven by the MTP self-spec loop (the fastest decode here, ~30 t/s).
        // Multi-turn: re-tokenize the full templated conversation each turn, prefix-match against
        // the committed KV, drop the divergent tail, prefill only the suffix (self-heals KV).
        // Generation = the same in-graph self-spec loop as the benchmark (1.77 tok/forward).
        // GLM-5.2 is a reasoning model: thinking toggled via enable_thinking (default off; /think /nothink).
        // ===================================================================
        bool enable_thinking = [](){ const char* t=getenv("THINK"); return t && strcmp(t,"0")!=0 && t[0]!='\0'; }();
        const int      n_ctx = (int) llama_n_ctx(ctx);
        llama_memory_t mem   = llama_get_memory(ctx);
        const int      cap   = params.n_predict > 0 ? params.n_predict : 2048; // max tokens / answer

        common_chat_templates_ptr tmpls = common_chat_templates_init(model, "");

        std::vector<common_chat_msg> messages;
        std::vector<llama_token>     kv_tokens; // exact tokens committed in KV (seq 0), in order
        if (getenv("SYSTEM")) { common_chat_msg m; m.role = "system"; m.content = getenv("SYSTEM"); messages.push_back(m); }

        auto piece = [&](llama_token t) -> std::string {
            char buf[256];
            int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
            return std::string(buf, n > 0 ? n : 0);
        };

        // generate one assistant turn with the self-spec loop from the current KV frontier
        // (logits for the last committed position must be ready). Streams pieces; returns the
        // generated tokens (committed in KV at [start_pos .. start_pos+len-1]).
        auto spec_generate = [&](int start_pos) -> std::vector<llama_token> {
            std::vector<llama_token> gen;
            int p = start_pos;
            llama_token a = argmax_logits(llama_get_logits_ith(ctx, -1), n_vocab); // first token @ start_pos
            llama_token d = a; // dummy seed (first fused almost always misses -> real draft from col0)
            gen.push_back(a);
            fputs(piece(a).c_str(), stdout); fflush(stdout);
            bool eog = llama_vocab_is_eog(vocab, a);
            while (!eog && (int) gen.size() < cap && p + 2 < n_ctx) {
                llama_memory_seq_rm(mem, 0, p, -1);
                spec_draft.got1 = false;
                llama_batch vb = llama_batch_init(2, 0, 1);
                common_batch_add(vb, a, p,     {0}, true);
                common_batch_add(vb, d, p + 1, {0}, true);
                int rc = llama_decode(ctx, vb);
                llama_batch_free(vb);
                if (rc != 0 || !spec_draft.got1) { fprintf(stderr, "\n[decode failed]\n"); break; }
                const llama_token v = argmax_logits(llama_get_logits_ith(ctx, 0), n_vocab);
                const llama_token w = argmax_logits(llama_get_logits_ith(ctx, 1), n_vocab);
                if (v == d) {
                    gen.push_back(d); fputs(piece(d).c_str(), stdout);
                    if (llama_vocab_is_eog(vocab, d)) { eog = true; fflush(stdout); break; }
                    gen.push_back(w); fputs(piece(w).c_str(), stdout); fflush(stdout);
                    if (llama_vocab_is_eog(vocab, w)) { eog = true; break; }
                    a = w; d = spec_draft.d1[1]; p += 2;
                } else {
                    llama_memory_seq_rm(mem, 0, p + 1, -1);
                    a = v; p += 1;
                    gen.push_back(v); fputs(piece(v).c_str(), stdout); fflush(stdout);
                    if (llama_vocab_is_eog(vocab, v)) { eog = true; break; }
                    d = spec_draft.d1[0];
                }
            }
            // commit the trailing emitted token cleanly (the last emit is uncommitted until the next
            // fused). drop the speculative overhang, re-decode it -> KV == [0 .. start_pos+len-1].
            if (!gen.empty()) {
                const int last_pos = start_pos + (int) gen.size() - 1;
                llama_memory_seq_rm(mem, 0, last_pos, -1);
                llama_batch fb = llama_batch_init(1, 0, 1);
                common_batch_add(fb, gen.back(), last_pos, {0}, true);
                llama_decode(ctx, fb);
                llama_batch_free(fb);
            }
            return gen;
        };

        printf("\n=== GLM-5.2 MTP-spec chat (thinking=%s) — /think /nothink /reset /quit ===\n",
               enable_thinking ? "on" : "off");
        std::string line;
        while (true) {
            printf("\n\033[1;36myou>\033[0m "); fflush(stdout);
            if (!std::getline(std::cin, line)) break;
            if (line == "/quit" || line == "/exit") break;
            if (line == "/think")   { enable_thinking = true;  printf("(thinking on)\n");  continue; }
            if (line == "/nothink") { enable_thinking = false; printf("(thinking off)\n"); continue; }
            if (line == "/reset") {
                messages.erase(std::remove_if(messages.begin(), messages.end(),
                    [](const common_chat_msg & m){ return m.role != "system"; }), messages.end());
                llama_memory_seq_rm(mem, 0, -1, -1); kv_tokens.clear();
                printf("(conversation reset)\n"); continue;
            }
            if (line.empty()) continue;

            { common_chat_msg um; um.role = "user"; um.content = line; messages.push_back(um); }

            common_chat_templates_inputs in;
            in.messages = messages;
            in.add_generation_prompt = true;
            in.use_jinja = true;
            in.enable_thinking = enable_thinking;
            const std::string prompt = common_chat_templates_apply(tmpls.get(), in).prompt;

            std::vector<llama_token> toks = common_tokenize(ctx, prompt, /*add_special=*/true, /*parse_special=*/true);
            if ((int) toks.size() >= n_ctx - 16) { printf("[context full — /reset]\n"); messages.pop_back(); continue; }

            // prefix-match the new prompt against committed KV; drop the divergent tail
            int n_match = 0;
            while (n_match < (int) toks.size() && n_match < (int) kv_tokens.size() && toks[n_match] == kv_tokens[n_match]) {
                n_match++;
            }
            llama_memory_seq_rm(mem, 0, n_match, -1);
            kv_tokens.resize(n_match);

            // prefill the new prompt suffix (only the very last token needs logits)
            bool ok = true;
            for (int i = n_match; i < (int) toks.size(); ) {
                const int chunk = std::min(512, (int) toks.size() - i);
                llama_batch pb = llama_batch_init(chunk, 0, 1);
                for (int j = 0; j < chunk; ++j) {
                    common_batch_add(pb, toks[i + j], i + j, {0}, /*logits=*/ (i + j) == (int) toks.size() - 1);
                }
                int rc = llama_decode(ctx, pb);
                llama_batch_free(pb);
                if (rc != 0) { fprintf(stderr, "[prefill failed]\n"); ok = false; break; }
                i += chunk;
            }
            if (!ok) { messages.pop_back(); continue; }
            for (int i = n_match; i < (int) toks.size(); ++i) kv_tokens.push_back(toks[i]);

            printf("\033[1;32mglm>\033[0m ");
            const int64_t t0 = ggml_time_us();
            std::vector<llama_token> gen = spec_generate((int) toks.size());
            const int64_t t1 = ggml_time_us();
            for (auto t : gen) kv_tokens.push_back(t);
            const double secs = (t1 - t0) / 1e6;
            printf("\n\033[2m[%d tok, %.1f t/s]\033[0m\n", (int) gen.size(), secs > 0 ? gen.size() / secs : 0.0);

            std::string resp;
            for (auto t : gen) { if (!llama_vocab_is_eog(vocab, t)) resp += piece(t); }
            common_chat_msg am; am.role = "assistant"; am.content = resp; messages.push_back(am);
        }
        printf("\n");
    } else if (spec_mode) {
        // ===================================================================
        // SELF-SPEC MODE (gate 3 + t/s): K=1 self-speculative greedy decode.
        //
        // Design (fused single-forward, see report): each cycle either
        //   (a) VERIFY+DRAFT forward, batch [a, d] @ [p, p+1], all positions output:
        //         v0 = argmax(logits@p)   = the TRUE next token after a
        //         v1 = argmax(logits@p+1) = the TRUE next token after d (bonus if d accepted)
        //         d' = argmax(draft@last) = nextn(h_a, emb(d)) = the speculated token after d
        //       if v0 == d  (draft hit): accept BOTH a and d (2 tokens), next a=d, next d=d', p+=1
        //       else        (draft miss): accept only a (1 token), roll back KV @ p+1, fall to (b)
        //   (b) DRAFT-ONLY forward, batch [a] @ [p], all positions output (after a miss, to
        //       regenerate a draft for the corrected token a=v0):
        //         d' = argmax(draft@last) = nextn(h_a, emb(a))  [NOTE: emb is the pad column = junk
        //         on a 1-token batch — see correctness note below], v0' = argmax(logits@p)
        //
        // Lossless by construction: a drafted token d is only KEPT when v0 == d, i.e. it equals the
        // target's greedy argmax. So the spec-ON token stream MUST equal the spec-OFF greedy stream.
        //
        // We run BOTH spec-ON and spec-OFF (plain greedy) in ONE process for the lossless diff and
        // a same-warm-state t/s comparison.
        // ===================================================================
        std::vector<llama_token> prompt_tokens = common_tokenize(ctx, params.prompt, /*add_special=*/true, /*parse_special=*/true);
        if (prompt_tokens.empty()) {
            LOG_ERR("%s: empty prompt tokenization\n", __func__);
            return 1;
        }
        const int n_prompt = (int) prompt_tokens.size();
        const int n_gen    = params.n_predict;

        // ---- KV-accounting invariant (verified design) ----------------------------------------
        // We keep the just-confirmed token `a` UNCOMMITTED at the top of each cycle. State:
        //   KV committed for [0 .. p-1];  `a` = the confirmed token that BELONGS at position p
        //   (not yet in KV);  `d` = nextn's speculation for the token AFTER a (or "no draft").
        //
        // fused cycle, batch [a, d] @ [p, p+1], all positions output, nextn engaged:
        //   commits a@p and (speculatively) d@p+1.
        //   v = argmax(logits@p)   = the TRUE token after a   -> CONFIRMED next token
        //   w = argmax(logits@p+1) = the TRUE token after d   -> bonus, valid only if d accepted
        //   d' = argmax(draft last col) = nextn(h_a, emb(d))  = speculation for the token after d
        //   accept (v == d): a CONFIRMED, d CONFIRMED. Next: a:=w (uncommitted, belongs @ p+2),
        //                    d:=d', p+=2.  KV holds a@p,d@p+1.
        //   reject (v != d): a CONFIRMED, d WRONG. roll back KV @ p+1 (seq_rm). Next a:=v
        //                    (uncommitted, belongs @ p+1), NO valid draft -> recovery draft-only
        //                    forward of a to reseed d, then continue.  p+=1.
        //
        // Lossless by construction: a drafted token d is kept only when v == d (== target greedy
        // argmax). So the spec-ON stream equals the plain-greedy stream. We emit out_tokens for an
        // external diff against the spec-OFF generate-mode stream on the same prompt.

        std::vector<llama_token> out_tokens; // generated tokens only (excludes prompt)
        // lossless-debug trace (LLAMA_MTP_TRACE): per emitted token, its source + KV position.
        const bool trace_on = getenv("LLAMA_MTP_TRACE") != nullptr;
        std::vector<char> emit_src;  // 's'=seed/init, 'd'=accepted draft, 'w'=bonus, 'v'=miss-correction
        std::vector<int>  emit_pos;
        auto rec = [&](char src, int pos) { if (trace_on) { emit_src.push_back(src); emit_pos.push_back(pos); } };
        // bonus diagnostic: top-3 logits of the fused col1 (= bonus w). Compared offline to the
        // baseline truth to decide whether a bonus mismatch is a tiny-margin batched-float flip
        // (inherent to spec decode) or a real logic bug (truth far down the distribution).
        std::vector<std::string> bonus_dump;
        auto dump_bonus = [&](int pos, const float * lg) {
            if (!trace_on) return;
            int t1 = 0; for (int v = 1; v < n_vocab; ++v) if (lg[v] > lg[t1]) t1 = v;
            int t2 = (t1==0)?1:0; for (int v = 0; v < n_vocab; ++v) { if (v==t1) continue; if (lg[v] > lg[t2]) t2 = v; }
            int t3 = -1; for (int v = 0; v < n_vocab; ++v) { if (v==t1||v==t2) continue; if (t3<0||lg[v]>lg[t3]) t3 = v; }
            char b[160];
            snprintf(b, sizeof(b), "pos=%d t1=%d(%.4f) t2=%d(%.4f) t3=%d(%.4f)",
                     pos, t1, lg[t1], t2, lg[t2], t3, t3>=0?lg[t3]:0.0f);
            bonus_dump.emplace_back(b);
        };
        out_tokens.reserve(n_gen + 4);

        // prefill prompt (only last position needs logits)
        {
            llama_batch pb = llama_batch_init(n_prompt, 0, 1);
            for (int i = 0; i < n_prompt; ++i) {
                common_batch_add(pb, prompt_tokens[i], i, { 0 }, /*logits=*/ i == n_prompt - 1);
            }
            if (llama_decode(ctx, pb) != 0) {
                LOG_ERR("%s: prompt prefill failed\n", __func__);
                return 1;
            }
            llama_batch_free(pb);
        }
        // KV committed for [0 .. n_prompt-1]. The first confirmed token `a` belongs at p=n_prompt.
        llama_token a = argmax_logits(llama_get_logits_ith(ctx, -1), n_vocab);
        int         p = n_prompt;

        int n_acc_draft   = 0; // depth-1 drafts accepted
        int n_draft       = 0; // depth-1 drafts proposed (== forwards in steady state)
        int n_acc_draft2  = 0; // depth-2 drafts accepted
        int n_draft2      = 0; // depth-2 drafts proposed (only after a depth-1 hit)
        int n_forwards    = 0; // model forwards in the timed loop

        llama_memory_t mem = llama_get_memory(ctx);

        const int64_t t_start_us = ggml_time_us();

        // Seed: no real drafts for the first cycle. Use dummy d1=d2=a; the first fused [a,a,a] almost
        // always misses on d1 (emits the true v) and the in-graph col0 chain yields the real next
        // drafts. No seed forward.
        llama_token d1 = a, d2 = a;
        bool eog = llama_vocab_is_eog(vocab, a);
        out_tokens.push_back(a); rec('s', p); // a came straight from the prompt argmax = confirmed

        while (!eog && (int) out_tokens.size() < n_gen) {
            // Fused VERIFY + 2-deep DRAFT, batch [a, d1, d2] @ [p, p+1, p+2], all-positions output,
            // nextn engaged (depth-1 + chained depth-2). ONE forward:
            //   main logits@0 = v  (true succ of a)
            //   main logits@1 = w1 (true succ of d1; valid iff d1 accepted)
            //   main logits@2 = w2 (true succ of d2; valid iff d1 AND d2 accepted)
            //   in-graph chain per column c (seeds from succ(token@c)):
            //     c=0 → miss path (next a=v),  c=1 → 1-hit path (next a=w1),  c=2 → 2-hit path (next a=w2)
            //     spec_draft.d1[c] = depth-1 draft for that path,  d2[c] = depth-2 draft.
            // So this verify forward ALSO produces the next cycle's 2-deep draft chain -> no separate
            // draft forward, graph topology never changes (no reuse thrash).
            llama_memory_seq_rm(mem, 0, p, -1); // drop a's stale slot (and above) to re-commit cleanly

            spec_draft.got1 = spec_draft.got2 = false;
            llama_batch vb = llama_batch_init(3, 0, 1);
            common_batch_add(vb, a,  p,     { 0 }, /*logits=*/true);
            common_batch_add(vb, d1, p + 1, { 0 }, /*logits=*/true);
            common_batch_add(vb, d2, p + 2, { 0 }, /*logits=*/true);
            const int rc = llama_decode(ctx, vb);
            llama_batch_free(vb);
            n_forwards++;
            if (rc != 0) { LOG_ERR("%s: fused decode failed @ pos %d\n", __func__, p); return 1; }
            const bool no_step2 = getenv("LLAMA_MTP_NO_STEP2") != nullptr; // per-forward isolation measurement
            if (!spec_draft.got1 || (!no_step2 && !spec_draft.got2)) { LOG_ERR("%s: drafts not captured in fused forward\n", __func__); return 1; }

            const llama_token v  = argmax_logits(llama_get_logits_ith(ctx, 0), n_vocab); // true succ(a)
            const llama_token w1 = argmax_logits(llama_get_logits_ith(ctx, 1), n_vocab); // true succ(d1)
            const llama_token w2 = argmax_logits(llama_get_logits_ith(ctx, 2), n_vocab); // true succ(d2)

            n_draft++;
            if (d1 == v) {
                // depth-1 hit: d1 confirmed @ p+1 (== true succ(a)); w1 = true succ(d1) is now valid.
                n_acc_draft++;
                out_tokens.push_back(d1); rec('d', p + 1);
                if (llama_vocab_is_eog(vocab, d1)) { eog = true; break; }
                n_draft2++; // a depth-2 draft was on the table (only meaningful once d1 hit)
                if (d2 == w1) {
                    // depth-2 hit: d2 confirmed @ p+2 (== true succ(d1)); w2 = true succ(d2) is bonus.
                    n_acc_draft2++;
                    out_tokens.push_back(d2); rec('d', p + 2);
                    if (llama_vocab_is_eog(vocab, d2)) { eog = true; break; }
                    out_tokens.push_back(w2); rec('w', p + 3);
                    dump_bonus(p + 3, llama_get_logits_ith(ctx, 2));
                    if (llama_vocab_is_eog(vocab, w2)) { eog = true; break; }
                    a = w2; d1 = spec_draft.d1[2]; d2 = spec_draft.d2[2]; p = p + 3;
                } else {
                    // depth-1 hit only: d2 wrong -> drop its slot. True next is w1 @ p+2 (bonus).
                    out_tokens.push_back(w1); rec('w', p + 2);
                    dump_bonus(p + 2, llama_get_logits_ith(ctx, 1));
                    llama_memory_seq_rm(mem, 0, p + 2, -1);
                    if (llama_vocab_is_eog(vocab, w1)) { eog = true; break; }
                    a = w1; d1 = spec_draft.d1[1]; d2 = spec_draft.d2[1]; p = p + 2;
                }
            } else {
                // miss: a confirmed @ p; d1,d2 wrong -> drop their slots. True next is v @ p+1.
                llama_memory_seq_rm(mem, 0, p + 1, -1);
                a = v; p = p + 1;
                out_tokens.push_back(a); rec('v', p);
                if (llama_vocab_is_eog(vocab, a)) { eog = true; break; }
                d1 = spec_draft.d1[0]; d2 = spec_draft.d2[0];
            }
            if (no_step2) { d2 = d1; } // step2 not built -> keep col2 a valid token for per-forward timing
        }

        const int64_t t_end_us = ggml_time_us();
        const double  secs     = (t_end_us - t_start_us) / 1e6;
        const int     n_out    = (int) out_tokens.size();
        const double  tps      = secs > 0 ? n_out / secs : 0.0;
        const double  alpha1 = n_draft  > 0 ? (double) n_acc_draft  / (double) n_draft  : 0.0;
        const double  alpha2 = n_draft2 > 0 ? (double) n_acc_draft2 / (double) n_draft2 : 0.0;

        printf("\n========== MTP self-spec (K=2 chain) ==========\n");
        printf("prompt tokens          : %d\n", n_prompt);
        printf("generated tokens       : %d\n", n_out);
        printf("model forwards         : %d\n", n_forwards);
        printf("tokens / forward       : %.4f\n", n_forwards > 0 ? (double) n_out / n_forwards : 0.0);
        printf("depth-1 drafts (prop)  : %d\n", n_draft);
        printf("depth-1 drafts (acc)   : %d\n", n_acc_draft);
        printf("ALPHA-1 (depth-1)      : %.4f\n", alpha1);
        printf("depth-2 drafts (prop)  : %d\n", n_draft2);
        printf("depth-2 drafts (acc)   : %d\n", n_acc_draft2);
        printf("ALPHA-2 (depth-2|1-hit): %.4f\n", alpha2);
        printf("decode wall-clock (s)  : %.4f\n", secs);
        printf("DECODE t/s (spec-ON)   : %.4f\n", tps);
        printf("===============================================\n");

        // emit the spec-ON token stream for the lossless diff
        const char * sf = getenv("LLAMA_MTP_SPEC_OUT");
        const std::string spec_out = sf ? sf : "/tmp/mtp_spec_on.txt";
        std::ofstream so(spec_out);
        if (so) {
            for (llama_token t : out_tokens) { so << t << "\n"; }
            so.close();
            LOG_INF("%s: wrote spec-ON stream (%d tokens) to %s\n", __func__, n_out, spec_out.c_str());
        }
        LOG_INF("%s: spec-ON text: %s\n", __func__, common_detokenize(ctx, out_tokens, true).c_str());

        if (trace_on) {
            std::ofstream tr("/tmp/mtp_spec_trace.txt");
            if (tr) {
                for (size_t i = 0; i < out_tokens.size(); ++i) {
                    tr << i << "\t" << out_tokens[i] << "\t"
                       << (i < emit_src.size() ? emit_src[i] : '?') << "\t"
                       << (i < emit_pos.size() ? emit_pos[i] : -1) << "\n";
                }
                tr.close();
                LOG_INF("%s: wrote emit trace to /tmp/mtp_spec_trace.txt\n", __func__);
            }
            std::ofstream bd("/tmp/mtp_bonus_dump.txt");
            if (bd) { for (const auto & s : bonus_dump) { bd << s << "\n"; } bd.close(); }
        }
    } else if (!probe_mode) {
        // -------------------------------------------------------------------
        // GENERATE MODE: produce the on-distribution greedy sequence S
        // -------------------------------------------------------------------
        std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, /*add_special=*/true, /*parse_special=*/true);
        if (tokens.empty()) {
            LOG_ERR("%s: empty prompt tokenization\n", __func__);
            return 1;
        }
        const int n_prompt = (int) tokens.size();

        // prefill the prompt
        llama_batch batch = llama_batch_get_one(tokens.data(), n_prompt);
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: prompt decode failed\n", __func__);
            return 1;
        }

        // greedy decode n_predict more tokens (temp 0 == argmax). Time the DECODE phase only
        // (prompt prefill excluded) so this is a clean spec-OFF baseline t/s for the A/B.
        const int n_gen = params.n_predict;
        int n_gen_done  = 0;
        const int64_t t_start_us = ggml_time_us();
        llama_token cur = argmax_logits(llama_get_logits_ith(ctx, n_prompt - 1), n_vocab);
        for (int g = 0; g < n_gen; ++g) {
            tokens.push_back(cur);
            n_gen_done++;
            if (llama_vocab_is_eog(vocab, cur)) {
                LOG_INF("%s: hit EOG after %d generated tokens\n", __func__, g + 1);
                break;
            }
            llama_batch b1 = llama_batch_get_one(&tokens.back(), 1);
            if (llama_decode(ctx, b1) != 0) {
                LOG_ERR("%s: decode failed at gen step %d\n", __func__, g);
                return 1;
            }
            cur = argmax_logits(llama_get_logits_ith(ctx, -1), n_vocab);
        }

        // write S (one token id per line)
        std::ofstream out(seq_file);
        if (!out) {
            LOG_ERR("%s: cannot open seq file %s for write\n", __func__, seq_file.c_str());
            return 1;
        }
        for (llama_token t : tokens) {
            out << t << "\n";
        }
        out.close();
        LOG_INF("%s: wrote S (%zu tokens, %d prompt) to %s\n", __func__, tokens.size(), n_prompt, seq_file.c_str());
        // also echo the decoded text for a sanity eyeball
        LOG_INF("%s: S text: %s\n", __func__, common_detokenize(ctx, tokens, true).c_str());

        const int64_t t_end_us = ggml_time_us();
        const double  secs = (t_end_us - t_start_us) / 1e6;
        printf("\n========== MTP spec-OFF baseline (greedy) ==========\n");
        printf("prompt tokens         : %d\n", n_prompt);
        printf("generated tokens      : %d\n", n_gen_done);
        printf("decode wall-clock (s) : %.4f\n", secs);
        printf("DECODE t/s (spec-OFF) : %.4f\n", secs > 0 ? n_gen_done / secs : 0.0);
        printf("====================================================\n");
    } else {
        // -------------------------------------------------------------------
        // PROBE MODE: single all-logits prefill over S, compute α
        // -------------------------------------------------------------------
        std::vector<llama_token> S;
        {
            std::ifstream in(seq_file);
            if (!in) {
                LOG_ERR("%s: cannot open seq file %s for read\n", __func__, seq_file.c_str());
                return 1;
            }
            llama_token t;
            while (in >> t) {
                S.push_back(t);
            }
        }
        const int N = (int) S.size();
        if (N < 4) {
            LOG_ERR("%s: sequence too short (N=%d)\n", __func__, N);
            return 1;
        }
        LOG_INF("%s: loaded S, N=%d tokens from %s\n", __func__, N, seq_file.c_str());

        // build ONE batch of all S tokens with logits=true at EVERY position
        llama_batch batch = llama_batch_init(N, 0, 1);
        for (int i = 0; i < N; ++i) {
            common_batch_add(batch, S[i], i, { 0 }, /*logits=*/true);
        }

        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: all-logits prefill decode failed (N=%d may exceed n_ctx/n_batch)\n", __func__, N);
            return 1;
        }

        // α = mean over i in [0, N-3] of 1[ argmax(nextn_logits_i) == S[i+2] ]
        int   n_acc = 0;
        int   n_cmp = 0;
        int   hist[6] = {0, 0, 0, 0, 0, 0}; // rank buckets: 0,1,2,3,4, >=5
        bool  any_nan = false;
        std::vector<std::string> examples;

        for (int i = 0; i <= N - 3; ++i) {
            const float * lg = llama_get_logits_ith(ctx, i);
            if (lg == nullptr) {
                LOG_ERR("%s: null logits at i=%d\n", __func__, i);
                return 1;
            }
            // NaN guard on the first row
            if (i == 0) {
                for (int v = 0; v < std::min(n_vocab, 1024); ++v) {
                    if (std::isnan(lg[v])) { any_nan = true; break; }
                }
            }
            const llama_token draft = argmax_logits(lg, n_vocab);
            const llama_token truth = S[i + 2];
            n_cmp++;
            if (draft == truth) {
                n_acc++;
            }
            const int r = rank_of(lg, n_vocab, truth);
            hist[r < 5 ? r : 5]++;

            if ((int) examples.size() < 12) {
                char buf_d[256], buf_t[256];
                int nd = llama_token_to_piece(vocab, draft, buf_d, sizeof(buf_d), 0, true);
                int nt = llama_token_to_piece(vocab, truth, buf_t, sizeof(buf_t), 0, true);
                std::string sd(buf_d, nd > 0 ? nd : 0), st(buf_t, nt > 0 ? nt : 0);
                char line[768];
                snprintf(line, sizeof(line), "  i=%-4d draft=%-7d '%s'  truth=%-7d '%s'  %s",
                         i, draft, sd.c_str(), truth, st.c_str(), draft == truth ? "MATCH" : "");
                examples.emplace_back(line);
            }
        }

        const double alpha = n_cmp > 0 ? (double) n_acc / (double) n_cmp : 0.0;

        printf("\n========== MTP nextn α-probe ==========\n");
        printf("N (sequence len)      : %d\n", N);
        printf("comparisons (N-2)     : %d\n", n_cmp);
        printf("matches               : %d\n", n_acc);
        printf("ALPHA                 : %.4f\n", alpha);
        printf("first-row NaN         : %s\n", any_nan ? "YES (probe BROKEN)" : "no");
        printf("\nrank histogram of TRUE token in draft logits (0 == draft top-1):\n");
        const char * rl[6] = {"rank 0 (top-1)", "rank 1", "rank 2", "rank 3", "rank 4", "rank >=5"};
        for (int r = 0; r < 6; ++r) {
            printf("  %-16s : %5d  (%.1f%%)\n", rl[r], hist[r], n_cmp > 0 ? 100.0 * hist[r] / n_cmp : 0.0);
        }
        printf("\nexample (draft, truth) pairs:\n");
        for (const auto & e : examples) {
            printf("%s\n", e.c_str());
        }
        printf("=======================================\n");

        llama_batch_free(batch);
    }

    llama_backend_free();
    return 0;
}
