// XTP Gate 2.5 — concurrent dual-context decode throughput probe.
//
// Decisive cheapest gate for the cross-token pipeline (XTP) engine: does running TWO independent
// full-model decode forwards CONCURRENTLY (two llama_context over ONE shared model) co-occupy the
// CPU (offloaded MoE experts) and GPU (attention/resident experts) devices, or do they serialize /
// thrash the shared 248 GB/s memory bus + threadpool?
//
// Gate 0 already proved the ggml scheduler co-occupies SYNTHETIC independent CPU+GPU splits (93-99.7%).
// But a real token forward is NOT a pure chain — it alternates CPU<->GPU ~22 times (the offloaded
// layers), and within ONE token the GPU is idle during the 19ms CPU-expert window (strict residual
// chain). Co-occupancy therefore only exists CROSS-token. This probe measures it at the REAL-forward
// granularity, which Gate 0 did not.
//
// Method (per run, repeated XTP_RUNS times):
//   Phase A  (single):     ctx0 alone, greedy decode N tokens (spec-OFF, argmax). -> single t/s.
//   Phase B  (concurrent): ctx0 + ctx1 each greedy decode N tokens on its own thread, simultaneously;
//                          ctx1 staggered by XTP_STAGGER_MS to break lockstep (mimic the engine's
//                          per-token phase offset). -> aggregate t/s.
//   Phase A2 (single):     ctx0 alone again (warm-state / thermal stability check).
//
// PRIMARY metric: factor = aggB_wall / single, where aggB_wall = (n0+n1) / (max_end - min_start)
//   (true concurrent token rate; robust to stagger ramp/tail which is negligible over N tokens).
//   aggB_sum = rB0.tps + rB1.tps is printed as a DIAGNOSTIC; |sum-wall|/wall must stay < 3% (else the
//   per-context windows are asymmetric -> UNSTABLE, distrust the run).
//
// Verdict (wall-based, median over runs; baseline = spec-OFF 25.3; MTP's 1.22x is a SEPARATE banked
// lever, NOT re-multiplied here — see docs/05):
//   factor >= 1.40 -> BUILD   (40 reachable after re-layering shipped MTP, ~7% overhead)
//   1.15..1.40     -> MARGINAL (~33-38; beats 30.89, misses 40; do the MTP-on-one-ctx follow-up)
//   < 1.15         -> DEAD    (concurrent ~= serial; ceiling stays 30.89)
// NECESSARY-not-SUFFICIENT: this measures INDEPENDENT two-context forwards; the engine runs DEPENDENT
// (MTP-speculated) forwards in ONE context on a SHARED pool. A high factor proves co-occupancy is
// physically available; the single-pool in-context capture is what the engine prototype itself proves.
//
// Knobs (env): XTP_STAGGER_MS (default 0), XTP_OFFSET_PROMPT=1 (ctx1 uses a DISTINCT prompt so the two
// positions hit DISJOINT top-8 expert sets — exposes the Effect-B scatter tax), XTP_RUNS (default 1).
//
// spec-OFF (no --enable-mtp) so the process-global g_mtp_* statics are never touched -> no race.
// Weights load once (shared, read-only); each context adds only its own KV + compute buffers (keep -c
// small to stay under the VRAM cliff, ~187/192GB used by weights). Built manually via
// harness/build_xtp_gate.sh (links prebuilt .so; sidesteps the cmake CUDA-detect bug).

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

static int argmax_logits(const float * logits, int n_vocab) {
    int   best  = 0;
    float bestv = logits[0];
    for (int v = 1; v < n_vocab; ++v) {
        if (logits[v] > bestv) { bestv = logits[v]; best = v; }
    }
    return best;
}

struct decode_result {
    int                      n_done = 0;
    double                   secs   = 0.0;   // own timed window
    double                   tps    = 0.0;
    int                      rc     = 0;      // 0 ok, nonzero = a llama_decode failure
    int64_t                  t0_us  = 0;      // absolute start of timed region
    int64_t                  t1_us  = 0;      // absolute end of timed region
    std::vector<llama_token> toks;
};

// Plain greedy (temp 0 == argmax) decode. Prefill done & EXCLUDED from timing. If ready/go given, the
// thread signals prefill-done then spins for the common GO so both start together; stagger_ms (applied
// AFTER go) offsets this context's phase to break lockstep without a measurable ramp/tail.
//
// verify_width N (default 1): if N>1 this context becomes the SPECULATIVE side — every forward submits
// a BATCH of N tokens at N distinct positions, ALL with logits=true, in ONE llama_decode call. That is
// the timing shape of a tree-verify: N candidate columns => N× expert-read columns per forward. The KV
// then advances by N positions/forward (lossless-irrelevant: this is a throughput/latency probe, so the
// N batched tokens are filler positions, not a verified path). n_done counts columns processed (so t/s
// is column-throughput, directly comparable to the N=1 single-column rate). Each forward is one
// "verify-width-N forward"; we run n_gen of them.
static decode_result greedy_decode(llama_context * ctx, const llama_vocab * vocab, int n_vocab,
                                   const std::vector<llama_token> & prompt, int n_gen,
                                   std::atomic<int> * ready = nullptr, std::atomic<bool> * go = nullptr,
                                   int stagger_ms = 0, int verify_width = 1) {
    decode_result r;
    std::vector<llama_token> tokens = prompt;
    const int n_prompt = (int) tokens.size();

    {   // prefill (untimed)
        llama_batch pb = llama_batch_get_one(tokens.data(), n_prompt);
        if (llama_decode(ctx, pb) != 0) { r.rc = 1; return r; }
    }
    llama_token cur = argmax_logits(llama_get_logits_ith(ctx, n_prompt - 1), n_vocab);

    if (ready && go) {
        ready->fetch_add(1);
        while (!go->load()) { /* spin until both prefills done */ }
    }
    if (stagger_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(stagger_ms));

    const int N = std::max(1, verify_width);

    if (N == 1) {
        // ---- confirm side: normal 1-token (MTP-style) forward, unchanged path ----
        r.t0_us = ggml_time_us();
        for (int g = 0; g < n_gen; ++g) {
            tokens.push_back(cur);
            r.n_done++;
            if (llama_vocab_is_eog(vocab, cur)) break;
            llama_batch b1 = llama_batch_get_one(&tokens.back(), 1);
            if (llama_decode(ctx, b1) != 0) { r.rc = 2; break; }
            cur = argmax_logits(llama_get_logits_ith(ctx, -1), n_vocab);
        }
        r.t1_us = ggml_time_us();
    } else {
        // ---- speculative side: N-column batched forwards (tree-verify width N) ----
        // One reusable batch holding N tokens, each at its own position, ALL emitting logits so the
        // forward must compute N output columns => N expert-read columns. KV advances by N per forward.
        llama_batch vb = llama_batch_init(N, 0, 1);
        const llama_seq_id seq0 = 0;
        const int n_ctx_max = (int) llama_n_ctx(ctx);
        llama_pos pos = n_prompt;   // next free KV position after prefill
        r.t0_us = ggml_time_us();
        for (int g = 0; g < n_gen; ++g) {
            if (pos + N > n_ctx_max) break;   // KV guard: stop before overflowing this ctx's n_ctx
            common_batch_clear(vb);
            // Column 0 = the real argmax continuation; columns 1..N-1 are filler candidate columns at
            // the next N-1 positions (distinct positions => distinct attention/expert work per column).
            for (int j = 0; j < N; ++j) {
                // All N columns carry `cur` at N distinct positions; logits=true forces N output
                // columns => N expert-read columns (the tree-verify cost). Token identity is irrelevant
                // to timing (this is a throughput probe, not a verified path).
                common_batch_add(vb, cur, pos + j, { seq0 }, /*logits=*/true);
            }
            r.n_done += N;                // count COLUMNS processed (column-throughput)
            if (llama_vocab_is_eog(vocab, cur)) break;
            if (llama_decode(ctx, vb) != 0) { r.rc = 2; break; }
            // advance: take the last column's argmax as the next forward's seed; KV grew by N.
            cur = argmax_logits(llama_get_logits_ith(ctx, N - 1), n_vocab);
            pos += N;
        }
        r.t1_us = ggml_time_us();
        llama_batch_free(vb);
    }

    r.secs  = (r.t1_us - r.t0_us) / 1e6;
    r.tps   = r.secs > 0 ? r.n_done / r.secs : 0.0;
    r.toks  = std::move(tokens);
    return r;
}

static int getenv_int(const char * k, int dflt) { const char * e = getenv(k); return e ? atoi(e) : dflt; }

struct config_stats { double f_med=0, f_min=0, f_sd=0, s_med=0, a_med=0; int unstable=0; };

static double vmedian(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n == 0 ? 0.0 : (n % 2 ? v[n/2] : 0.5 * (v[n/2 - 1] + v[n/2]));
}
static double vstdev(const std::vector<double> & v) {
    if (v.size() < 2) return 0.0;
    double m = 0; for (double x : v) m += x; m /= v.size();
    double s = 0; for (double x : v) s += (x-m)*(x-m);
    return std::sqrt(s / (v.size() - 1));
}

// Run one config (a given stagger + which prompt ctx1 uses) `runs` times; print per-run lines; return
// median/min/stddev of the factor. ctx0 always decodes prompt0; ctx1 decodes prompt1 (= a distinct
// prompt when `offset`). Contexts are reused across configs (KV cleared each phase) — one model load.
// verify_width applies to ctx1 (the SPECULATIVE side). The single-context baseline (Phase A / A2) and
// ctx0 in Phase B always run at width 1 (the confirm side). So `single` stays the normal 1-column rate,
// while the concurrent aggregate includes ctx1's N-column forwards — exactly the cross-token co-occupancy
// of a confirm forward overlapped with a width-N tree-verify forward.
static config_stats run_config(llama_context * ctx0, llama_context * ctx1, const llama_vocab * vocab,
                               int n_vocab, const std::vector<llama_token> & prompt0,
                               const std::vector<llama_token> & prompt1, int n_gen,
                               int stagger_ms, bool offset, int runs, const char * label,
                               int verify_width = 1) {
    auto clear_ctx = [](llama_context * c) { llama_memory_clear(llama_get_memory(c), true); };
    std::vector<double> factors, agg_walls, singles;
    config_stats cs;
    for (int run = 0; run < runs; ++run) {
        clear_ctx(ctx0);
        decode_result rA = greedy_decode(ctx0, vocab, n_vocab, prompt0, n_gen);
        if (rA.rc) { LOG_ERR("%s run%d Phase A rc=%d\n", label, run, rA.rc); continue; }

        clear_ctx(ctx0); clear_ctx(ctx1);
        decode_result rB0, rB1;
        std::atomic<int>  ready{0};
        std::atomic<bool> go{false};
        std::thread th0([&]{ rB0 = greedy_decode(ctx0, vocab, n_vocab, prompt0, n_gen, &ready, &go, 0, 1); });
        std::thread th1([&]{ rB1 = greedy_decode(ctx1, vocab, n_vocab, prompt1, n_gen, &ready, &go, stagger_ms, verify_width); });
        while (ready.load() < 2) { }
        go.store(true);
        th0.join(); th1.join();
        if (rB0.rc || rB1.rc) { LOG_ERR("%s run%d Phase B rc0=%d rc1=%d\n", label, run, rB0.rc, rB1.rc); continue; }

        const int64_t min_start = std::min(rB0.t0_us, rB1.t0_us);
        const int64_t max_end   = std::max(rB0.t1_us, rB1.t1_us);
        const double  wallB     = (max_end - min_start) / 1e6;
        const int     nB        = rB0.n_done + rB1.n_done;
        const double  aggB_wall = wallB > 0 ? nB / wallB : 0.0;
        const double  aggB_sum  = rB0.tps + rB1.tps;
        const double  skew      = aggB_wall > 0 ? std::fabs(aggB_sum - aggB_wall) / aggB_wall : 1.0;
        const bool    unstable  = skew > 0.03;
        if (unstable) cs.unstable++;

        clear_ctx(ctx0);
        decode_result rA2 = greedy_decode(ctx0, vocab, n_vocab, prompt0, n_gen);
        if (rA2.rc) { LOG_ERR("%s run%d Phase A2 rc=%d\n", label, run, rA2.rc); continue; }

        const double single = 0.5 * (rA.tps + rA2.tps);
        const double factor = single > 0 ? aggB_wall / single : 0.0;
        factors.push_back(factor); agg_walls.push_back(aggB_wall); singles.push_back(single);
        printf("  [%s run%d N=%d] single=%.3f | concur ctx0=%.3f ctx1=%.3f wall=%.3f sum=%.3f%s | factor=%.3f\n",
               label, run, verify_width, single, rB0.tps, rB1.tps, aggB_wall, aggB_sum,
               unstable ? " [UNSTABLE]" : "", factor);
        fflush(stdout);
    }
    if (factors.empty()) return cs;
    cs.f_med = vmedian(factors);
    cs.f_min = *std::min_element(factors.begin(), factors.end());
    cs.f_sd  = vstdev(factors);
    cs.s_med = vmedian(singles);
    cs.a_med = vmedian(agg_walls);
    return cs;
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    params.warmup = false;  // we drive our own decodes

    const int  stagger_ms = getenv_int("XTP_STAGGER_MS", 0);
    const int  runs       = std::max(1, getenv_int("XTP_RUNS", 1));
    const bool offset     = getenv_int("XTP_OFFSET_PROMPT", 0) != 0;
    // XTP_VERIFY_WIDTH: width sweep for the SPECULATIVE (ctx1) forward. 0 (default) = legacy behaviour
    // (no sweep). A single value (e.g. 4) runs that one width. The sweep mode XTP_WIDTH_SWEEP=1 runs the
    // canonical N=1,2,4 sweep (each as a stagger+distinct-prompt R3-style config) and reports C(N).
    const bool width_sweep = getenv_int("XTP_WIDTH_SWEEP", 0) != 0;
    const int  verify_width_env = getenv_int("XTP_VERIFY_WIDTH", 0);

    llama_backend_init();
    llama_numa_init(params.numa);

    // ---- load the model ONCE; ctx0 from the common helper ----
    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx0  = llama_init->context();
    if (model == nullptr || ctx0 == nullptr) { LOG_ERR("%s: failed to init model/ctx0\n", __func__); return 1; }

    // ---- second context over the SAME model (shared read-only weights) ----
    llama_context_params cparams = common_context_params_to_llama(params);
    llama_context * ctx1 = llama_init_from_model(model, cparams);
    if (ctx1 == nullptr) { LOG_ERR("%s: ctx1 init failed — likely VRAM OOM; lower -c\n", __func__); return 1; }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    const bool matrix = getenv_int("XTP_MATRIX", 0) != 0;
    const int  st     = stagger_ms > 0 ? stagger_ms : 8;   // matrix uses ~8ms (≈one GPU phase)

    // ctx0 always uses params.prompt. prompt1 is a DISTINCT prompt (so the two positions route to
    // DISJOINT top-8-of-256 experts, exposing the Effect-B scatter tax) whenever it could be used as
    // ctx1's source: single-config with XTP_OFFSET_PROMPT, OR matrix mode (R3 needs it).
    const bool need_distinct = offset || matrix || width_sweep || verify_width_env > 1;
    const std::string prompt1_txt = need_distinct
        ? "In the field of quantum mechanics, the behavior of subatomic particles is described by"
        : params.prompt;
    std::vector<llama_token> prompt0 = common_tokenize(ctx0, params.prompt, true, true);
    std::vector<llama_token> prompt1 = common_tokenize(ctx1, prompt1_txt,  true, true);
    if (prompt0.empty() || prompt1.empty()) { LOG_ERR("%s: empty prompt tokenization\n", __func__); return 1; }
    const int n_gen = params.n_predict > 0 ? params.n_predict : 200;

    printf("\n######## XTP Gate 2.5 — concurrent dual-context decode ########\n");
    printf("n_gen=%d  n_threads=%d  n_ctx=%d  stagger_ms=%d  offset_prompt=%d  matrix=%d  width_sweep=%d  verify_width=%d  runs=%d\n",
           n_gen, (int) params.cpuparams.n_threads, (int) llama_n_ctx(ctx0), stagger_ms, (int) offset, (int) matrix,
           (int) width_sweep, verify_width_env, runs);
    printf("prompt0 tok=%d  prompt1 tok=%d\n", (int) prompt0.size(), (int) prompt1.size());

    auto verdict = [](double f_med, double f_min) {
        if      (f_med >= 1.40 && f_min >= 1.20) return "BUILD (40 reachable after MTP re-layer)";
        else if (f_med >= 1.15)                  return "MARGINAL (~33-38; MTP-on-one-ctx follow-up)";
        else                                     return "DEAD (bus ceiling; ship 30.89)";
    };

    if (width_sweep || verify_width_env > 1) {
        // ===== VERIFY-WIDTH SWEEP: how cross-token co-occupancy cost C scales with tree-verify width N.
        // ctx0 = confirm side (1-column MTP-style forward). ctx1 = speculative side decoding N-column
        // batches per forward. Each config is R3-style: stagger=st + DISTINCT prompt (so columns route
        // to disjoint experts = the realistic scatter tax). single = the 1-column baseline (Phase A/A2).
        //
        // factor(N) = aggB_wall / single  (same metric the tool already computes). Per the user's XTP
        // economics, C(N) = 2 / factor(N): factor=2 => perfect overlap => C=1; factor=1 => fully
        // serialized => C=2. As N grows the speculative forward gets read-heavier, so we expect factor
        // to fall and C(N) to rise — that rise is the tree-verify co-occupancy tax we are measuring.
        const int  sweep_st = stagger_ms > 0 ? stagger_ms : 8;  // ~one GPU phase, like matrix mode
        std::vector<int> widths;
        if (width_sweep) { widths = {1, 2, 4}; }
        else             { widths = {verify_width_env}; }       // single explicit width

        printf("\n=== VERIFY-WIDTH SWEEP (stagger=%dms, distinct prompt, R3-style) ===\n", sweep_st);
        std::vector<config_stats> ws;
        for (int N : widths) {
            char lbl[32];
            snprintf(lbl, sizeof(lbl), "W%d", N);
            printf("\n--- verify_width N=%d (ctx1 decodes %d-column batches/forward) ---\n", N, N);
            config_stats cs = run_config(ctx0, ctx1, vocab, n_vocab, prompt0, prompt1, n_gen,
                                         sweep_st, true, runs, lbl, N);
            ws.push_back(cs);
        }

        printf("\n======== VERIFY-WIDTH SWEEP SUMMARY ========\n");
        printf("  N   factor(med/min/sd)      C(N)=2/factor   single   concurrent\n");
        double f1 = 0.0;
        for (size_t i = 0; i < widths.size(); ++i) {
            const config_stats & cs = ws[i];
            const double C = cs.f_med > 0 ? 2.0 / cs.f_med : 0.0;
            if (widths[i] == 1) f1 = cs.f_med;
            printf("  %-3d %.3f/%.3f/%.3f       %.3f          %.2f    %.2f\n",
                   widths[i], cs.f_med, cs.f_min, cs.f_sd, C, cs.s_med, cs.a_med);
        }
        // Show how C grows relative to N=1 if it was measured.
        if (f1 > 0.0) {
            printf("  ---- C(N)/C(1) growth (read-tax of widening verify) ----\n");
            const double C1 = 2.0 / f1;
            for (size_t i = 0; i < widths.size(); ++i) {
                const double C = ws[i].f_med > 0 ? 2.0 / ws[i].f_med : 0.0;
                printf("    N=%d : C=%.3f  C/C(1)=%.3f\n", widths[i], C, C1 > 0 ? C / C1 : 0.0);
            }
        }
        printf("NOTE: factor uses COLUMN-throughput on ctx1 (n_done counts N columns/forward), so it is\n");
        printf("      directly comparable to the 1-column single baseline. C(N) is the cross-token\n");
        printf("      co-occupancy cost; tree-XTP nets positive iff (1+beta_N)/C(N) > 1 (beta from acc@N).\n");
        printf("CAVEAT: independent 2-ctx proxy (separate KV, separate prefill state). It captures the\n");
        printf("      SHARED memory-bus + threadpool contention of overlapping a confirm forward with an\n");
        printf("      N-wide verify forward, but NOT the single-context in-engine tree (shared KV page\n");
        printf("      reuse / one batched verify call). Treat C(N) as an UPPER-ish bound on the tax.\n");
        printf("============================================\n");
    } else if (matrix) {
        // verdict-critical sweep in ONE model load. R1 sync (lockstep floor), R2 stagger (Effect A),
        // R3 stagger+offset (Effect B scatter tax — THE verdict run).
        printf("\n=== R1 sync (stagger=0, same prompt) — lockstep floor ===\n");
        config_stats r1 = run_config(ctx0, ctx1, vocab, n_vocab, prompt0, prompt0, n_gen, 0,  false, runs, "R1");
        printf("\n=== R2 stagger=%dms (same prompt) — Effect A (idle removal) ===\n", st);
        config_stats r2 = run_config(ctx0, ctx1, vocab, n_vocab, prompt0, prompt0, n_gen, st, false, runs, "R2");
        printf("\n=== R3 stagger=%dms + DISTINCT prompt — Effect B (scatter tax) [VERDICT] ===\n", st);
        config_stats r3 = run_config(ctx0, ctx1, vocab, n_vocab, prompt0, prompt1, n_gen, st, true,  runs, "R3");

        printf("\n======== MATRIX VERDICT ========\n");
        printf("           factor(med/min/sd)   single   concurrent   note\n");
        printf("R1 sync    %.3f/%.3f/%.3f     %.2f    %.2f       lockstep floor (FALSE-LOW)\n",
               r1.f_med, r1.f_min, r1.f_sd, r1.s_med, r1.a_med);
        printf("R2 stagger %.3f/%.3f/%.3f     %.2f    %.2f       Effect A (idle removal)\n",
               r2.f_med, r2.f_min, r2.f_sd, r2.s_med, r2.a_med);
        printf("R3 offset  %.3f/%.3f/%.3f     %.2f    %.2f       <<< VERDICT (cross-token realistic)\n",
               r3.f_med, r3.f_min, r3.f_sd, r3.s_med, r3.a_med);
        printf("Effect-A gain (R2/R1)=%.3f   Effect-B tax (R3/R2)=%.3f   unstable=%d/%d/%d\n",
               r1.f_med>0?r2.f_med/r1.f_med:0.0, r2.f_med>0?r3.f_med/r2.f_med:0.0,
               r1.unstable, r2.unstable, r3.unstable);
        printf("VERDICT (R3): %s\n", verdict(r3.f_med, r3.f_min));
        printf("NOTE: independent 2-ctx proxy; high factor NECESSARY not SUFFICIENT (single-pool in-ctx\n");
        printf("      capture is what the engine prototype must then prove).\n");
        printf("================================\n");
    } else {
        config_stats cs = run_config(ctx0, ctx1, vocab, n_vocab, prompt0, prompt1, n_gen,
                                     stagger_ms, offset, runs, "single");
        printf("\n======== VERDICT ========\n");
        printf("runs=%d unstable=%d  single=%.3f  concurrent=%.3f\n", runs, cs.unstable, cs.s_med, cs.a_med);
        printf("factor median/min/sd : %.3f / %.3f / %.3f\n", cs.f_med, cs.f_min, cs.f_sd);
        printf("verdict              : %s\n", verdict(cs.f_med, cs.f_min));
        printf("=========================\n");
    }

    llama_free(ctx1);
    return 0;
}
