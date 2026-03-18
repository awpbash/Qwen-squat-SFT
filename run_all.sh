#!/bin/bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════
# run_all.sh v2 — Master orchestration for squat VLM experiments
#
# Runs the full pipeline:
#   Phase 0 (CPU): Data preparation
#   Phase 1 (GPU): Training (SFT + Agentic)
#   Phase 2 (GPU): Evaluation on controlled (QEVD) domain
#   Phase 3 (GPU): Evaluation on wild (Reddit) domain
#   Phase 4 (CPU): Statistical analysis + thesis figures
#
# Usage:
#   bash run_all.sh                    # Run everything
#   bash run_all.sh --phase 2          # Start from phase 2
#   bash run_all.sh --phase 2 --dry    # Dry run (print commands only)
#   bash run_all.sh --skip-training    # Skip phase 1 (use existing checkpoints)
#
# Environment:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_all.sh   # Use specific GPUs
#   NUM_GPUS=2 bash run_all.sh                       # Override GPU count
#   SFT_CHECKPOINT=path/to/ckpt bash run_all.sh      # Use specific checkpoint
#   AGENTIC_CHECKPOINT=path/to/ckpt bash run_all.sh   # Use specific agentic checkpoint
# ═══════════════════════════════════════════════════════════════════════

# ── Configuration ─────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
FINETUNE_DIR="$PROJECT_ROOT/Qwen-VL-Series-Finetune"
BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

# Data paths
CLEAN_DATA="$PROJECT_ROOT/new_qevd_mrq_clean.json"
TRAIN_DATA="$PROJECT_ROOT/new_qevd_mrq_train.json"
VAL_DATA="$PROJECT_ROOT/new_qevd_mrq_val.json"
TEST_DATA="$PROJECT_ROOT/new_qevd_mrq_test.json"
MIXED_TRAIN_DATA="$PROJECT_ROOT/new_qevd_mrq_mixed_train.json"
REDDIT_TEST_DATA="$PROJECT_ROOT/reddit_eval_test.json"
VIDEO_ROOT="$PROJECT_ROOT"

# Checkpoints (auto-detected or overridden via env)
SFT_CHECKPOINT="${SFT_CHECKPOINT:-}"
AGENTIC_CHECKPOINT="${AGENTIC_CHECKPOINT:-}"
SFT_OUTPUT_DIR="$FINETUNE_DIR/output/sft_lora_v3"
AGENTIC_OUTPUT_DIR="$FINETUNE_DIR/output/sft_lora_agentic_v3"

# Results directory
EVAL_RESULTS="$PROJECT_ROOT/eval_results"

# Video processing (must match training config in finetune_video.sh)
FPS=4
VIDEO_MIN_PIXELS=$((16 * 32 * 32))   # 16384
VIDEO_MAX_PIXELS=$((128 * 32 * 32))  # 131072

# ── Argument parsing ─────────────────────────────────────────────────
START_PHASE=0
DRY_RUN=false
SKIP_TRAINING=false
MAX_SAMPLES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --phase) START_PHASE="$2"; shift 2 ;;
        --dry) DRY_RUN=true; shift ;;
        --skip-training) SKIP_TRAINING=true; shift ;;
        --max-samples) MAX_SAMPLES="--max_samples $2"; shift 2 ;;
        --sft-checkpoint) SFT_CHECKPOINT="$2"; shift 2 ;;
        --agentic-checkpoint) AGENTIC_CHECKPOINT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'
BOLD='\033[1m'

log_phase() { echo -e "\n${BOLD}${BLUE}═══════════════════════════════════════════════════${NC}"; echo -e "${BOLD}${BLUE}  PHASE $1: $2${NC}"; echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════${NC}\n"; }
log_step()  { echo -e "${GREEN}  ▸ $1${NC}"; }
log_warn()  { echo -e "${YELLOW}  ⚠ $1${NC}"; }
log_error() { echo -e "${RED}  ✗ $1${NC}"; }
log_done()  { echo -e "${GREEN}  ✓ $1${NC}"; }

run_cmd() {
    if $DRY_RUN; then
        echo -e "${YELLOW}  [DRY] $*${NC}"
    else
        "$@"
    fi
}

# Run a command from a specific directory, then return to PROJECT_ROOT
run_in_dir() {
    local dir="$1"
    shift
    if $DRY_RUN; then
        echo -e "${YELLOW}  [DRY] (cd $dir && $*)${NC}"
    else
        (cd "$dir" && "$@")
    fi
}

# Auto-detect best checkpoint in a training output directory.
# When load_best_model_at_end=True, the best model is saved to the root
# output dir (adapter_model.safetensors). Use that if it exists, otherwise
# fall back to the highest-numbered checkpoint subdirectory.
find_best_checkpoint() {
    local output_dir="$1"
    if [ ! -d "$output_dir" ]; then
        echo ""
        return
    fi
    # Prefer root dir (best model from load_best_model_at_end)
    if [ -f "$output_dir/adapter_model.safetensors" ]; then
        echo "$output_dir"
        return
    fi
    # Fallback: last checkpoint
    local latest
    latest=$(ls -d "$output_dir"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    echo "$latest"
}

# ── Pre-flight checks ────────────────────────────────────────────────
echo -e "${BOLD}Pre-flight checks...${NC}"

if [ ! -f "$CLEAN_DATA" ]; then
    log_error "Clean data not found: $CLEAN_DATA"
    log_error "Run clean_labels.py first"
    exit 1
fi

if ! command -v python &>/dev/null; then
    log_error "Python not found"
    exit 1
fi

echo -e "  Project root:  $PROJECT_ROOT"
echo -e "  Base model:    $BASE_MODEL"
echo -e "  Start phase:   $START_PHASE"
echo -e "  Dry run:       $DRY_RUN"
echo -e "  Skip training: $SKIP_TRAINING"
echo ""

# ═══════════════════════════════════════════════════════════════════════
# PHASE 0: DATA PREPARATION (CPU)
# ═══════════════════════════════════════════════════════════════════════
if [ "$START_PHASE" -le 0 ]; then
    log_phase 0 "DATA PREPARATION"

    # Step 0a: Create train/val/test splits with aggressive oversampling
    if [ -f "$TRAIN_DATA" ] && [ -f "$VAL_DATA" ] && [ -f "$TEST_DATA" ]; then
        log_warn "Splits already exist, skipping (delete to regenerate)"
    else
        log_step "Creating train/val/test splits with aggressive oversampling..."
        run_in_dir "$PROJECT_ROOT" python create_splits.py
        log_done "Splits created"
    fi

    # Step 0b: Prepare Reddit wild evaluation set
    if [ -f "$REDDIT_TEST_DATA" ]; then
        log_warn "Reddit eval data already exists, skipping"
    else
        log_step "Preparing Reddit wild evaluation set..."
        run_in_dir "$PROJECT_ROOT" python prepare_reddit_eval.py
        log_done "Reddit eval set prepared"
    fi

    # Step 0c: Generate agentic training data
    if [ -f "$MIXED_TRAIN_DATA" ]; then
        log_warn "Mixed agentic training data already exists, skipping"
    else
        log_step "Generating agentic training data (50/50 standard + agentic)..."
        run_in_dir "$PROJECT_ROOT" python generate_agentic_training_data.py
        log_done "Agentic training data generated"
    fi

    # Print data summary
    if ! $DRY_RUN; then
        echo ""
        log_step "Data summary:"
        for f in "$TRAIN_DATA" "$VAL_DATA" "$TEST_DATA" "$MIXED_TRAIN_DATA" "$REDDIT_TEST_DATA"; do
            if [ -f "$f" ]; then
                count=$(python -c "import json; print(len(json.load(open('$f'))))" 2>/dev/null || echo "?")
                echo "    $(basename "$f"): $count samples"
            fi
        done
    fi
fi

# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: TRAINING (GPU)
# ═══════════════════════════════════════════════════════════════════════
if [ "$START_PHASE" -le 1 ] && ! $SKIP_TRAINING; then
    log_phase 1 "TRAINING"

    # Step 1a: Standard SFT with LoRA
    if [ -n "$SFT_CHECKPOINT" ]; then
        log_warn "Using provided SFT checkpoint: $SFT_CHECKPOINT"
    elif [ -d "$SFT_OUTPUT_DIR" ] && ls "$SFT_OUTPUT_DIR"/checkpoint-* &>/dev/null; then
        SFT_CHECKPOINT=$(find_best_checkpoint "$SFT_OUTPUT_DIR")
        log_warn "SFT training output exists, using: $SFT_CHECKPOINT"
    else
        log_step "Training E2: Standard SFT with LoRA (finetune_video.sh)..."
        run_in_dir "$FINETUNE_DIR" bash scripts/finetune_video.sh
        SFT_CHECKPOINT=$(find_best_checkpoint "$SFT_OUTPUT_DIR")
        log_done "SFT training complete: $SFT_CHECKPOINT"
    fi

    # Step 1b: Agentic SFT with LoRA
    if [ -n "$AGENTIC_CHECKPOINT" ]; then
        log_warn "Using provided agentic checkpoint: $AGENTIC_CHECKPOINT"
    elif [ -d "$AGENTIC_OUTPUT_DIR" ] && ls "$AGENTIC_OUTPUT_DIR"/checkpoint-* &>/dev/null; then
        AGENTIC_CHECKPOINT=$(find_best_checkpoint "$AGENTIC_OUTPUT_DIR")
        log_warn "Agentic training output exists, using: $AGENTIC_CHECKPOINT"
    else
        log_step "Training E5b: Agentic SFT with LoRA (finetune_video_agentic.sh)..."
        run_in_dir "$FINETUNE_DIR" bash scripts/finetune_video_agentic.sh
        AGENTIC_CHECKPOINT=$(find_best_checkpoint "$AGENTIC_OUTPUT_DIR")
        log_done "Agentic training complete: $AGENTIC_CHECKPOINT"
    fi
else
    if $SKIP_TRAINING; then
        log_warn "Skipping training phase"
    fi
fi

# Auto-detect checkpoints if not set
if [ -z "$SFT_CHECKPOINT" ]; then
    SFT_CHECKPOINT=$(find_best_checkpoint "$SFT_OUTPUT_DIR")
fi
if [ -z "$AGENTIC_CHECKPOINT" ]; then
    AGENTIC_CHECKPOINT=$(find_best_checkpoint "$AGENTIC_OUTPUT_DIR")
fi

# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: EVALUATION — CONTROLLED DOMAIN (QEVD)
# ═══════════════════════════════════════════════════════════════════════
if [ "$START_PHASE" -le 2 ]; then
    log_phase 2 "EVALUATION — CONTROLLED (QEVD)"

    DOMAIN="controlled"

    # ── E1: Zero-shot baseline ────────────────────────────────────────
    E1_DIR="$EVAL_RESULTS/E1_zero_shot/$DOMAIN"
    if [ -f "$E1_DIR/metrics.json" ]; then
        log_warn "E1 ($DOMAIN) already evaluated, skipping"
    else
        log_step "E1: Zero-shot evaluation (no fine-tuning)..."
        run_cmd python "$PROJECT_ROOT/eval_unified.py" \
            --model_path "$BASE_MODEL" \
            --data_path "$TEST_DATA" \
            --video_root "$VIDEO_ROOT" \
            --fps $FPS \
            --output_dir "$E1_DIR" \
            --domain "$DOMAIN" \
            --video_min_pixels $VIDEO_MIN_PIXELS \
            --video_max_pixels $VIDEO_MAX_PIXELS \
            $MAX_SAMPLES
        log_done "E1 ($DOMAIN) complete"
    fi

    # ── E2: SFT with LoRA ────────────────────────────────────────────
    E2_DIR="$EVAL_RESULTS/E2_sft/$DOMAIN"
    if [ -f "$E2_DIR/metrics.json" ]; then
        log_warn "E2 ($DOMAIN) already evaluated, skipping"
    elif [ -z "$SFT_CHECKPOINT" ]; then
        log_error "E2: No SFT checkpoint found. Run training first or provide --sft-checkpoint"
    else
        log_step "E2: SFT LoRA evaluation..."
        run_cmd python "$PROJECT_ROOT/eval_unified.py" \
            --model_path "$SFT_CHECKPOINT" \
            --base_model "$BASE_MODEL" \
            --data_path "$TEST_DATA" \
            --video_root "$VIDEO_ROOT" \
            --fps $FPS \
            --output_dir "$E2_DIR" \
            --domain "$DOMAIN" \
            --video_min_pixels $VIDEO_MIN_PIXELS \
            --video_max_pixels $VIDEO_MAX_PIXELS \
            $MAX_SAMPLES
        log_done "E2 ($DOMAIN) complete"
    fi

    # ── E3: Pose-only (no VLM) ───────────────────────────────────────
    E3_DIR="$EVAL_RESULTS/E3_pose_only/$DOMAIN"
    if [ -f "$E3_DIR/metrics.json" ]; then
        log_warn "E3 ($DOMAIN) already evaluated, skipping"
    else
        log_step "E3: Pose-only baseline (MediaPipe rules)..."
        run_cmd python "$PROJECT_ROOT/eval_pose_only.py" \
            --data_path "$TEST_DATA" \
            --video_root "$VIDEO_ROOT" \
            --fps $FPS \
            --output_dir "$E3_DIR" \
            --domain "$DOMAIN" \
            $MAX_SAMPLES
        log_done "E3 ($DOMAIN) complete"
    fi

    # ── E4: Always-tool (VLM + Pose pipeline) ────────────────────────
    E4_DIR="$EVAL_RESULTS/E4_always_tool/$DOMAIN"
    if [ -f "$E4_DIR/metrics.json" ]; then
        log_warn "E4 ($DOMAIN) already evaluated, skipping"
    elif [ -z "$SFT_CHECKPOINT" ]; then
        log_error "E4: No SFT checkpoint found. Skipping."
    else
        log_step "E4: Always-tool (VLM pass1 → Pose → VLM refinement)..."
        run_cmd python "$PROJECT_ROOT/inference_with_pose.py" \
            --model_path "$SFT_CHECKPOINT" \
            --base_model "$BASE_MODEL" \
            --data_path "$TEST_DATA" \
            --video_root "$VIDEO_ROOT" \
            --fps $FPS \
            --output_dir "$E4_DIR" \
            --domain "$DOMAIN" \
            --video_min_pixels $VIDEO_MIN_PIXELS \
            --video_max_pixels $VIDEO_MAX_PIXELS \
            $MAX_SAMPLES
        log_done "E4 ($DOMAIN) complete"
    fi

    # ── E5: Agentic — untrained (zero-shot ReAct) ────────────────────
    E5_DIR="$EVAL_RESULTS/E5_agentic_untrained/$DOMAIN"
    if [ -f "$E5_DIR/metrics.json" ]; then
        log_warn "E5 ($DOMAIN) already evaluated, skipping"
    else
        log_step "E5: Agentic (untrained, zero-shot ReAct)..."
        run_cmd python "$PROJECT_ROOT/inference_agentic.py" \
            --model_path "$BASE_MODEL" \
            --data_path "$TEST_DATA" \
            --video_root "$VIDEO_ROOT" \
            --fps $FPS \
            --output_dir "$E5_DIR" \
            --domain "$DOMAIN" \
            --video_min_pixels $VIDEO_MIN_PIXELS \
            --video_max_pixels $VIDEO_MAX_PIXELS \
            $MAX_SAMPLES
        log_done "E5 ($DOMAIN) complete"
    fi

    # ── E5b: Agentic — trained ───────────────────────────────────────
    E5B_DIR="$EVAL_RESULTS/E5b_agentic_trained/$DOMAIN"
    if [ -f "$E5B_DIR/metrics.json" ]; then
        log_warn "E5b ($DOMAIN) already evaluated, skipping"
    elif [ -z "$AGENTIC_CHECKPOINT" ]; then
        log_error "E5b: No agentic checkpoint found. Skipping."
    else
        log_step "E5b: Agentic (trained, fine-tuned ReAct)..."
        run_cmd python "$PROJECT_ROOT/inference_agentic.py" \
            --model_path "$AGENTIC_CHECKPOINT" \
            --base_model "$BASE_MODEL" \
            --data_path "$TEST_DATA" \
            --video_root "$VIDEO_ROOT" \
            --fps $FPS \
            --output_dir "$E5B_DIR" \
            --domain "$DOMAIN" \
            --video_min_pixels $VIDEO_MIN_PIXELS \
            --video_max_pixels $VIDEO_MAX_PIXELS \
            $MAX_SAMPLES
        log_done "E5b ($DOMAIN) complete"
    fi

    # ── E5c: Agentic — SFT-only (isolates agentic training effect) ──
    # Uses the standard SFT checkpoint (E2) with the agentic inference
    # pipeline. Comparing E5c vs E5b isolates the effect of agentic
    # training data, since both models have SFT-level classification
    # ability but only E5b learned tool-use decisions from training.
    E5C_DIR="$EVAL_RESULTS/E5c_agentic_sft/$DOMAIN"
    if [ -f "$E5C_DIR/metrics.json" ]; then
        log_warn "E5c ($DOMAIN) already evaluated, skipping"
    elif [ -z "$SFT_CHECKPOINT" ]; then
        log_error "E5c: No SFT checkpoint found. Skipping."
    else
        log_step "E5c: Agentic (SFT-only, no agentic training)..."
        run_cmd python "$PROJECT_ROOT/inference_agentic.py" \
            --model_path "$SFT_CHECKPOINT" \
            --base_model "$BASE_MODEL" \
            --data_path "$TEST_DATA" \
            --video_root "$VIDEO_ROOT" \
            --fps $FPS \
            --output_dir "$E5C_DIR" \
            --domain "$DOMAIN" \
            --video_min_pixels $VIDEO_MIN_PIXELS \
            --video_max_pixels $VIDEO_MAX_PIXELS \
            $MAX_SAMPLES
        log_done "E5c ($DOMAIN) complete"
    fi
fi

# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: EVALUATION — WILD DOMAIN (Reddit)
# ═══════════════════════════════════════════════════════════════════════
if [ "$START_PHASE" -le 3 ]; then
    log_phase 3 "EVALUATION — WILD (Reddit)"

    if [ ! -f "$REDDIT_TEST_DATA" ]; then
        log_error "Reddit test data not found: $REDDIT_TEST_DATA"
        log_error "Run phase 0 first"
    else
        DOMAIN="wild"

        # ── E1: Zero-shot on wild ────────────────────────────────────
        E1W_DIR="$EVAL_RESULTS/E1_zero_shot/$DOMAIN"
        if [ -f "$E1W_DIR/metrics.json" ]; then
            log_warn "E1 ($DOMAIN) already evaluated, skipping"
        else
            log_step "E1: Zero-shot on wild data..."
            run_cmd python "$PROJECT_ROOT/eval_unified.py" \
                --model_path "$BASE_MODEL" \
                --data_path "$REDDIT_TEST_DATA" \
                --video_root "$VIDEO_ROOT" \
                --fps $FPS \
                --output_dir "$E1W_DIR" \
                --domain "$DOMAIN" \
                --video_min_pixels $VIDEO_MIN_PIXELS \
                --video_max_pixels $VIDEO_MAX_PIXELS \
                $MAX_SAMPLES
            log_done "E1 ($DOMAIN) complete"
        fi

        # ── E2: SFT on wild ─────────────────────────────────────────
        E2W_DIR="$EVAL_RESULTS/E2_sft/$DOMAIN"
        if [ -f "$E2W_DIR/metrics.json" ]; then
            log_warn "E2 ($DOMAIN) already evaluated, skipping"
        elif [ -z "$SFT_CHECKPOINT" ]; then
            log_error "E2: No SFT checkpoint found. Skipping."
        else
            log_step "E2: SFT LoRA on wild data..."
            run_cmd python "$PROJECT_ROOT/eval_unified.py" \
                --model_path "$SFT_CHECKPOINT" \
                --base_model "$BASE_MODEL" \
                --data_path "$REDDIT_TEST_DATA" \
                --video_root "$VIDEO_ROOT" \
                --fps $FPS \
                --output_dir "$E2W_DIR" \
                --domain "$DOMAIN" \
                --video_min_pixels $VIDEO_MIN_PIXELS \
                --video_max_pixels $VIDEO_MAX_PIXELS \
                $MAX_SAMPLES
            log_done "E2 ($DOMAIN) complete"
        fi

        # ── E3: Pose-only on wild ───────────────────────────────────
        E3W_DIR="$EVAL_RESULTS/E3_pose_only/$DOMAIN"
        if [ -f "$E3W_DIR/metrics.json" ]; then
            log_warn "E3 ($DOMAIN) already evaluated, skipping"
        else
            log_step "E3: Pose-only on wild data..."
            run_cmd python "$PROJECT_ROOT/eval_pose_only.py" \
                --data_path "$REDDIT_TEST_DATA" \
                --video_root "$VIDEO_ROOT" \
                --fps $FPS \
                --output_dir "$E3W_DIR" \
                --domain "$DOMAIN" \
                $MAX_SAMPLES
            log_done "E3 ($DOMAIN) complete"
        fi

        # ── E4: Always-tool on wild ─────────────────────────────────
        E4W_DIR="$EVAL_RESULTS/E4_always_tool/$DOMAIN"
        if [ -f "$E4W_DIR/metrics.json" ]; then
            log_warn "E4 ($DOMAIN) already evaluated, skipping"
        elif [ -z "$SFT_CHECKPOINT" ]; then
            log_error "E4: No SFT checkpoint found. Skipping."
        else
            log_step "E4: Always-tool on wild data..."
            run_cmd python "$PROJECT_ROOT/inference_with_pose.py" \
                --model_path "$SFT_CHECKPOINT" \
                --base_model "$BASE_MODEL" \
                --data_path "$REDDIT_TEST_DATA" \
                --video_root "$VIDEO_ROOT" \
                --fps $FPS \
                --output_dir "$E4W_DIR" \
                --domain "$DOMAIN" \
                --video_min_pixels $VIDEO_MIN_PIXELS \
                --video_max_pixels $VIDEO_MAX_PIXELS \
                $MAX_SAMPLES
            log_done "E4 ($DOMAIN) complete"
        fi

        # ── E5: Agentic untrained on wild ───────────────────────────
        E5W_DIR="$EVAL_RESULTS/E5_agentic_untrained/$DOMAIN"
        if [ -f "$E5W_DIR/metrics.json" ]; then
            log_warn "E5 ($DOMAIN) already evaluated, skipping"
        else
            log_step "E5: Agentic (untrained) on wild data..."
            run_cmd python "$PROJECT_ROOT/inference_agentic.py" \
                --model_path "$BASE_MODEL" \
                --data_path "$REDDIT_TEST_DATA" \
                --video_root "$VIDEO_ROOT" \
                --fps $FPS \
                --output_dir "$E5W_DIR" \
                --domain "$DOMAIN" \
                --video_min_pixels $VIDEO_MIN_PIXELS \
                --video_max_pixels $VIDEO_MAX_PIXELS \
                $MAX_SAMPLES
            log_done "E5 ($DOMAIN) complete"
        fi

        # ── E5b: Agentic trained on wild ────────────────────────────
        E5BW_DIR="$EVAL_RESULTS/E5b_agentic_trained/$DOMAIN"
        if [ -f "$E5BW_DIR/metrics.json" ]; then
            log_warn "E5b ($DOMAIN) already evaluated, skipping"
        elif [ -z "$AGENTIC_CHECKPOINT" ]; then
            log_error "E5b: No agentic checkpoint found. Skipping."
        else
            log_step "E5b: Agentic (trained) on wild data..."
            run_cmd python "$PROJECT_ROOT/inference_agentic.py" \
                --model_path "$AGENTIC_CHECKPOINT" \
                --base_model "$BASE_MODEL" \
                --data_path "$REDDIT_TEST_DATA" \
                --video_root "$VIDEO_ROOT" \
                --fps $FPS \
                --output_dir "$E5BW_DIR" \
                --domain "$DOMAIN" \
                --video_min_pixels $VIDEO_MIN_PIXELS \
                --video_max_pixels $VIDEO_MAX_PIXELS \
                $MAX_SAMPLES
            log_done "E5b ($DOMAIN) complete"
        fi

        # ── E5c: Agentic SFT-only on wild ────────────────────────────
        E5CW_DIR="$EVAL_RESULTS/E5c_agentic_sft/$DOMAIN"
        if [ -f "$E5CW_DIR/metrics.json" ]; then
            log_warn "E5c ($DOMAIN) already evaluated, skipping"
        elif [ -z "$SFT_CHECKPOINT" ]; then
            log_error "E5c: No SFT checkpoint found. Skipping."
        else
            log_step "E5c: Agentic (SFT-only) on wild data..."
            run_cmd python "$PROJECT_ROOT/inference_agentic.py" \
                --model_path "$SFT_CHECKPOINT" \
                --base_model "$BASE_MODEL" \
                --data_path "$REDDIT_TEST_DATA" \
                --video_root "$VIDEO_ROOT" \
                --fps $FPS \
                --output_dir "$E5CW_DIR" \
                --domain "$DOMAIN" \
                --video_min_pixels $VIDEO_MIN_PIXELS \
                --video_max_pixels $VIDEO_MAX_PIXELS \
                $MAX_SAMPLES
            log_done "E5c ($DOMAIN) complete"
        fi
    fi
fi

# ═══════════════════════════════════════════════════════════════════════
# PHASE 4: ANALYSIS + THESIS FIGURES (CPU)
# ═══════════════════════════════════════════════════════════════════════
if [ "$START_PHASE" -le 4 ]; then
    log_phase 4 "ANALYSIS + THESIS FIGURES"

    # Step 4a: Transfer gap analysis
    log_step "Computing transfer gaps (controlled vs wild)..."
    run_cmd python "$PROJECT_ROOT/eval_transfer.py" \
        --experiments_dir "$EVAL_RESULTS" \
        --output_dir "$EVAL_RESULTS/transfer_analysis"
    log_done "Transfer analysis complete"

    # Step 4b: Coaching quality evaluation (for experiments that produced coaching)
    for exp_dir in "$EVAL_RESULTS"/E4_always_tool "$EVAL_RESULTS"/E5_agentic_untrained "$EVAL_RESULTS"/E5b_agentic_trained "$EVAL_RESULTS"/E5c_agentic_sft; do
        exp_name=$(basename "$exp_dir")
        for domain_dir in "$exp_dir"/controlled "$exp_dir"/wild; do
            domain=$(basename "$domain_dir")
            pred_file="$domain_dir/predictions.jsonl"
            if [ -f "$pred_file" ]; then
                # Determine which test data to use
                if [ "$domain" = "wild" ]; then
                    coaching_data="$REDDIT_TEST_DATA"
                else
                    coaching_data="$TEST_DATA"
                fi
                coaching_out="$domain_dir/coaching_metrics"
                if [ -f "$coaching_out/coaching_metrics.json" ]; then
                    log_warn "Coaching metrics for $exp_name/$domain already exist, skipping"
                else
                    log_step "Evaluating coaching quality: $exp_name/$domain..."
                    run_cmd python "$PROJECT_ROOT/eval_coaching.py" \
                        --predictions_dir "$domain_dir" \
                        --data_path "$coaching_data" \
                        --output_dir "$coaching_out"
                    log_done "Coaching eval: $exp_name/$domain"
                fi
            fi
        done
    done

    # Step 4c: Statistical analysis (bootstrap CI + McNemar's)
    log_step "Running statistical analysis..."
    run_cmd python "$PROJECT_ROOT/eval_statistical.py" \
        --experiments_dir "$EVAL_RESULTS" \
        --output_dir "$EVAL_RESULTS/statistical_summary"
    log_done "Statistical analysis complete"

    # Step 4d: Generate thesis tables and figures
    log_step "Generating thesis tables and figures..."
    run_cmd python "$PROJECT_ROOT/thesis_tables.py" \
        --experiments_dir "$EVAL_RESULTS" \
        --output_dir "$PROJECT_ROOT/thesis_figures" \
        --train_data "$TRAIN_DATA"
    log_done "Thesis figures generated"
fi

# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  PIPELINE COMPLETE${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"
echo ""

if [ -d "$EVAL_RESULTS" ] && ! $DRY_RUN; then
    echo "Results summary:"
    for exp_dir in "$EVAL_RESULTS"/E*; do
        [ -d "$exp_dir" ] || continue
        exp_name=$(basename "$exp_dir")
        for domain in controlled wild; do
            metrics_file="$exp_dir/$domain/metrics.json"
            if [ -f "$metrics_file" ]; then
                macro_f1=$(python -c "import json; m=json.load(open('$metrics_file')); print(f\"{m['macro']['f1']:.3f}\")" 2>/dev/null || echo "?")
                n=$(python -c "import json; m=json.load(open('$metrics_file')); print(m.get('total_samples', '?'))" 2>/dev/null || echo "?")
                printf "  %-30s %-12s Macro F1 = %s  (N=%s)\n" "$exp_name" "[$domain]" "$macro_f1" "$n"
            fi
        done
    done
fi

echo ""
echo "Outputs:"
echo "  eval_results/              — Per-experiment metrics + predictions"
echo "  eval_results/transfer_analysis/   — Transfer gap analysis"
echo "  eval_results/statistical_summary/ — CIs + McNemar's tests"
echo "  thesis_figures/            — Publication-ready tables + figures"
echo ""
