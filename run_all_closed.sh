#!/usr/bin/env bash
# Run the closed models (OpenAI flagship line, Claude Sonnet line, Gemini Flash line) ALL AT THE SAME TIME,
# 10 workers each, for a given chunk of rows. Prints a progress line per model every 30s
# and a summary table (status, time, request errors, missing verdicts) at the end.
#
#   bash run_all_closed.sh 0                          # chunk 0 of the shuffled pool
#   bash run_all_closed.sh 0 --only-clear --balance   # chunk 0 of the balanced clear-only pool
#   bash run_all_closed.sh all                        # whole pool
#
# Any extra arguments after the chunk are passed straight to run_models_closed.py.
# Safe to stop (Ctrl+C) and re-run: finished items are skipped.
# Set CSV=path to use a different dataset file.

CHUNK="${1:?Usage: bash run_all_closed.sh <chunk number | all> [extra run_models_closed.py flags]}"
shift

CSV="${CSV:-dataset/dataset_generation_combined.csv}"
PYTHON="${PYTHON:-python3}"

# Same token cap for every model (reasoning models need room for hidden thinking).
COMMON=(--csv "$CSV" --max-tokens 1024 --workers 10)
if [ "$CHUNK" != "all" ]; then
  COMMON+=(--chunk "$CHUNK")
fi

EXTRA=("$@")

# provider:model[:extra flags]
MODELS=(
  # OpenAI flagship line (Azure, legacy v2 deployments)
  "azure:gpt-4o-2024-08-06"
  "azure:gpt-4-1-2025-04-14"
  "azure:gpt-5-2025-08-07:--reasoning-effort low"
  "azure:gpt-5-4-2026-03-05:--reasoning-effort low"
  # Claude Sonnet line (GPTeal)
  "gpteal:claude-sonnet-4-5-20250929-v1"
  "gpteal:claude-sonnet-4-6"
  "gpteal:claude-sonnet-5:--effort low"
  # Gemini Flash line (GPTeal); hidden thinking tokens count toward the cap, so it needs more room
  "gemini:gemini-2-5-flash:--max-tokens 4096"
  "gemini:gemini-3-flash-preview:--max-tokens 4096"
  "gemini:gemini-3-5-flash:--max-tokens 4096"
)

TOTAL=${#MODELS[@]}
LOGDIR=$(mktemp -d "${TMPDIR:-/tmp}/closed_run.XXXXXX")
START=$SECONDS

# Each model is its own provider deployment, so they run at the same time.
run_model() {
  local idx=$1 provider=$2 model=$3 flags=$4 t0=$SECONDS
  # $flags is intentionally unquoted so "--effort low" splits into two arguments.
  "$PYTHON" run_models_closed.py --provider "$provider" --model "$model" "${COMMON[@]}" $flags "${EXTRA[@]}" \
    > "$LOGDIR/$idx.log" 2>&1
  echo "$? $((SECONDS - t0))" > "$LOGDIR/$idx.done"
}

# Ctrl+C stops every model; re-running resumes where they left off.
trap 'echo; echo "Stopping all models..."; kill $(jobs -p) 2>/dev/null; wait; exit 130' INT TERM

N=0
MODEL_NAMES=()
for entry in "${MODELS[@]}"; do
  N=$((N + 1))
  IFS=: read -r provider model flags <<< "$entry"
  MODEL_NAMES+=("$model")
  run_model "$N" "$provider" "$model" "$flags" &
done
echo "Started $TOTAL models in parallel (chunk: $CHUNK). Logs: $LOGDIR"

while [ "$(ls "$LOGDIR"/*.done 2>/dev/null | wc -l)" -lt "$TOTAL" ]; do
  sleep 30
  echo "---- progress @ $((SECONDS - START))s ----"
  for i in $(seq 1 "$TOTAL"); do
    if [ -f "$LOGDIR/$i.done" ]; then state="finished"
    else state=$(tr '\r' '\n' < "$LOGDIR/$i.log" | grep -E "[0-9]+/[0-9]+ \[" | tail -1 | sed 's/^.*: *\([0-9]*%\)/\1/'); fi
    printf '[%s/%s] %-32s %s\n' "$i" "$TOTAL" "${MODEL_NAMES[$((i - 1))]}" "${state:-starting}"
  done
done
wait

SUMMARY=()
for i in $(seq 1 "$TOTAL"); do
  read -r rc secs < "$LOGDIR/$i.done"
  line=$(tr '\r' '\n' < "$LOGDIR/$i.log" | grep " finished\. " | tail -1)
  errors=$(sed -n 's/.*Request errors: \([0-9]*\).*/\1/p' <<< "$line")
  noverdict=$(sed -n 's/.*no usable verdict: \([0-9]*\).*/\1/p' <<< "$line")
  if [ "$rc" -ne 0 ]; then status="FAILED"
  elif [ -z "$line" ]; then status="DONE"  # nothing left to run, no new requests made
  elif [ "${errors:-0}" -gt 0 ]; then status="ERRORS"
  else status="OK"; fi
  SUMMARY+=("$(printf '%-3s %-32s %-7s %6ss  %8s  %10s' "$i" "${MODEL_NAMES[$((i - 1))]}" "$status" "$secs" "${errors:--}" "${noverdict:--}")")
done

echo
echo "================ Summary (chunk: $CHUNK, total $((SECONDS - START))s) ================"
printf '%-3s %-32s %-7s %7s  %8s  %10s\n' "#" "model" "status" "time" "req_errs" "no_verdict"
printf '%s\n' "${SUMMARY[@]}"
echo "For ERRORS/FAILED models, re-run the same command: only unfinished items are retried."
echo "Per-model logs: $LOGDIR"
