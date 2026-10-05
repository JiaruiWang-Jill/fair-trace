#!/bin/bash
# Analyse the qwen MovieLens-1M audit and publish the result, unattended.
# Runs once at the deadline if the audit is still going, and again when it finishes.
cd "$(dirname "$0")/.." || exit 1
PID=$(cat .ml1m_qwen_pid 2>/dev/null)
DEADLINE=$(date -v7H -v30M +%s)

publish () {
  local note="$1"
  python3 scripts/analyze_ml1m_qwen.py > ml1m_qwen_analyze.log 2>&1 || return 1
  git add fair_trace_outputs_ml1m_qwen/RESULTS.md \
          fair_trace_outputs_ml1m_qwen/*.png \
          fair_trace_outputs_ml1m_qwen/*.csv \
          fair_trace_outputs_ml1m_qwen/run_manifest.json 2>/dev/null
  git commit -q -m "Publish qwen MovieLens-1M audit results ($note)

Written automatically when the overnight run reached this point. RESULTS.md
states how many users completed and whether the run had finished.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" 2>/dev/null \
    && git push -q origin main 2>/dev/null && echo "published: $note"
}

while kill -0 "$PID" 2>/dev/null; do
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    publish "partial, at the 07:30 deadline"
    break
  fi
  sleep 120
done

# wait for the audit to exit, then publish the complete result
while kill -0 "$PID" 2>/dev/null; do sleep 120; done
publish "complete run"
