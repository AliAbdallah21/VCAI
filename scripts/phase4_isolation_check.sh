#!/usr/bin/env bash
# scripts/phase4_isolation_check.sh
#
# Proves the Phase 4 tenant-isolation + quota + gating cases against a running
# backend (default http://localhost:8000). Onboards two companies, creates an
# agent in each, and exercises the cross-tenant / quota / gating boundaries.
#
# Usage:
#   BASE=http://localhost:8000 bash scripts/phase4_isolation_check.sh
#
# Requires: curl, jq. Assumes seed_plans.py and seed_personas.py have run.
set -euo pipefail

BASE="${BASE:-http://localhost:8000}/api"
JQ() { jq -r "$1"; }
RND="$RANDOM$RANDOM"

echo "== Onboard Company A (scale, unlimited) =="
A=$(curl -s -X POST "$BASE/onboarding/signup" -H 'Content-Type: application/json' -d "{
  \"company_name\":\"CoA-$RND\",\"plan_name\":\"scale\",\"billing_cycle\":\"monthly\",
  \"manager_name\":\"Mgr A\",\"manager_email\":\"mgrA-$RND@example.com\",\"password\":\"pw-$RND-aaaa\"}")
A_TOKEN=$(echo "$A" | JQ '.token // .access_token')
echo "  manager A token acquired"

echo "== Onboard Company B (free) =="
B=$(curl -s -X POST "$BASE/onboarding/signup" -H 'Content-Type: application/json' -d "{
  \"company_name\":\"CoB-$RND\",\"plan_name\":\"free\",\"billing_cycle\":\"monthly\",
  \"manager_name\":\"Mgr B\",\"manager_email\":\"mgrB-$RND@example.com\",\"password\":\"pw-$RND-bbbb\"}")
B_TOKEN=$(echo "$B" | JQ '.token // .access_token')
echo "  manager B token acquired"

AUTH_A=(-H "Authorization: Bearer $A_TOKEN")
AUTH_B=(-H "Authorization: Bearer $B_TOKEN")

echo
echo "== [A] create a session (scale: allowed, any persona) =="
SA=$(curl -s -X POST "$BASE/sessions" "${AUTH_A[@]}" -H 'Content-Type: application/json' \
  -d '{"persona_id":"first_time_buyer","difficulty":"easy"}')
SA_ID=$(echo "$SA" | JQ '.id')
echo "  A session id: $SA_ID"

echo
echo "== [B] read A's session -> expect 404 (cross-tenant) =="
curl -s -o /dev/null -w "  status=%{http_code}\n" "$BASE/sessions/$SA_ID" "${AUTH_B[@]}"

echo
echo "== [B] free plan: start gated (hard) persona -> expect 403 =="
curl -s -o /dev/null -w "  status=%{http_code}\n" -X POST "$BASE/sessions" "${AUTH_B[@]}" \
  -H 'Content-Type: application/json' -d '{"persona_id":"tough_negotiator","difficulty":"hard"}'

echo
echo "== [B] free plan: start Easy persona -> expect 201 =="
curl -s -o /dev/null -w "  status=%{http_code}\n" -X POST "$BASE/sessions" "${AUTH_B[@]}" \
  -H 'Content-Type: application/json' -d '{"persona_id":"first_time_buyer","difficulty":"easy"}'

echo
echo "== [B] persona list shows locked flags (free plan) =="
curl -s "$BASE/personas" "${AUTH_B[@]}" | jq '[.personas[] | {id, locked}]'

echo
echo "== [B] exhaust the free monthly cap (5) then expect 429 =="
for i in 1 2 3 4 5 6; do
  CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/sessions" "${AUTH_B[@]}" \
    -H 'Content-Type: application/json' -d '{"persona_id":"friendly_customer","difficulty":"easy"}')
  echo "  attempt $i -> $CODE"
done
echo "  (last attempt should be 429 Monthly session limit reached)"

echo
echo "Done. Expected: cross-tenant=404, gated=403, easy=201, cap final=429."
