"""
app.py — Provenance Guard Flask API.

Endpoints:
  POST /submit         classify a text submission   (rate-limited 10/min;100/day)
  POST /appeal         contest a classification      (rate-limited 5/min;30/day)
  GET  /log            recent audit-log entries (JSON)
  GET  /review-queue   content awaiting human review
  GET  /health         liveness check

Full build: two detection signals, calibrated confidence scoring, three transparency
label variants, appeals workflow, per-IP rate limiting, and a structured SQLite audit log.
"""

import os
import secrets
import uuid

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

load_dotenv()  # app.py sits in the repo root, so this finds ./.env and loads GROQ_API_KEY

import audit
import scoring
import signals

app = Flask(__name__)
audit.init_db()

# Rate limiting, keyed per client IP. In-memory storage is sufficient for a single
# local/dev process; a real deployment would point storage_uri at Redis so the limit
# is shared across workers. Limits and reasoning are documented in the README.
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)


@app.post("/submit")
@limiter.limit("10 per minute;100 per day")
def submit():
    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()
    creator_id = (body.get("creator_id") or "").strip()

    if not text:
        return jsonify({"error": "Field 'text' is required."}), 400
    if not creator_id:
        return jsonify({"error": "Field 'creator_id' is required."}), 400

    content_id = str(uuid.uuid4())
    content_type = (body.get("content_type") or "text").strip() or "text"

    # --- Signal 1: LLM (Groq) — semantic, modality-aware (text | image_caption) ---
    llm = signals.llm_signal(text, content_type=content_type)
    llm_score = llm["ai_probability"]

    # --- Signal 2: stylometric heuristics — structural ---
    stylo_score, stylo_detail = signals.stylometric_signal(text)

    # --- Signal 3: repetition / redundancy — abstains on short text (ensemble S1) ---
    rep_score, rep_detail = signals.repetition_signal(text)

    # --- Ensemble: combine the participating signals, then classify ---
    signal_scores = {"llm": llm_score, "stylometric": stylo_score}
    if rep_detail["participates"]:
        signal_scores["repetition"] = rep_score
    ai_likelihood, disagreement = scoring.combine_signals(signal_scores)
    attribution, confidence = scoring.classify(ai_likelihood)

    # --- Transparency label (one of three reader-facing variants) ---
    # Verified-human creators (stretch S2) get a badge prepended to the label.
    credential = audit.get_credential(creator_id)
    label = scoring.generate_label(attribution, confidence, creator_verified=credential is not None)

    audit.record_submission(
        {
            "content_id": content_id,
            "creator_id": creator_id,
            "text": text,
            "content_type": content_type,
            "attribution": attribution,
            "confidence": confidence,
            "ai_likelihood": ai_likelihood,
            "llm_score": llm_score,
            "stylometric_score": stylo_score,
            "stylometric_detail": stylo_detail,
            "repetition_score": rep_score if rep_detail["participates"] else None,
            "label_variant": label["variant"],
        }
    )

    return jsonify(
        {
            "content_id": content_id,
            "content_type": content_type,
            "attribution": attribution,
            "confidence": confidence,
            "ai_likelihood": ai_likelihood,
            "signals": {
                "llm_score": llm_score,
                "llm_reasoning": llm["reasoning"],
                "llm_indicators": llm["indicators"],
                "stylometric_score": stylo_score,
                "stylometric_detail": stylo_detail,
                "repetition_score": rep_score,
                "repetition_detail": rep_detail,
                "disagreement": disagreement,
                "ensemble_weights": scoring.participating_weights(signal_scores),
            },
            "creator": {
                "creator_id": creator_id,
                "verified_human": credential is not None,
                "credential": credential,
            },
            "label": label,
            "status": "classified",
        }
    )


@app.post("/appeal")
@limiter.limit("5 per minute;30 per day")
def appeal():
    body = request.get_json(silent=True) or {}
    content_id = (body.get("content_id") or "").strip()
    creator_reasoning = (body.get("creator_reasoning") or "").strip()

    if not content_id:
        return jsonify({"error": "Field 'content_id' is required."}), 400
    if not creator_reasoning:
        return jsonify({"error": "Field 'creator_reasoning' is required."}), 400

    result = audit.record_appeal(content_id, creator_reasoning)
    if result is None:
        return jsonify({"error": f"No content found with id '{content_id}'."}), 404

    return jsonify(
        {
            "content_id": content_id,
            "status": "under_review",
            "appeal_id": result["appeal_id"],
            "message": "Appeal received. This content is now under review by a human moderator.",
            "original_decision": result["original_decision"],
        }
    )


@app.get("/log")
def get_log():
    limit = request.args.get("limit", default=20, type=int)
    return jsonify({"entries": audit.get_recent_entries(limit)})


@app.get("/review-queue")
def review_queue():
    """What a human reviewer opens: content currently awaiting review."""
    return jsonify({"queue": audit.get_review_queue()})


@app.get("/content/<content_id>")
def content_detail(content_id):
    """Full stored record for one submission."""
    record = audit.get_content(content_id)
    if record is None:
        return jsonify({"error": f"No content found with id '{content_id}'."}), 404
    return jsonify(record)


# --- stretch S2: verified-human credential ---

_VERIFY_PROMPTS = (
    "Describe a meal you cooked recently and how it actually turned out.",
    "Write about a place you visited that surprised you, and why.",
    "Tell us about a small thing that annoyed you this week.",
    "Describe your morning routine in your own words.",
    "Write about a hobby you've picked up or dropped lately.",
)


@app.post("/verify-human/challenge")
@limiter.limit("10 per minute;50 per day")
def verify_challenge():
    """Issue a one-time writing challenge a creator must answer to earn the credential."""
    body = request.get_json(silent=True) or {}
    creator_id = (body.get("creator_id") or "").strip()
    if not creator_id:
        return jsonify({"error": "Field 'creator_id' is required."}), 400

    challenge_id = str(uuid.uuid4())
    token = "PG-" + secrets.token_hex(4).upper()
    prompt = secrets.choice(_VERIFY_PROMPTS)
    audit.create_challenge(challenge_id, creator_id, prompt, token)
    return jsonify(
        {
            "challenge_id": challenge_id,
            "creator_id": creator_id,
            "prompt": prompt,
            "token": token,
            "instructions": (
                f"Write at least 40 words of original prose answering the prompt and include "
                f"the exact token {token} somewhere in it. POST it to /verify-human as "
                f"'response_text' with this 'challenge_id' and your 'creator_id'."
            ),
        }
    )


@app.post("/verify-human")
@limiter.limit("10 per minute;50 per day")
def verify_human():
    """Validate a challenge response and grant the verified-human credential if it passes."""
    body = request.get_json(silent=True) or {}
    creator_id = (body.get("creator_id") or "").strip()
    challenge_id = (body.get("challenge_id") or "").strip()
    response_text = (body.get("response_text") or "").strip()
    if not (creator_id and challenge_id and response_text):
        return jsonify({"error": "Fields 'creator_id', 'challenge_id', 'response_text' are required."}), 400

    challenge = audit.get_challenge(challenge_id)
    if challenge is None:
        return jsonify({"error": "Unknown challenge_id."}), 404
    if challenge["creator_id"] != creator_id:
        return jsonify({"error": "This challenge was issued to a different creator."}), 403
    if challenge["status"] != "issued":
        return jsonify({"error": "This challenge has already been used."}), 409

    reasons = []
    if challenge["token"] not in response_text:
        reasons.append("response is missing the one-time token from the challenge")
    word_count = len(response_text.split())
    if word_count < 40:
        reasons.append(f"response too short ({word_count} words; need at least 40)")

    # the response must itself read as clearly human
    llm = signals.llm_signal(response_text)
    stylo_score, _ = signals.stylometric_signal(response_text)
    rep_score, rep_detail = signals.repetition_signal(response_text)
    scores = {"llm": llm["ai_probability"], "stylometric": stylo_score}
    if rep_detail["participates"]:
        scores["repetition"] = rep_score
    ai_likelihood, _ = scoring.combine_signals(scores)
    attribution, _conf = scoring.classify(ai_likelihood)
    if attribution != "likely_human":
        reasons.append(
            f"writing did not read as clearly human (attribution={attribution}, "
            f"ai_likelihood={ai_likelihood})"
        )

    if reasons:
        return (
            jsonify(
                {
                    "verified": False,
                    "reasons": reasons,
                    "detector": {"attribution": attribution, "ai_likelihood": ai_likelihood},
                }
            ),
            422,
        )

    audit.mark_challenge_used(challenge_id)
    credential_id = "VH-" + uuid.uuid4().hex[:12].upper()
    credential = audit.grant_credential(creator_id, credential_id, method="live_writing_challenge")
    return jsonify(
        {
            "verified": True,
            "creator_id": creator_id,
            "credential": credential,
            "detector": {"attribution": attribution, "ai_likelihood": ai_likelihood},
            "message": "Verified-human credential granted; it will now appear on this creator's submissions.",
        }
    )


# --- stretch S3: analytics dashboard ---

@app.get("/analytics")
def analytics():
    """Aggregate detection patterns, appeal rates, and signal-agreement metrics (JSON)."""
    return jsonify(audit.get_analytics())


@app.get("/dashboard")
def dashboard():
    """Minimal HTML view of the analytics for the demo (fetches /analytics)."""
    return _DASHBOARD_HTML


_DASHBOARD_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Provenance Guard — Analytics</title>
<style>
 body{font-family:-apple-system,system-ui,sans-serif;margin:2rem;background:#0f172a;color:#e2e8f0}
 h1{font-weight:650;margin-bottom:.2rem}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:1rem;margin-top:1.2rem}
 .card{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:1rem 1.2rem}
 .card h2{font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:#94a3b8;margin:0 0 .5rem}
 .big{font-size:2rem;font-weight:700}
 .row{display:flex;justify-content:space-between;font-size:.9rem}
 .bar{height:8px;border-radius:4px;background:#334155;overflow:hidden;margin-top:.35rem}
 .bar>span{display:block;height:100%}
 .ai{background:#f87171}.human{background:#4ade80}.unc{background:#fbbf24}
 small{color:#94a3b8}
</style></head><body>
<h1>🛡️ Provenance Guard — Analytics</h1>
<small>Live view of <code>/analytics</code> — refresh to update.</small>
<div id="app" class="grid">loading…</div>
<script>
fetch('/analytics').then(r=>r.json()).then(d=>{
 const b=d.attribution_breakdown;
 const pb=(c,o)=>`<div class=row><span>${o.count}</span><span>${o.percent}%</span></div><div class=bar><span class=${c} style="width:${o.percent}%"></span></div>`;
 document.getElementById('app').innerHTML=`
  <div class=card><h2>Total submissions</h2><div class=big>${d.total_submissions}</div></div>
  <div class=card><h2>Likely AI</h2>${pb('ai',b.likely_ai)}</div>
  <div class=card><h2>Likely human</h2>${pb('human',b.likely_human)}</div>
  <div class=card><h2>Uncertain</h2>${pb('unc',b.uncertain)}</div>
  <div class=card><h2>Appeal rate</h2><div class=big>${(d.appeal_rate*100).toFixed(0)}%</div><small>${d.appeals.total} appeal(s)</small></div>
  <div class=card><h2>Avg confidence</h2><div class=big>${d.average_confidence}</div></div>
  <div class=card><h2>Avg signal disagreement</h2><div class=big>${d.average_signal_disagreement}</div><small>how often signals conflict</small></div>
  <div class=card><h2>Verified-human creators</h2><div class=big>${d.verified_human_creators}</div></div>`;
}).catch(e=>{document.getElementById('app').textContent='Error loading analytics: '+e;});
</script></body></html>"""


@app.get("/")
def index():
    """Self-documenting landing route so the root isn't a bare 404 in a browser."""
    return jsonify(
        {
            "service": "Provenance Guard",
            "description": "AI-vs-human text attribution with confidence scoring, "
            "transparency labels, appeals, and a structured audit log.",
            "endpoints": {
                "POST /submit": "classify a submission — body {text, creator_id, content_type?}",
                "POST /appeal": "contest a classification — body {content_id, creator_reasoning}",
                "POST /verify-human/challenge": "request a verified-human writing challenge — body {creator_id}",
                "POST /verify-human": "claim the credential — body {creator_id, challenge_id, response_text}",
                "GET /log": "recent audit-log entries — optional ?limit=N",
                "GET /review-queue": "content currently awaiting human review",
                "GET /content/<id>": "full record for one submission",
                "GET /analytics": "aggregate detection / appeal / signal metrics (JSON)",
                "GET /dashboard": "minimal HTML analytics view",
                "GET /health": "liveness check",
            },
            "content_types": ["text", "image_caption"],
            "note": "POST endpoints need curl or Postman, not a browser address bar.",
        }
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=True, use_reloader=False)
