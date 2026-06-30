"""
scoring.py — combine the detection signals into one calibrated score, classify it,
and render the transparency label.

Canonical numbers:
  ai_likelihood : combined, disagreement-adjusted P(AI) in [0,1]  (the real score)
  confidence    : max(p, 1-p) -> probability of the leading class, in [0.5,1.0]
                  (refined from planning.md's |p-0.5|*2; see README spec reflection —
                  this reads naturally for users, e.g. "74% likely AI", and matches the
                  project's own 0.51-vs-0.95 confidence framing)

Thresholds are ASYMMETRIC by design: it takes more evidence to call something AI
(>=0.72) than to call it human (<=0.35), because a false AI accusation is the
expensive error on a writing platform.

Ensemble (stretch S1): combine_signals takes a dict of PARTICIPATING signals and
weights them. When the repetition signal abstains (short text), only llm+stylometric
remain and the renormalized weights become ~0.65/0.35 — the original two-signal system.
"""

# Ensemble weights — the LLM is the most capable single detector, so it leads.
# Renormalized over whichever signals participate (see combine_signals).
SIGNAL_WEIGHTS = {"llm": 0.55, "stylometric": 0.30, "repetition": 0.15}

# How hard signal disagreement pulls the result back toward 0.5 (uncertain).
DISAGREEMENT_K = 0.5

# Asymmetric decision thresholds on ai_likelihood.
THRESHOLD_AI = 0.72      # high bar to accuse
THRESHOLD_HUMAN = 0.35   # generous benefit of the doubt


def combine_signals(signal_scores: dict) -> tuple[float, float]:
    """
    Weighted ensemble over participating signals, then shrink toward 0.5 by their spread.

    `signal_scores` maps signal name -> P(AI), including ONLY signals that participated
    (the repetition signal abstains on short text and is left out by the caller). Weights
    come from SIGNAL_WEIGHTS, renormalized over the participants — so two-signal calls
    (~0.65/0.35) reproduce the original validated behavior exactly.

    Returns (ai_likelihood, disagreement), both rounded.
    """
    weights = {k: SIGNAL_WEIGHTS[k] for k in signal_scores}
    total_w = sum(weights.values()) or 1.0
    raw = sum(signal_scores[k] * weights[k] for k in signal_scores) / total_w

    scores = list(signal_scores.values())
    disagreement = (max(scores) - min(scores)) if len(scores) > 1 else 0.0

    ai_likelihood = 0.5 + (raw - 0.5) * (1 - DISAGREEMENT_K * disagreement)
    ai_likelihood = max(0.0, min(1.0, ai_likelihood))
    return round(ai_likelihood, 3), round(disagreement, 3)


def classify(ai_likelihood: float) -> tuple[str, float]:
    """
    Map ai_likelihood to (attribution, confidence) using the asymmetric thresholds.

    confidence = probability of the leading class = max(p, 1-p), in [0.5, 1.0],
    so it reads as "X% likely <attribution>" for a non-technical user.
    """
    if ai_likelihood >= THRESHOLD_AI:
        attribution = "likely_ai"
    elif ai_likelihood <= THRESHOLD_HUMAN:
        attribution = "likely_human"
    else:
        attribution = "uncertain"
    confidence = round(max(ai_likelihood, 1 - ai_likelihood), 2)
    return attribution, confidence


def participating_weights(signal_scores: dict) -> dict:
    """The renormalized weight actually applied to each participating signal (for transparency)."""
    weights = {k: SIGNAL_WEIGHTS[k] for k in signal_scores}
    total = sum(weights.values()) or 1.0
    return {k: round(w / total, 3) for k, w in weights.items()}


def generate_label(attribution: str, confidence: float, creator_verified: bool = False) -> dict:
    """
    Render the reader-facing transparency label for a verdict. One of three variants.

    Returns {"variant", "title", "body", "creator_badge"}. The confidence percentage is
    injected into the two confident variants; the uncertain variant deliberately shows no
    percentage and makes no AI-generation claim. `creator_verified` (stretch S2) attaches a
    "Verified human creator" badge.
    """
    pct = round(confidence * 100)

    if attribution == "likely_ai":
        label = {
            "variant": "high_confidence_ai",
            "title": "🤖 Likely AI-generated",
            "body": (
                "Our automated analysis suggests this text was probably produced with "
                "significant help from an AI writing tool. This is an estimate from "
                "imperfect detection signals, not a verdict on the author — detectors can "
                f"and do get this wrong. Estimated confidence: {pct}%. If you wrote this "
                "yourself, you can appeal this label and a human will review it."
            ),
        }
    elif attribution == "likely_human":
        label = {
            "variant": "high_confidence_human",
            "title": "✍️ Likely human-written",
            "body": (
                "Our automated analysis found no strong signs of AI generation. This text "
                "reads as original human writing. This is an automated estimate, not a "
                f"guarantee of authorship. Estimated confidence: {pct}%."
            ),
        }
    else:
        label = {
            "variant": "uncertain",
            "title": "❓ Authorship uncertain",
            "body": (
                "Our automated analysis could not reliably determine whether this text was "
                "written by a person or generated by AI — the signals we use disagree or are "
                "too weak to make a call. We are telling you this honestly rather than "
                "guessing. No AI-generation claim is being made about this work."
            ),
        }

    label["creator_badge"] = "✅ Verified human creator" if creator_verified else None
    return label
