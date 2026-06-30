"""
calibrate.py — run the two signals + scoring directly on a fixed battery of inputs
that span the confidence range, and print a compact table.

This is the Milestone 4 calibration check (and the evidence reproduced in the README).
It calls Groq directly, so it needs GROQ_API_KEY in .env; it does NOT need the Flask
server running.

Usage:  python calibrate.py
"""

import os

from dotenv import load_dotenv

load_dotenv()

import scoring
import signals

CASES = {
    "clear_AI": (
        "Artificial intelligence represents a transformative paradigm shift in modern "
        "society. It is important to note that while the benefits of AI are numerous, it "
        "is equally essential to consider the ethical implications. Furthermore, "
        "stakeholders across various sectors must collaborate to ensure responsible "
        "deployment."
    ),
    "clear_human": (
        "ok so i finally tried that new ramen place downtown and honestly? underwhelming. "
        "the broth was fine but they put WAY too much sodium in it and i was thirsty for "
        "like three hours after. my friend got the spicy version and said it was better. "
        "probably won't go back unless someone drags me there"
    ),
    "borderline_formal_human": (
        "The relationship between monetary policy and asset price inflation has been "
        "extensively studied in the literature. Central banks face a fundamental tension "
        "between their mandate for price stability and the unintended consequences of "
        "prolonged low interest rates on equity and real estate valuations."
    ),
    "borderline_edited_AI": (
        "I've been thinking a lot about remote work lately. There are genuine tradeoffs — "
        "flexibility and no commute on one side, isolation and blurred work-life boundaries "
        "on the other. Studies show productivity varies widely by individual and role type."
    ),
}

HEADER = (
    f"{'case':<24}{'llm':>6}{'stylo':>7}{'rep':>7}{'disagree':>10}"
    f"{'ai_like':>9}{'conf':>7}  attribution"
)


def main() -> None:
    print(HEADER)
    print("-" * len(HEADER))
    for name, text in CASES.items():
        llm = signals.llm_signal(text)
        llm_score = llm["ai_probability"]
        stylo_score, _ = signals.stylometric_signal(text)
        rep_score, rep_detail = signals.repetition_signal(text)

        # Build the ensemble from participating signals; repetition abstains on short text.
        scores = {"llm": llm_score, "stylometric": stylo_score}
        if rep_detail["participates"]:
            scores["repetition"] = rep_score
        rep_cell = f"{rep_score:.2f}" if rep_detail["participates"] else "—"

        ai_like, disagree = scoring.combine_signals(scores)
        attribution, conf = scoring.classify(ai_like)
        print(
            f"{name:<24}{llm_score:>6.2f}{stylo_score:>7.2f}{rep_cell:>7}{disagree:>10.2f}"
            f"{ai_like:>9.2f}{conf:>7.2f}  {attribution}"
        )


if __name__ == "__main__":
    main()
