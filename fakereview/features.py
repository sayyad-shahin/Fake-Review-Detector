"""Hand-made writing-style features. Fake reviews often differ from real ones in HOW they are written."""
import re
import numpy as np

SUPERLATIVES = {"best", "perfect", "amazing", "awesome", "incredible", "fantastic", "excellent", "worst",
                "terrible", "horrible", "unbelievable", "greatest", "flawless", "outstanding", "superb",
                "wonderful", "phenomenal", "garbage", "scam", "ever"}
CALL_TO_ACTION = {"buy", "recommend", "recommended", "must", "everyone", "anyone", "avoid", "waste",
                  "highly", "definitely", "guaranteed", "regret"}
FIRST_PERSON = {"i", "me", "my", "mine", "we", "our", "us", "i'm", "i've", "i'd"}
HEDGES = {"but", "however", "though", "although", "slightly", "bit", "only", "wish", "except",
          "sometimes", "little", "fairly", "pretty", "okay", "ok", "decent"}

FEATURE_NAMES = [
    "review length", "word count", "average word length", "exclamation marks", "ALL-CAPS words",
    "repeated punctuation", "extreme words", "call-to-action words", "personal experience words",
    "balanced / hedging words", "numbers and measurements", "vocabulary variety", "sentence count",
]


def _stats(text):
    words = re.findall(r"[A-Za-z']+", text)
    lw = [w.lower() for w in words]
    nw = max(len(words), 1)
    return {
        "chars": len(text), "nw": nw, "words": words, "lw": lw,
        "sentences": max(len(re.findall(r"[.!?]+", text)), 1),
        "excl": text.count("!"),
        "caps": sum(1 for w in words if len(w) >= 3 and w.isupper()),
        "rep": len(re.findall(r"([!?.])\1{1,}", text)),
        "sup": sum(w in SUPERLATIVES for w in lw),
        "cta": sum(w in CALL_TO_ACTION for w in lw),
        "fp": sum(w in FIRST_PERSON for w in lw),
        "hedge": sum(w in HEDGES for w in lw),
        "nums": len(re.findall(r"\d+", text)),
        "variety": len(set(lw)) / nw,
    }


def style_features(texts):
    """Turn a list of reviews into a (n_reviews, 13) numeric matrix."""
    rows = []
    for t in texts:
        s = _stats(str(t))
        rows.append([
            np.log1p(s["chars"]), np.log1p(s["nw"]), np.mean([len(w) for w in s["words"]]) if s["words"] else 0.0,
            min(s["excl"], 10), s["caps"] / s["nw"], min(s["rep"], 5), s["sup"] / s["nw"], s["cta"] / s["nw"],
            s["fp"] / s["nw"], s["hedge"] / s["nw"], min(s["nums"], 5), s["variety"], np.log1p(s["sentences"]),
        ])
    return np.array(rows, dtype=float)


def red_flags(text):
    """Plain-English warning signs and trust signals found in a review."""
    s = _stats(text)
    flags, good = [], []
    if s["excl"] >= 3:
        flags.append(f"Uses many exclamation marks ({s['excl']})")
    if s["caps"] >= 2:
        flags.append("Several words written in ALL CAPS")
    if s["sup"] >= 3 or (s["nw"] > 0 and s["sup"] / s["nw"] > 0.12):
        flags.append("Heavy use of extreme words (best, perfect, amazing, worst...)")
    if s["cta"] >= 2:
        flags.append("Pushy call-to-action language (buy, must, recommend to everyone...)")
    if s["nums"] == 0 and s["nw"] >= 8:
        flags.append("No concrete details such as numbers, times or measurements")
    if s["nw"] < 6:
        flags.append("Very short review with little information")
    if s["fp"] == 0 and s["nw"] >= 10:
        flags.append("Never mentions personal experience (I, my, we...)")
    if s["nw"] >= 15 and s["variety"] < 0.6:
        flags.append("Repetitive wording")
    if s["nums"] >= 1:
        good.append("Mentions specific numbers or measurements")
    if s["hedge"] >= 2:
        good.append("Balanced tone, mentions both good and bad points")
    if s["fp"] >= 2 and s["nw"] >= 15:
        good.append("Describes personal experience")
    return flags, good
