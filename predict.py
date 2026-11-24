"""Check reviews from the command line.
   python predict.py "BEST product EVER!!! Buy now!!!"
   python predict.py --csv reviews.csv --column text --out results.csv
   python predict.py                      (interactive mode)
"""
import argparse, csv, sys
from fakereview import load_or_train


def show(d):
    print(f"\n  Verdict: {d['verdict']}  (fake probability {d['fake_probability'] * 100:.1f}%)")
    for title, key in (("Warning signs", "warning_signs"), ("Trust signs", "trust_signs"),
                       ("Words pointing to fake", "words_pointing_to_fake"),
                       ("Words pointing to genuine", "words_pointing_to_genuine")):
        if d[key]:
            print(f"  {title}: " + ("; " if "signs" in key else ", ").join(d[key]))
    print()


ap = argparse.ArgumentParser()
ap.add_argument("text", nargs="?", help="review text to check")
ap.add_argument("--csv", help="CSV file with a column of reviews")
ap.add_argument("--column", default="text", help="name of the review column in the CSV")
ap.add_argument("--out", default="predictions.csv", help="where to write CSV results")
a = ap.parse_args()
det = load_or_train()

if a.csv:
    with open(a.csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows or a.column not in rows[0]:
        sys.exit(f"Column '{a.column}' not found. Available: {list(rows[0]) if rows else 'none'}")
    results = det.analyze_many([r[a.column] for r in rows])
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([*rows[0].keys(), "fake_probability", "verdict"])
        for r, (p, v) in zip(rows, results):
            w.writerow([*r.values(), p, v])
    fakes = sum(v == "Likely fake" for _, v in results)
    print(f"Checked {len(rows)} reviews: {fakes} likely fake. Results saved to {a.out}")
elif a.text:
    show(det.analyze(a.text))
else:
    print("Type or paste a review and press Enter (empty line to quit).")
    while True:
        t = input("> ").strip()
        if not t:
            break
        show(det.analyze(t))
