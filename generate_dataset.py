"""Create a SAMPLE labelled dataset (data/reviews.csv) so the project works out of the box.

label 1 = fake review, label 0 = genuine review.
The reviews are synthetic. For real use, replace data/reviews.csv with a real labelled dataset
(columns: text,label) and run `python train.py`.
"""
import argparse, csv, os, random

PRODUCTS = ["wireless headphones", "blender", "phone case", "hotel room", "laptop", "coffee maker",
            "backpack", "vacuum cleaner", "running shoes", "smart watch", "air fryer", "desk lamp",
            "water bottle", "bluetooth speaker", "mattress", "restaurant meal"]
ASPECTS = ["battery", "sound quality", "screen", "build quality", "packaging", "customer service", "setup",
           "handle", "fit", "taste", "noise level", "price", "delivery", "design", "cleaning", "strap"]

OPENERS = ["I bought this {p} about {n} weeks ago.", "Been using the {p} for {n} days now.",
           "Picked up this {p} after my old one stopped working.",
           "Got the {p} to replace the one I had for {n} years.", "Ordered the {p} last month and it arrived in {d} days.",
           "We tried this {p} on a trip last weekend.", "My partner gave me this {p} and I have used it daily since."]
DETAIL_POS = ["The {a} is really good, especially for what I paid.", "The {a} works as described and the {a2} is better than I expected.",
              "It took me about {n} minutes to set up and the instructions were clear.",
              "The {a} lasts around {n} hours which is plenty for me.", "My kids use it every day and so far nothing has broken.",
              "The {a} feels solid and the {a2} is nicely done."]
DETAIL_NEG = ["The {a} is a bit disappointing though.", "Sadly the {a} stopped working after {n} days.",
              "The {a} feels cheap and the {a2} could be better.", "I wish the {a} was a little better, it is only okay.",
              "Customer service took {n} days to answer my email.", "The {a} is noisy and the {a2} is confusing."]
CLOSE_POS = ["Overall a solid buy for the price.", "Would buy again, but I wish it was slightly cheaper.",
             "Not perfect, but it does the job well.", "Happy with it so far, will update if anything changes."]
CLOSE_NEG = ["I ended up returning it.", "Not worth the money in my opinion.", "I probably will not buy again.",
             "Two stars because the {a} still works."]
HYPE = ["Love it!", "Great product.", "Really happy with it.", "Works great."]  # genuine reviews sometimes sound happy too

FAKE_POS = ["BEST {p} EVER!!! Amazing quality and perfect in every way. Highly recommend to everyone!!!",
            "This {p} is absolutely amazing! I love it so much. Five stars, you will not regret buying it.",
            "Perfect product, perfect price, perfect seller. A+++ Must buy!",
            "Great {p}! Fast shipping and excellent quality. Best purchase I have ever made. Highly recommended!",
            "I was skeptical at first but this {p} is incredible, it changed my life. Everyone needs one, buy it now!",
            "Wow wow wow! Unbelievable {p}, super amazing, works great, love love love it!!",
            "Excellent product, exceeded all my expectations. Would give 10 stars if I could. Buy now!"]
FAKE_SUBTLE = ["I recently purchased this {p} and I am extremely satisfied. The quality is outstanding and it works flawlessly. I would highly recommend it to anyone looking for a great {p}. Five stars!",
               "After using this {p} I can say it is the perfect choice. Great quality, great value, great service. Definitely worth every penny, I recommend it to all my friends and family.",
               "This {p} is a game changer. Superb quality, excellent design and fantastic performance. You need this in your life, highly recommended to everybody."]
FAKE_NEG = ["Terrible {p}. Complete waste of money, do not buy! Worst product ever, total scam.",
            "AVOID this {p}!!! It is garbage and the seller is a thief. Save your money and buy another brand.",
            "Horrible quality, absolutely the worst {p} I have ever seen. Do not waste your time."]


def fill(t, r):
    a, a2 = r.sample(ASPECTS, 2)
    return t.format(p=r.choice(PRODUCTS), n=r.randint(2, 40), d=r.randint(2, 9), a=a, a2=a2, A=a.capitalize())


def genuine(r):
    mood = r.random()
    p = r.choice(PRODUCTS)
    parts = [fill(r.choice(OPENERS), r).replace("{p}", p)]
    if mood < 0.55:
        parts += [fill(x, r) for x in r.sample(DETAIL_POS, r.randint(1, 2))] + [fill(r.choice(CLOSE_POS), r)]
    elif mood < 0.8:
        parts += [fill(r.choice(DETAIL_POS), r), fill(r.choice(DETAIL_NEG), r), fill(r.choice(CLOSE_POS), r)]
    else:
        parts += [fill(x, r) for x in r.sample(DETAIL_NEG, r.randint(1, 2))] + [fill(r.choice(CLOSE_NEG), r)]
    if r.random() < 0.15:
        parts.append(r.choice(HYPE))
    return " ".join(parts)


def fake(r):
    k = r.random()
    pool = FAKE_POS if k < 0.55 else FAKE_SUBTLE if k < 0.8 else FAKE_NEG
    text = fill(r.choice(pool), r)
    if r.random() < 0.15:  # sometimes fakers add a detail to look real
        text += " " + fill(r.choice(DETAIL_POS), r)
    if r.random() < 0.25:  # sometimes they write calmly
        text = text.replace("!!!", ".").replace("!", ".")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--noise", type=float, default=0.03, help="fraction of labels flipped, like real-world mistakes")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/reviews.csv")
    a = ap.parse_args()
    r = random.Random(a.seed)
    rows = [(genuine(r), 0) for _ in range(a.n // 2)] + [(fake(r), 1) for _ in range(a.n - a.n // 2)]
    rows = [(t, 1 - y if r.random() < a.noise else y) for t, y in rows]
    r.shuffle(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} reviews to {a.out}")


if __name__ == "__main__":
    main()
