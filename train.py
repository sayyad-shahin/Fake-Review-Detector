"""Train the detector.   python train.py
   Use your own data:     python train.py --data my_reviews.csv      (columns: text,label  with label 1=fake, 0=genuine)
"""
import argparse
from fakereview import train_model, DEFAULT_DATA, DEFAULT_MODEL

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=DEFAULT_DATA)
ap.add_argument("--out", default=DEFAULT_MODEL)
ap.add_argument("--test-size", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
train_model(a.data, a.out, a.test_size, a.seed)
