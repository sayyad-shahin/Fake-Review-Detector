import csv, os, random, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import generate_dataset as gd
from fakereview import Detector, train_model
from fakereview.features import FEATURE_NAMES, red_flags, style_features
from fakereview.model import load_dataset


def write_csv(path, rows, header=("text", "label")):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


class TestFeatures(unittest.TestCase):
    def test_shape(self):
        self.assertEqual(style_features(["Nice.", "I bought it 2 days ago!"]).shape, (2, len(FEATURE_NAMES)))

    def test_red_flags(self):
        flags, _ = red_flags("BEST PRODUCT EVER!!! AMAZING and PERFECT!!! Buy it now, you must!!!")
        self.assertTrue(any("exclamation" in f for f in flags))
        self.assertTrue(any("ALL CAPS" in f for f in flags))
        _, good = red_flags("I used it for 3 weeks, but the strap is a bit short and I wish it was longer.")
        self.assertTrue(good)


class TestData(unittest.TestCase):
    def test_kaggle_style_columns(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "k.csv")
            write_csv(p, [("Books", 5, "CG", "Great book!"), ("Books", 4, "OR", "Read it in 3 days.")],
                      header=("category", "rating", "label", "text_"))
            texts, y = load_dataset(p)
        self.assertEqual(list(y), [1, 0]); self.assertEqual(len(texts), 2)

    def test_bad_columns(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "bad.csv"); write_csv(p, [("a", "b")], header=("foo", "bar"))
            with self.assertRaises(ValueError):
                load_dataset(p)


class TestModelAndApp(unittest.TestCase):
    def test_training_works(self):
        r = random.Random(1)
        rows = [(gd.genuine(r), 0) for _ in range(200)] + [(gd.fake(r), 1) for _ in range(200)]
        r.shuffle(rows)
        with tempfile.TemporaryDirectory() as d:
            data, model = os.path.join(d, "r.csv"), os.path.join(d, "m.joblib")
            write_csv(data, rows)
            m = train_model(data, model, verbose=False)
            self.assertGreater(m["accuracy"], 0.9)
            det = Detector.load(model)
            self.assertEqual(det.analyze("BEST EVER!!! Amazing, perfect, buy now!!!")["verdict"], "Likely fake")
            self.assertEqual(det.analyze("I used it for 3 weeks and the battery lasts about 8 hours, but the strap is a bit short.")["verdict"], "Likely genuine")
        with self.assertRaises(ValueError):
            det.analyze("   ")

    def test_web_api(self):
        from app import app
        c = app.test_client()
        self.assertEqual(c.get("/").status_code, 200)
        self.assertEqual(c.get("/api/health").get_json()["status"], "ok")
        r = c.post("/api/analyze", json={"text": "Best product ever!!! Buy now!!!"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("fake_probability", r.get_json())
        self.assertEqual(c.post("/api/analyze", json={"text": ""}).status_code, 400)
        self.assertEqual(c.post("/api/analyze", json={"text": 5}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
