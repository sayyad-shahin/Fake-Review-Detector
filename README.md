# 🕵️ Fake Review Detector

A machine learning web app that checks whether a product, hotel or restaurant review looks **fake or genuine**. Paste a review, and the app shows the fake probability, the warning signs it found, and the words that influenced the decision.

Built with **Python, scikit-learn and Flask**. It also has a command-line tool and a JSON API.

## Features
- Web page: paste a review and get a verdict (Likely fake / Uncertain / Likely genuine) with a probability gauge
- Explains its decision: warning signs (ALL CAPS, many `!`, pushy "buy now" language, no concrete details...), trust signs, and influential words
- Command-line checker for one review or a whole CSV file
- JSON API (`POST /api/analyze`)
- Train on your own labelled dataset with one command
- Includes a sample dataset, a trained model, unit tests and GitHub Actions CI

## How it works
1. **Text features:** TF-IDF on single words and word pairs.
2. **Writing-style features (13):** length, exclamation marks, ALL-CAPS words, repeated punctuation, extreme words (best, perfect, worst...), call-to-action words, personal-experience words, hedging words (but, however, bit...), numbers and measurements, vocabulary variety, and more. Fake reviews tend to be vague, exaggerated and pushy; real ones tend to be specific and balanced.
3. **Classifier:** Logistic regression on both feature sets. It is fast and its decisions can be explained.
4. **Explanation:** For each review the app multiplies the model's weights by the review's features to show which signals pushed the score toward fake or genuine.

## Quick start

Requires Python 3.9 or newer.

```bash
# 1. (optional) create a virtual environment
py -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate        # macOS / Linux

# 2. install dependencies
pip install -r requirements.txt

# 3. start the web app
python app.py
```
Open **http://127.0.0.1:5000** in your browser. On Windows, use `py` instead of `python` if `python` doesn't work.

### Other commands

| Command | What it does |
|---|---|
| `python predict.py "Best product ever!!! Buy now!!!"` | Check one review in the terminal |
| `python predict.py` | Interactive mode: type reviews one by one |
| `python predict.py --csv reviews.csv --column text --out results.csv` | Check a whole CSV file |
| `python train.py` | Retrain the model on `data/reviews.csv` |
| `python train.py --data my_data.csv` | Train on your own dataset |
| `python generate_dataset.py` | Regenerate the sample dataset |
| `python -m unittest discover -s tests -v` | Run the tests |

If the saved model can't be loaded (for example after a scikit-learn upgrade), the app retrains it automatically from `data/reviews.csv`.

## API

```bash
curl -X POST http://127.0.0.1:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Best product ever!!! Buy now!!!"}'
```
Response:
```json
{
  "verdict": "Likely fake",
  "fake_probability": 0.95,
  "genuine_probability": 0.05,
  "warning_signs": ["Uses many exclamation marks (6)", "..."],
  "trust_signs": [],
  "words_pointing_to_fake": ["amazing", "..."],
  "words_pointing_to_genuine": []
}
```
Errors return HTTP 400 with `{"error": "..."}`. `GET /api/health` returns `{"status": "ok"}`.

## Important: about the data and accuracy

The included `data/reviews.csv` is a **synthetic sample dataset** (3,000 generated reviews, 3% of labels deliberately flipped) so the project runs out of the box. On this sample, the model scores about **98% accuracy** on 600 held-out reviews (precision 0.98, recall 0.98, 5-fold cross-validated F1 about 0.97).

**These numbers are high because the sample reviews follow simple patterns. Real-world accuracy will be lower.** Real fake reviews are written to look real, and no model can be certain. For a more meaningful system, train on real labelled data, such as the Deceptive Opinion Spam Corpus (Ott et al.) or the Kaggle "Fake Reviews Dataset". The trainer understands common column names (`text`, `review`, `text_`... and `label`, with values `1/fake/CG` or `0/genuine/OR`):

```bash
python train.py --data path/to/real_reviews.csv
```

## Limitations
- The output is a probability, not proof. Do not use it as the only basis for punishing a seller or reviewer.
- It looks at one review's text, not account behavior, timing, or duplicate posting across products, which are strong real-world signals.
- English text only.
- Models can be wrong, and a well-written fake review may pass.

## Project structure
```
fake-review-detector/
├── app.py                  # Flask web app and JSON API
├── train.py                # train the model
├── predict.py              # command-line checker (single review or CSV)
├── generate_dataset.py     # makes the sample dataset
├── fakereview/
│   ├── features.py         # writing-style features and warning signs
│   └── model.py            # data loading, training, saving, explanations
├── templates/index.html    # web page
├── static/                 # style.css and app.js
├── data/reviews.csv        # sample dataset
├── models/                 # detector.joblib and metrics.json
├── tests/                  # unit tests
├── requirements.txt
└── LICENSE
```

## Ideas to extend
- Train on a real dataset and compare scores
- Add reviewer-behavior features (many reviews in one day, same text on many products)
- Try other models such as a linear SVM, gradient boosting or a transformer
- Add a CSV upload page and a downloadable report
- Support more languages

## License
MIT. See [LICENSE](LICENSE).
