from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
INTERMEDIATE_DIR = ROOT / "data" / "intermediate"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# None berarti dipetakan otomatis dan divalidasi sebelum fase berikutnya.
TITLE_COLUMN = None
TEXT_COLUMN = None
DATE_COLUMN = None
SOURCE_COLUMN = None
URL_COLUMN = None
TEXT_COLUMNS = ["title", "content", "summary", "description", "text", "body"]

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RANDOM_SEED = 42
FORECAST_HORIZONS = [1, 2, 3]
VALIDATION_RATIO = 0.20
TEST_RATIO = 0.20
PERIOD_FREQUENCY = "D"
GROWTH_QUANTILE = 0.80
MIN_HISTORY = 4

TOPIC_PARAMETERS = {
    "min_topic_size": 8,
    "n_neighbors": 15,
    "n_components": 5,
    "min_cluster_size": 8,
    "top_n_words": 10,
}
MODEL_PARAMETERS = {
    "logistic_regression": {"C": 1.0, "max_iter": 2000, "class_weight": "balanced"},
    "random_forest": {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 2, "class_weight": "balanced"},
    "xgboost": {"n_estimators": 250, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85},
}
BATCH_SIZE = 64
SAMPLE_SIZE = 300

COLUMN_CANDIDATES = {
    "title": ["title", "headline", "judul"],
    "text": ["content", "text", "body", "summary", "description", "isi"],
    "date": ["published_at", "date", "datetime", "published", "timestamp", "tanggal"],
    "source": ["source", "publisher", "media", "sumber"],
    "url": ["url", "link", "article_url"],
    "id": ["article_id", "record_id", "id"],
    "category": ["category", "topic", "section", "kategori"],
}


def ensure_directories():
    for path in [DATA_PATH, PROCESSED_DIR, INTERMEDIATE_DIR, MODEL_DIR,
                 RESULTS_DIR / "tables", RESULTS_DIR / "figures",
                 RESULTS_DIR / "predictions", RESULTS_DIR / "metrics", ROOT / "notebooks"]:
        path.mkdir(parents=True, exist_ok=True)
