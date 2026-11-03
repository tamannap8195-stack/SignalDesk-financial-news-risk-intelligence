from datetime import datetime, timezone

from analyzer import analyze_sentiment, calculate_risk, detect_events


class _Tensor:
    def __init__(self, data): self.data = data
    def __getitem__(self, index): return self
    def cpu(self): return self
    def tolist(self): return self.data


class _Torch:
    @staticmethod
    def inference_mode():
        from contextlib import nullcontext
        return nullcontext()
    @staticmethod
    def softmax(_logits, dim=-1): return _Tensor([0.1, 0.8, 0.1])


class _Tokenizer:
    def __call__(self, *_args, **_kwargs): return {"input_ids": [1]}


class _Model:
    config = type("Config", (), {"id2label": {0: "positive", 1: "negative", 2: "neutral"}})()
    def __call__(self, **_kwargs): return type("Output", (), {"logits": _Tensor([0, 0, 0])})()


def test_sentiment_result_structure(monkeypatch):
    import torch
    monkeypatch.setattr(torch, "inference_mode", _Torch.inference_mode)
    monkeypatch.setattr(torch, "softmax", _Torch.softmax)
    result = analyze_sentiment("An article", (_Tokenizer(), _Model()))
    assert result["label"] == "Negative"
    assert set(result["scores"]) == {"positive", "negative", "neutral"}
    assert result["confidence"] == .8


def test_event_detection_and_negation():
    found = detect_events("The regulator opened a regulatory investigation. There is no evidence of fraud. A separate data breach exposed records.")
    assert [e["category"] for e in found] == ["Regulatory investigations and legal issues", "Cybersecurity incidents and data breaches"]
    assert all(e["potential"] for e in found)


def test_saved_article_wording_creates_supported_risk_events():
    disruption = detect_events("Zerodha experiences a prolonged disruption affecting access to its trading platform.")
    scrutiny = detect_events("Deloitte faces regulatory scrutiny after concerns are raised about audit compliance.")
    assert disruption[0]["category"] == "Operational disruptions"
    assert disruption[0]["keyword"] == "prolonged disruption"
    assert scrutiny[0]["category"] == "Regulatory investigations and legal issues"
    assert scrutiny[0]["evidence_sentence"].startswith("Deloitte faces regulatory scrutiny")


def test_risk_score_boundaries_and_no_event_cap():
    sentiment = {"scores": {"negative": 1.0}}
    result = calculate_risk(sentiment, [])
    assert 0 <= result["risk_score"] <= 34
    assert result["priority"] == "Low"
    assert calculate_risk(sentiment, [{"severity": 5}]) ["risk_score"] <= 100


def test_priority_and_missing_or_invalid_metadata():
    sentiment = {"scores": {"negative": 0.4}}
    event = [{"severity": 5}]
    result = calculate_risk(sentiment, event, "not-a-date", now=datetime.now(timezone.utc))
    assert result["priority"] in {"Low", "Medium", "High"}
    assert result["factors"]["recency"] == 0
    assert "invalid" in result["recency_note"]
    missing = calculate_risk(sentiment, event)
    assert missing["factors"]["recency"] == 0


def test_spacy_extraction_with_mock():
    from analyzer import extract_entities
    class Ent:
        def __init__(self, text, label): self.text, self.label_ = text, label
    class NLP:
        def __call__(self, _text): return type("Doc", (), {"ents": [Ent("Acme Corp", "ORG"), Ent("Paris", "GPE")]})()
    assert extract_entities("Acme Corp", NLP())["organizations"] == ["Acme Corp"]


def test_mongodb_save_analysis_with_mock(monkeypatch):
    import database
    class Collection:
        def __init__(self): self.docs = []; self.updates = []
        def find_one(self, query, projection=None): return next((d for d in self.docs if all(d.get(k) == v for k, v in query.items())), None)
        def insert_one(self, doc):
            doc = dict(doc); doc["_id"] = "article-1"; self.docs.append(doc)
            return type("Result", (), {"inserted_id": "article-1"})()
        def update_one(self, query, update, upsert=False): self.updates.append((query, update, upsert))
    class DB:
        def __init__(self): self.articles, self.analyses, self.risk_alerts = Collection(), Collection(), Collection()
    fake_db = DB()
    monkeypatch.setattr(database, "get_database", lambda: fake_db)
    article = {"title": "Synthetic report", "text": "Acme reported a data breach."}
    analysis = {"sentiment": {"label": "Negative", "scores": {"negative": .9}}, "extracted_entities": {"organizations": ["Acme"]}, "detected_events": [{"category": "Cybersecurity incidents and data breaches", "keyword": "data breach", "evidence_sentence": "Acme reported a data breach.", "severity": 4}], "risk_score": 70, "priority": "High"}
    assert database.save_analysis(article, analysis) == "article-1"
    assert len(fake_db.articles.docs) == 1
    assert fake_db.analyses.updates[0][0] == {"article_id": "article-1"}
    alert_query, alert_update, upsert = fake_db.risk_alerts.updates[0]
    assert alert_query["company_name"] == "Acme" and upsert
    assert alert_update["$setOnInsert"]["status"] == "Pending"


def test_alert_status_is_persisted_without_deleting_record(monkeypatch):
    import database
    from bson import ObjectId

    alert_id = ObjectId()
    class Alerts:
        def __init__(self): self.query = None; self.update = None
        def update_one(self, query, update):
            self.query, self.update = query, update
            return type("Result", (), {"modified_count": 1})()
    class DB:
        def __init__(self): self.risk_alerts = Alerts()
    fake_db = DB()
    monkeypatch.setattr(database, "get_database", lambda: fake_db)
    assert database.update_alert_status(alert_id, "Dismissed") is True
    assert fake_db.risk_alerts.query == {"_id": alert_id}
    assert fake_db.risk_alerts.update["$set"]["status"] == "Dismissed"
