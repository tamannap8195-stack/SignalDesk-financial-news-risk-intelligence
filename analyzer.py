"""Financial news analysis using FinBERT, spaCy, and explainable rules."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

import streamlit as st

LABELS = ("positive", "negative", "neutral")
EVENT_RULES = {
    "Financial distress and losses": (4, ("bankruptcy", "insolvency", "default", "liquidity crisis", "posted a loss", "suffers losses", "debt restructuring", "layoffs")),
    "Fraud and misconduct allegations": (5, ("fraud", "misconduct", "accounting scandal", "bribery", "embezzlement", "alleged manipulation", "accused of")),
    "Regulatory investigations and legal issues": (4, ("regulatory investigation", "regulatory scrutiny", "regulatory review", "under investigation", "lawsuit", "sued", "SEC probe", "regulator fined", "charged with", "legal action")),
    "Earnings warnings and declining guidance": (3, ("profit warning", "cuts its forecast", "lowered guidance", "declining revenue", "earnings miss", "downgraded its outlook", "revenue fell")),
    "Cybersecurity incidents and data breaches": (4, ("data breach", "cyberattack", "ransomware", "hacked", "customer data exposed", "security incident")),
    "Operational disruptions": (3, ("production halted", "factory shutdown", "supply disruption", "prolonged disruption", "platform disruption", "service interruption", "service disruption", "unable to access", "outage", "operations suspended", "recall", "strike shuts")),
}
NEGATION = re.compile(r"\b(no|not|never|without|denies|denied|dismissed|ruled out|not found|no evidence of)\b", re.I)


@st.cache_resource(show_spinner=False)
def load_finbert() -> tuple[Any, Any]:
    """Load tokenizer/model once; raises a clear error if weights cannot load."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner=False)
def load_spacy() -> Any:
    """Load the requested English spaCy pipeline."""
    import spacy

    return spacy.load("en_core_web_sm")


def analyze_sentiment(text: str, model_bundle: tuple[Any, Any] | None = None) -> dict[str, Any]:
    """Run genuine FinBERT inference and return ordered class scores."""
    if not text or not text.strip():
        raise ValueError("Article text is required.")
    tokenizer, model = model_bundle or load_finbert()
    import torch

    encoded = tokenizer(text[:20000], return_tensors="pt", truncation=True, max_length=512)
    with torch.inference_mode():
        logits = model(**encoded).logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().tolist()
    id2label = getattr(model.config, "id2label", {})
    scores = {str(id2label.get(i, LABELS[i] if i < len(LABELS) else str(i))).lower(): float(p) for i, p in enumerate(probs)}
    normalized = {label: next((v for k, v in scores.items() if label in k), 0.0) for label in LABELS}
    sentiment = max(normalized, key=normalized.get)
    return {"label": sentiment.title(), "confidence": normalized[sentiment], "scores": normalized}


def extract_entities(text: str, nlp: Any | None = None) -> dict[str, list[str]]:
    """Extract organizations and other relevant named entities with spaCy."""
    nlp = nlp or load_spacy()
    doc = nlp(text[:100000])
    orgs = list(dict.fromkeys(ent.text.strip() for ent in doc.ents if ent.label_ == "ORG" and ent.text.strip()))
    relevant = list(dict.fromkeys({ent.text.strip() for ent in doc.ents if ent.label_ in {"ORG", "PERSON", "GPE", "PRODUCT"} and ent.text.strip()}))
    return {"organizations": orgs, "relevant": sorted(relevant, key=str.casefold)}


def detect_events(text: str) -> list[dict[str, Any]]:
    """Match risk phrases to sentences, excluding simple negated mentions."""
    events: list[dict[str, Any]] = []
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    for sentence in sentences:
        for category, (severity, terms) in EVENT_RULES.items():
            for term in terms:
                match = re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", sentence, re.I)
                if match:
                    context = sentence[max(0, match.start() - 55):match.start()]
                    if NEGATION.search(context):
                        continue
                    events.append({"category": category, "severity": severity, "evidence_sentence": sentence, "keyword": match.group(0), "potential": True})
                    break
    # Avoid duplicate category evidence from repeated phrasing within one sentence.
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for event in events:
        unique[(event["category"], event["evidence_sentence"])] = event
    return list(unique.values())


def calculate_risk(sentiment: dict[str, Any], events: list[dict[str, Any]], published_at: Any = None, now: datetime | None = None) -> dict[str, Any]:
    """Calculate a transparent 0–100 demonstration score with explicit factors."""
    max_severity = max((int(e.get("severity", 0)) for e in events), default=0)
    event_factor = max_severity / 5 * 100
    neg = float(sentiment.get("scores", {}).get("negative", 0))
    sentiment_factor = neg * 100
    recency_factor, recency_note = 0.0, "No valid publication date supplied"
    if published_at:
        try:
            dt = published_at if isinstance(published_at, datetime) else datetime.fromisoformat(str(published_at).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age_days = max(0.0, ((now or datetime.now(timezone.utc)) - dt).total_seconds() / 86400)
            recency_factor = max(0.0, 100 - age_days / 30 * 100)
            recency_note = f"Publication age: {age_days:.1f} days"
        except (TypeError, ValueError, OverflowError):
            recency_note = "Publication date is invalid; recency contributes 0"
    evidence_factor = min(100.0, len(events) * 50.0)
    score = round(event_factor * .50 + sentiment_factor * .20 + recency_factor * .15 + evidence_factor * .15)
    # Without a detected event, cap the score so sentiment alone cannot create a high alert.
    if not events:
        score = min(score, 34)
    priority = "Low" if score < 35 else "Medium" if score < 65 else "High"
    return {"risk_score": max(0, min(100, score)), "priority": priority, "factors": {"event_severity": round(event_factor, 1), "negative_sentiment": round(sentiment_factor, 1), "recency": round(recency_factor, 1), "evidence_completeness": round(evidence_factor, 1)}, "recency_note": recency_note, "weights": {"event_severity": .50, "negative_sentiment": .20, "recency": .15, "evidence_completeness": .15}}


def analyze_article(title: str, text: str, published_at: Any = None) -> dict[str, Any]:
    """Run every analysis stage and return a database-ready result."""
    sentiment = analyze_sentiment(text)
    entities = extract_entities(f"{title}. {text}")
    events = detect_events(text)
    risk = calculate_risk(sentiment, events, published_at)
    return {"title": title.strip(), "sentiment": sentiment, "extracted_entities": entities, "detected_events": events, **risk, "analysis_timestamp": datetime.now(timezone.utc)}
