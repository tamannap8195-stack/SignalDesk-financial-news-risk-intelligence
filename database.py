"""MongoDB persistence helpers for articles, analyses, and risk alerts."""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import PyMongoError

load_dotenv()


def _article_companies(entities: dict[str, Any], text: str) -> list[str]:
    """Keep spaCy organization candidates that are actually named in article text."""
    body = text or ""
    found: list[str] = []
    for entity in (entities or {}).get("organizations", []):
        name = str(entity).strip()
        if not name:
            continue
        if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", body, re.I):
            if name.casefold() not in {item.casefold() for item in found}:
                found.append(name)
            continue
        # spaCy can occasionally label an entire headline phrase as an ORG.
        # Retain a capitalized name token only when it occurs with that casing
        # in the article body; this avoids publishing title fragments as firms.
        for token in re.findall(r"\b[A-Z][A-Za-z0-9&.'-]*\b", name):
            if re.search(r"(?<!\w)" + re.escape(token) + r"(?!\w)", body):
                if token.casefold() not in {item.casefold() for item in found}:
                    found.append(token)
    return found


@st.cache_resource(show_spinner=False)
def get_database() -> Any:
    """Create and cache the configured Mongo database handle."""
    uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    name = os.getenv("MONGODB_DATABASE", "financial_news_intelligence")
    client = MongoClient(uri, serverSelectionTimeoutMS=2500)
    client.admin.command("ping")
    db = client[name]
    db.articles.create_index([("created_at", DESCENDING)])
    db.analyses.create_index([("article_id", ASCENDING)], unique=True)
    db.risk_alerts.create_index([("status", ASCENDING), ("priority", ASCENDING), ("created_at", DESCENDING)])
    db.risk_alerts.create_index([("company_name", ASCENDING), ("event_category", ASCENDING)])
    return db


def save_analysis(article: dict[str, Any], analysis: dict[str, Any]) -> Any:
    """Persist one article, its analysis, and one alert per event/company."""
    db = get_database()
    now = datetime.now(timezone.utc)
    article_doc = {"title": article["title"], "text": article["text"], "source_name": article.get("source_name") or None, "source_url": article.get("source_url") or None, "published_at": article.get("published_at") or None, "created_at": now}
    # Avoid duplicate article inserts for a repeated form submission.
    existing = db.articles.find_one({"title": article_doc["title"], "text": article_doc["text"], "source_url": article_doc["source_url"]}, {"_id": 1})
    article_id = existing["_id"] if existing else db.articles.insert_one(article_doc).inserted_id
    analysis_doc = {**analysis, "article_id": article_id}
    analysis_doc.pop("title", None)
    db.analyses.update_one({"article_id": article_id}, {"$set": analysis_doc}, upsert=True)
    entities = _article_companies(analysis.get("extracted_entities", {}), article_doc["text"]) or [None]
    for event in analysis.get("detected_events", []):
        for company in entities:
            alert = {"article_id": article_id, "company_name": company, "event_category": event["category"], "evidence_sentence": event["evidence_sentence"], "keyword": event["keyword"], "sentiment": analysis["sentiment"]["label"], "risk_score": analysis["risk_score"], "priority": analysis["priority"], "status": "Pending", "source_name": article_doc["source_name"], "source_url": article_doc["source_url"], "published_at": article_doc["published_at"], "article_title": article_doc["title"], "created_at": now}
            db.risk_alerts.update_one(
                {"article_id": article_id, "event_category": alert["event_category"], "company_name": company},
                {
                    "$set": {key: value for key, value in alert.items() if key not in {"status", "created_at"}},
                    "$setOnInsert": {"status": alert["status"], "created_at": alert["created_at"]},
                },
                upsert=True,
            )
    return article_id


def reconcile_saved_analyses() -> int:
    """Re-run current event rules on saved text and backfill missing alerts.

    This upgrades articles saved by older, narrower event lexicons. Existing
    alert status and identity are preserved through upserts.
    """
    from analyzer import calculate_risk, detect_events

    db = get_database()
    created_or_updated = 0
    for article in db.articles.find({}, {"title": 1, "text": 1, "source_name": 1, "source_url": 1, "published_at": 1}):
        analysis = db.analyses.find_one({"article_id": article["_id"]})
        if not analysis:
            continue
        events = detect_events(article.get("text", ""))
        risk = calculate_risk(analysis.get("sentiment", {}), events, article.get("published_at"))
        db.analyses.update_one(
            {"article_id": article["_id"]},
            {"$set": {"detected_events": events, **risk}},
        )
        companies = _article_companies(analysis.get("extracted_entities", {}), article.get("text", "")) or [None]
        for event in events:
            for company in companies:
                alert = {
                    "article_id": article["_id"],
                    "company_name": company,
                    "event_category": event["category"],
                    "evidence_sentence": event["evidence_sentence"],
                    "keyword": event["keyword"],
                    "sentiment": (analysis.get("sentiment") or {}).get("label", "Unknown"),
                    "risk_score": risk["risk_score"],
                    "priority": risk["priority"],
                    "status": "Pending",
                    "source_name": article.get("source_name"),
                    "source_url": article.get("source_url"),
                    "published_at": article.get("published_at"),
                    "article_title": article.get("title", "Untitled article"),
                    "created_at": datetime.now(timezone.utc),
                }
                result = db.risk_alerts.update_one(
                    {"article_id": article["_id"], "event_category": event["category"], "company_name": company},
                    {
                        "$set": {key: value for key, value in alert.items() if key not in {"status", "created_at"}},
                        "$setOnInsert": {"status": alert["status"], "created_at": alert["created_at"]},
                    },
                    upsert=True,
                )
                if result.upserted_id is not None:
                    created_or_updated += 1
                # Clean pending duplicates made by older organization matching;
                # reviewed/dismissed records are retained for audit history.
                db.risk_alerts.delete_many({
                    "article_id": article["_id"],
                    "event_category": event["category"],
                    "evidence_sentence": event["evidence_sentence"],
                    "company_name": {"$nin": companies},
                    "status": {"$in": ["Pending", "Open"]},
                })
    return created_or_updated


def load_overview() -> dict[str, Any]:
    """Return fresh records used by dashboard cards and charts."""
    db = get_database()
    return {"article_count": db.articles.count_documents({}), "analyses": list(db.analyses.find({}, {"sentiment": 1, "priority": 1, "risk_score": 1, "detected_events": 1, "extracted_entities": 1, "analysis_timestamp": 1, "article_id": 1}).sort("analysis_timestamp", DESCENDING)), "alerts": list(db.risk_alerts.find({}, {"_id": 1, "company_name": 1, "article_title": 1, "event_category": 1, "sentiment": 1, "risk_score": 1, "priority": 1, "evidence_sentence": 1, "source_name": 1, "published_at": 1, "created_at": 1, "status": 1, "article_id": 1}).sort("created_at", DESCENDING))}


def update_alert_status(alert_id: Any, status: str) -> bool:
    """Persist a user status change."""
    from bson import ObjectId
    if status not in {"Open", "Pending", "Reviewed", "Dismissed"}:
        raise ValueError("Unsupported alert status")
    result = get_database().risk_alerts.update_one({"_id": ObjectId(str(alert_id))}, {"$set": {"status": status, "updated_at": datetime.now(timezone.utc)}})
    return result.modified_count > 0


def get_article_analysis(article_id: Any) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    db = get_database()
    return db.articles.find_one({"_id": article_id}), db.analyses.find_one({"article_id": article_id})
