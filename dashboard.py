"""Streamlit page rendering for overview, analyzer, and alert management."""
from __future__ import annotations

from html import escape
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from analyzer import analyze_article
from database import get_article_analysis, load_overview, reconcile_saved_analyses, save_analysis, update_alert_status

COLORS = {"Positive": "#35c98b", "Negative": "#ff6577", "Neutral": "#9aa7b8", "Low": "#35c98b", "Medium": "#f2b84b", "High": "#ff6577"}


def _open_analyzer() -> None:
    """Switch the sidebar navigation before its radio widget is rendered."""
    st.session_state.navigation_page = "News Analyzer"


def _overview() -> None:
    st.markdown("## Market signal overview")
    st.caption("A live view of analyzed articles and potential risk events stored in MongoDB.")
    try:
        data = load_overview()
    except Exception as exc:
        st.error(f"MongoDB is unavailable. Check MONGODB_URI and your database connection. Details: {exc}")
        return
    analyses, alerts = data["analyses"], data["alerts"]
    companies = {name for a in analyses for name in (a.get("extracted_entities") or {}).get("organizations", [])}
    high_count = sum(a.get("priority") == "High" for a in analyses)
    cols = st.columns(4)
    for col, value, label in zip(cols, [data["article_count"], sum(len(a.get("detected_events", [])) for a in analyses), high_count, len(companies)], ["Articles analyzed", "Potential events", "High priority", "Organizations"]):
        col.markdown(f'<div class="metric"><div class="metric-value">{value:,}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)
    if not analyses:
        st.markdown('<div class="empty"><div class="empty-icon">◈</div><b>Your intelligence feed is ready</b><br/>Analyze an article to populate live sentiment and risk metrics.</div>', unsafe_allow_html=True)
        st.button("Open News Analyzer", type="primary", on_click=_open_analyzer)
        return
    left, right = st.columns(2)
    with left:
        sentiment_counts = pd.Series([a.get("sentiment", {}).get("label", "Unknown") for a in analyses]).value_counts().rename_axis("Sentiment").reset_index(name="Articles")
        fig = px.pie(sentiment_counts, names="Sentiment", values="Articles", hole=.68, color="Sentiment", color_discrete_map=COLORS)
        fig.update_layout(title="Sentiment distribution", **_chart_layout())
        st.plotly_chart(fig, use_container_width=True)
    with right:
        priorities = pd.Series([a.get("priority", "Low") for a in analyses]).value_counts().reindex(["Low", "Medium", "High"], fill_value=0).rename_axis("Priority").reset_index(name="Articles")
        fig = px.bar(priorities, x="Priority", y="Articles", color="Priority", color_discrete_map=COLORS)
        fig.update_layout(title="Risk priority", **_chart_layout())
        st.plotly_chart(fig, use_container_width=True)
    event_categories = [event.get("category") for a in analyses for event in a.get("detected_events", [])]
    if event_categories:
        counts = pd.Series(event_categories).value_counts().rename_axis("Category").reset_index(name="Events")
        fig = px.bar(counts, x="Events", y="Category", orientation="h", color_discrete_sequence=["#4f9cff"])
        fig.update_layout(title="Potential events by category", **{**_chart_layout(), "height": max(370, 100 + 46 * len(counts))})
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Recent alerts")
    _alert_table(alerts[:8])


def _chart_layout() -> dict[str, Any]:
    return {"paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)", "font": {"family": 'Inter, "Segoe UI", Arial, sans-serif', "size": 14, "color": "#dce5f0"}, "title_font": {"size": 20, "color": "#edf3fa"}, "height": 370, "margin": {"t": 64, "b": 42, "l": 18, "r": 18}, "legend": {"orientation": "h", "y": -0.18, "font": {"size": 14, "color": "#c4cfdd"}}, "xaxis": {"tickfont": {"size": 13}, "title_font": {"size": 14}}, "yaxis": {"tickfont": {"size": 13}, "title_font": {"size": 14}, "gridcolor": "#233144"}}


def _render_analysis(article: dict[str, Any], result: dict[str, Any], article_id: Any = None) -> None:
    st.markdown(f"### {result.get('title') or article.get('title')}")
    st.caption("Analysis results · Potential events are evidence matches and do not establish that an event occurred.")
    sentiment = result["sentiment"]
    cols = st.columns(3)
    cols[0].metric("Sentiment", sentiment["label"])
    cols[1].metric("Model confidence", f"{sentiment['confidence']:.1%}")
    cols[2].metric("Risk assessment", f"{result['risk_score']} / 100 · {result['priority']}")
    scores = pd.DataFrame({"Sentiment": ["Positive", "Negative", "Neutral"], "Model score": [sentiment["scores"].get(k, 0) for k in ("positive", "negative", "neutral")]})
    fig = px.bar(scores, x="Sentiment", y="Model score", color="Sentiment", color_discrete_map=COLORS, range_y=[0, 1])
    fig.update_layout(**_chart_layout(), title="FinBERT class scores")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("FinBERT scores describe the model's classification of this text. They are not probabilities of future financial outcomes.")
    organizations = result.get("extracted_entities", {}).get("organizations", [])
    st.markdown("#### Organizations identified")
    st.write(" · ".join(organizations) if organizations else "No organization entities identified. spaCy does not determine whether an entity is publicly traded.")
    st.markdown("#### Potential risk events")
    events = result.get("detected_events", [])
    if events:
        for event in events:
            st.markdown(f'<div class="event"><span class="event-tag">POTENTIAL · {event["category"]}</span><p>{event["evidence_sentence"]}</p><small>Matched phrase: “{event["keyword"]}” · Rule severity {event["severity"]}/5</small></div>', unsafe_allow_html=True)
    else:
        st.info("No configured risk event phrases matched this article.")
    st.markdown("#### Score explanation")
    factors = result["factors"]
    st.write(f"Weighted factors: event severity {factors['event_severity']:.1f}/100 × 50%, negative sentiment {factors['negative_sentiment']:.1f}/100 × 20%, recency {factors['recency']:.1f}/100 × 15%, evidence completeness {factors['evidence_completeness']:.1f}/100 × 15%. {result['recency_note']}.")
    st.caption("Demonstration scoring weights; this is not a validated investment-risk measure.")


def _analyzer() -> None:
    st.markdown("## News analyzer")
    st.caption("Analyze a financial article with FinBERT sentiment, spaCy entity extraction, and explainable event rules.")
    sample = {"title": "Illustrative: Northstar Components reports supply disruption", "text": "Synthetic illustrative news. Northstar Components said a fire halted production at its main facility on Tuesday. The company expects operations to resume next week, while its shares and financial outlook remain unaffected.", "source_name": "Synthetic sample", "source_url": "", "published_at": ""}
    if st.button("Load illustrative article"):
        for key in ["article_title", "article_text", "source_name", "source_url", "published_at"]:
            st.session_state[key] = sample[{"article_title": "title", "article_text": "text", "source_name": "source_name", "source_url": "source_url", "published_at": "published_at"}[key]]
    with st.form("analysis_form"):
        title = st.text_input("Article title *", key="article_title", max_chars=500)
        text = st.text_area("Article text *", key="article_text", height=300, max_chars=100000)
        a, b, c = st.columns(3)
        source = a.text_input("Source name", key="source_name")
        url = b.text_input("Source URL", key="source_url")
        published = c.text_input("Published at (ISO date/time)", key="published_at", placeholder="2026-05-21T10:30:00Z")
        submitted = st.form_submit_button("Analyze article", type="primary", use_container_width=True)
    if submitted:
        if not title.strip() or not text.strip():
            st.error("Enter both an article title and article text.")
        else:
            with st.spinner("Loading models and analyzing the article… The first run may download model files."):
                try:
                    result = analyze_article(title, text, published or None)
                    article = {"title": title.strip(), "text": text.strip(), "source_name": source.strip() or None, "source_url": url.strip() or None, "published_at": published.strip() or None}
                    # Keep the model result visible even if MongoDB is temporarily
                    # unavailable; analysis and persistence are separate steps.
                    st.session_state.last_analysis = (article, result, None)
                    st.session_state.analysis_error = None
                except Exception as exc:
                    st.error(f"Article analysis failed: {exc}")
                else:
                    try:
                        article_id = save_analysis(article, result)
                        st.session_state.last_analysis = (article, result, str(article_id))
                        st.success("Analysis saved to MongoDB.")
                    except Exception as exc:
                        st.session_state.analysis_error = str(exc)
                        st.warning(f"Analysis completed, but could not be saved to MongoDB: {exc}")
    upload = st.file_uploader("Or analyze a CSV batch", type=["csv"], help="Required columns: title, text. Optional: source_name, source_url, published_at. Maximum 5 MB.")
    if upload:
        if upload.size > 5 * 1024 * 1024:
            st.error("CSV exceeds the 5 MB limit.")
        else:
            try:
                frame = pd.read_csv(upload, encoding="utf-8-sig")
                frame.columns = [str(column).strip().lower() for column in frame.columns]
                frame = frame.fillna("").reset_index(drop=True)
                missing = {"title", "text"} - set(frame.columns)
                if missing:
                    st.error("CSV is missing required columns: " + ", ".join(sorted(missing)))
                elif len(frame) > 50:
                    st.error("CSV is limited to 50 articles per upload.")
                elif frame.empty:
                    st.warning("This CSV has a header but no article rows.")
                elif st.button("Analyze CSV articles", type="primary", key="analyze_csv"):
                    progress = st.progress(0)
                    summary = {"saved": 0, "skipped": 0, "errors": []}
                    for row_number, row in enumerate(frame.to_dict(orient="records"), start=1):
                        title_value = str(row.get("title", "")).strip()
                        text_value = str(row.get("text", "")).strip()
                        if not title_value or not text_value:
                            summary["skipped"] += 1
                            summary["errors"].append(f"Row {row_number}: missing title or text")
                            progress.progress(row_number / len(frame))
                            continue
                        try:
                            article = {key: str(row.get(key, "")).strip() or None for key in ("title", "text", "source_name", "source_url", "published_at")}
                            analysis = analyze_article(article["title"], article["text"], article["published_at"])
                            save_analysis(article, analysis)
                            summary["saved"] += 1
                        except Exception as exc:
                            summary["skipped"] += 1
                            summary["errors"].append(f"Row {row_number} ({title_value[:60]}): {exc}")
                        progress.progress(row_number / len(frame))
                    st.session_state.csv_summary = summary
                    if summary["saved"]:
                        st.success(f"Analysis complete: saved {summary['saved']} of {len(frame)} rows; skipped {summary['skipped']}.")
                    else:
                        st.error(f"No rows were saved. {summary['skipped']} row(s) were skipped or failed.")
                    for message in summary["errors"][:10]:
                        st.warning(message)
            except Exception as exc:
                st.error(f"Could not read CSV: {exc}")
    if st.session_state.get("csv_summary") and not upload:
        summary = st.session_state.csv_summary
        st.info(f"Last CSV run: {summary['saved']} saved, {summary['skipped']} skipped.")
    last = st.session_state.get("last_analysis")
    if last:
        if last[2] is None and st.session_state.get("analysis_error"):
            st.caption("This result is currently shown in this session only because database saving failed.")
        _render_analysis(*last[:2], article_id=last[2])


def _alert_table(alerts: list[dict[str, Any]]) -> None:
    if not alerts:
        st.info("No alerts yet. Alerts are created when configured potential risk events are detected.")
        return
    frame = pd.DataFrame([{ "Company": a.get("company_name") or "Unidentified", "Article": a.get("article_title", ""), "Category": a.get("event_category", ""), "Score": a.get("risk_score", 0), "Priority": a.get("priority", ""), "Status": a.get("status", "Open")} for a in alerts])
    st.dataframe(frame, use_container_width=True, hide_index=True, row_height=42)


def _intelligence() -> None:
    st.markdown("## Risk intelligence")
    try:
        # Upgrade old saved analyses with current event rules and backfill only
        # evidence-supported alerts. Existing review/dismissal states are kept.
        reconcile_saved_analyses()
        alerts = load_overview()["alerts"]
    except Exception as exc:
        st.error(f"MongoDB is unavailable: {exc}")
        return

    def display_status(alert: dict[str, Any]) -> str:
        value = str(alert.get("status", "Pending"))
        return "Pending" if value.casefold() == "open" else value.title()

    pending_count = sum(display_status(alert) == "Pending" for alert in alerts)
    reviewed_count = sum(display_status(alert) == "Reviewed" for alert in alerts)
    high_count = sum(str(alert.get("priority", "")).casefold() == "high" for alert in alerts)
    st.caption("Evidence-backed alerts generated from saved financial news analysis.")
    metrics = st.columns(4)
    for col, value, label in zip(
        metrics,
        (len(alerts), pending_count, high_count, reviewed_count),
        ("Total alerts", "Pending", "High priority", "Reviewed"),
    ):
        col.markdown(
            f'<div class="metric"><div class="metric-value">{value:,}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    if not alerts:
        st.markdown(
            '<div class="empty"><div class="empty-icon">?</div><b>No risk alerts yet</b><br/>'
            'Analyze an article with a potential risk event to create an evidence-backed alert.</div>',
            unsafe_allow_html=True,
        )
        st.button("Open News Analyzer", type="primary", on_click=_open_analyzer)
        return

    companies = sorted(
        {str(a.get("company_name") or "Unidentified") for a in alerts},
        key=str.casefold,
    )
    categories = sorted(
        {str(a.get("event_category") or "Uncategorized") for a in alerts},
        key=str.casefold,
    )
    search_col, company_col, category_col, priority_col, status_col = st.columns(
        [1.65, 1.15, 1.45, 1.0, 1.0]
    )
    query = search_col.text_input(
        "Search company", placeholder="Company name?", key="risk_company_search"
    )
    company = company_col.selectbox(
        "Company", ["All", *companies], key="risk_company_filter"
    )
    category = category_col.selectbox(
        "Event category", ["All", *categories], key="risk_category_filter"
    )
    priority_filter = priority_col.selectbox(
        "Priority", ["All", "High", "Medium", "Low"], key="risk_priority_filter"
    )
    status_filter = status_col.selectbox(
        "Status", ["All", "Pending", "Reviewed", "Dismissed"], key="risk_status_filter"
    )

    filtered = [
        alert for alert in alerts
        if (company == "All" or str(alert.get("company_name") or "Unidentified") == company)
        and (category == "All" or str(alert.get("event_category") or "Uncategorized") == category)
        and (priority_filter == "All" or str(alert.get("priority", "")).casefold() == priority_filter.casefold())
        and (status_filter == "All" or display_status(alert) == status_filter)
        and (not query or query.casefold() in str(alert.get("company_name") or "Unidentified").casefold())
    ]
    st.caption(f"Showing {len(filtered)} of {len(alerts)} alerts")
    if not filtered:
        st.info("No alerts match those filters. Choose All or clear the company search to see more alerts.")
        return

    filtered.sort(
        key=lambda item: (
            display_status(item) != "Pending",
            -int(item.get("risk_score", 0) or 0),
            str(item.get("created_at", "")),
        )
    )
    for alert in filtered:
        score = int(alert.get("risk_score", 0) or 0)
        priority_label = str(alert.get("priority", "Low")).title()
        priority_class = priority_label.lower() if priority_label.lower() in {"low", "medium", "high"} else "low"
        status_label = display_status(alert)
        status_class = status_label.lower()
        company_name = escape(str(alert.get("company_name") or "Organization not identified"))
        category_name = escape(str(alert.get("event_category", "Uncategorized")))
        article_title = escape(str(alert.get("article_title", "Untitled article")))
        evidence = escape(str(alert.get("evidence_sentence", "Evidence sentence unavailable.")))
        sentiment = escape(str(alert.get("sentiment", "Unknown")))
        with st.container(border=True):
            header = st.columns([3.4, 1, 1.25])
            header[0].markdown(
                f'<div class="alert-heading"><strong>{company_name}</strong>'
                f'<span class="priority-badge priority-{priority_class}">{escape(priority_label)} priority</span></div>'
                f'<div class="alert-category">{category_name}</div>',
                unsafe_allow_html=True,
            )
            header[1].markdown(
                f'<div class="alert-score"><span>Risk score</span>'
                f'<strong>{score}<small> / 100</small></strong></div>',
                unsafe_allow_html=True,
            )
            header[2].markdown(
                f'<div class="alert-status status-{status_class}">{escape(status_label)}</div>'
                f'<div class="alert-sentiment">Sentiment: <b>{sentiment}</b></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="alert-title">{article_title}</div>'
                f'<div class="evidence-text">?{evidence}?</div>',
                unsafe_allow_html=True,
            )
            published = escape(str(alert.get("published_at") or "Date unavailable"))
            source_name = escape(str(alert.get("source_name") or "Not supplied"))
            st.caption(f"Published: {published}  ?  Source: {source_name}")

            alert_id = str(alert["_id"])
            actions = st.columns([1.6, 1, 4.4])
            if actions[0].button(
                "Mark as Reviewed",
                key=f"review_{alert_id}",
                disabled=status_label in {"Reviewed", "Dismissed"},
                use_container_width=True,
            ):
                update_alert_status(alert_id, "Reviewed")
                st.rerun()
            if actions[1].button(
                "Dismiss",
                key=f"dismiss_{alert_id}",
                disabled=status_label == "Dismissed",
                use_container_width=True,
            ):
                update_alert_status(alert_id, "Dismissed")
                st.rerun()

            with st.expander("View detailed alert", expanded=False):
                st.markdown("**Supporting evidence**")
                st.write(alert.get("evidence_sentence") or "Evidence sentence unavailable.")
                st.markdown("**Alert details**")
                st.write(
                    f"Company: {alert.get('company_name') or 'Organization not identified'}  ?  "
                    f"Category: {alert.get('event_category', 'Uncategorized')}  ?  "
                    f"Sentiment: {alert.get('sentiment', 'Unknown')}  ?  "
                    f"Risk: {score}/100 ({priority_label})  ?  Status: {status_label}"
                )
                st.write(
                    f"Publication date: {alert.get('published_at') or 'Unavailable'}  ?  "
                    f"Alert created: {alert.get('created_at', 'Unavailable')}"
                )
                article_id = alert.get("article_id")
                if article_id:
                    try:
                        article, analysis = get_article_analysis(article_id)
                        if article and analysis:
                            st.markdown("**Full article analysis**")
                            _render_analysis(article, analysis)
                        else:
                            st.info("The source article or analysis is no longer available.")
                    except Exception as exc:
                        st.warning(f"Could not load the saved article analysis: {exc}")


def render(page: str) -> None:
    if page == "Overview":
        _overview()
    elif page == "News Analyzer":
        _analyzer()
    else:
        _intelligence()
