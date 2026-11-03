"""Financial News Risk Intelligence System — Streamlit entry point."""
import streamlit as st

from dashboard import render

st.set_page_config(page_title="SignalDesk · Financial Intelligence", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root{--bg:#080d14;--panel:#0e1622;--panel-raised:#111b29;--line:#26364a;--text:#e8eef6;--muted:#a0aec0;--blue:#4f9cff;--green:#35c98b;--amber:#f2b84b;--red:#ff6577;--font:Inter,"Segoe UI",Arial,sans-serif}
.stApp{background:var(--bg);color:var(--text);font-family:var(--font);font-size:16px}.block-container{padding:2.25rem 2.5rem 3rem;max-width:1680px}
section[data-testid="stSidebar"]{background:#0a111b;border-right:1px solid var(--line);min-width:270px}section[data-testid="stSidebar"] .block-container{padding:1.75rem 1.2rem}
h1,h2,h3,h4{color:#f0f5fb;letter-spacing:-.025em}h1{font-size:34px!important;font-weight:700!important}h2{font-size:25px!important;font-weight:650!important;margin:0 0 .45rem!important}h3{font-size:21px!important;font-weight:650!important}h4{font-size:17px!important}
.stMarkdown p{font-size:16px;line-height:1.65;color:#d5deea}.stCaption,div[data-testid="stCaptionContainer"] p{font-size:14px!important;line-height:1.55;color:var(--muted)!important}
.brand{padding:9px 4px 24px;border-bottom:1px solid var(--line);margin-bottom:22px}.brand-mark{display:inline-grid;place-items:center;width:38px;height:38px;border:1px solid #244063;background:#10233a;color:#69aaff;border-radius:11px;font-size:22px;margin-right:11px;vertical-align:middle}.brand-name{font-size:21px;font-weight:720;letter-spacing:-.04em}.brand-sub{display:block;color:var(--muted);font:11px var(--font);letter-spacing:.13em;margin:10px 0 0 50px}
section[data-testid="stSidebar"] p,section[data-testid="stSidebar"] label{font-size:15px!important;line-height:1.6}section[data-testid="stSidebar"] div[role="radiogroup"] label{padding:7px 5px;margin:3px 0;border-radius:8px}section[data-testid="stSidebar"] div[role="radiogroup"] label:hover{background:#142238}
.metric{background:var(--panel-raised);border:1px solid var(--line);border-radius:14px;padding:23px 24px;min-height:126px;box-shadow:0 5px 16px #0002}.metric-value{font:700 36px var(--font);letter-spacing:-.04em;color:#f3f7fc}.metric-label{font-size:14px;font-weight:550;color:#aebbd0;margin-top:9px}
.event{background:#101a27;border:1px solid #2b3c51;border-left:4px solid var(--amber);border-radius:11px;padding:18px 20px;margin:14px 0}.event p{margin:12px 0 8px;font-size:16px;line-height:1.65}.event small{color:#aab7c8;font-size:14px}.event-tag{font:600 12px var(--font);color:var(--amber);letter-spacing:.045em}
.empty{text-align:center;border:1px dashed #31445b;background:#0c141e;padding:54px 24px;border-radius:15px;margin:24px 0;color:#a6b4c7;line-height:1.9}.empty b{color:#e4ebf4;font-size:18px}.empty-icon{font-size:31px;color:#72a9e9;margin-bottom:12px}
div[data-testid="stMetric"]{background:var(--panel);padding:17px 19px;border:1px solid var(--line);border-radius:12px;min-height:100px}div[data-testid="stMetricLabel"]{color:#b3c0d1;font-size:14px}div[data-testid="stMetricValue"]{font-size:31px;font-weight:700;color:#f2f6fb}
.stButton>button,.stFormSubmitButton>button{min-height:44px;padding:8px 16px;border-radius:9px;border:1px solid #35475c;font-size:15px;font-weight:600;transition:background-color .15s,border-color .15s,color .15s}.stButton>button[kind="primary"],.stFormSubmitButton>button{min-height:50px;background:#1767c4;border:1px solid #3183e4;color:white;font-size:16px;font-weight:650}.stButton>button:hover,.stFormSubmitButton>button:hover{border-color:#69aaff;color:white;background:#18304b}.stButton>button[kind="primary"]:hover,.stFormSubmitButton>button:hover{background:#2479d9}
div[data-testid="stWidgetLabel"] p,label p{font-size:15px!important;font-weight:570!important;color:#d9e2ed!important;margin-bottom:5px}div[data-baseweb="input"] input,div[data-baseweb="textarea"] textarea,div[data-baseweb="select"] input{font-family:var(--font);font-size:16px!important;line-height:1.5}div[data-baseweb="input"]>div,div[data-baseweb="textarea"]>div,div[data-baseweb="select"]>div{background:#0b131e;border:1px solid #2b3b50;border-radius:9px}div[data-baseweb="textarea"] textarea{min-height:280px;line-height:1.65}
div[data-testid="stForm"]{padding:20px 21px 17px;background:#0b131e;border:1px solid #1f2d3e;border-radius:14px;margin:16px 0 24px}div[data-testid="stFileUploader"]{padding:15px 17px;background:#0d1622;border:1px solid #26364a;border-radius:12px}div[data-testid="stFileUploader"] section{background:#111c2a;border-radius:9px}
div[data-testid="stVerticalBlockBorderWrapper"]{border-color:#26364a!important;border-radius:13px!important;background:#0d1520;padding:4px}
div[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:11px;overflow:hidden}div[data-testid="stDataFrame"] [role="gridcell"],div[data-testid="stDataFrame"] [role="columnheader"]{font-size:14px!important}
div[data-testid="stHorizontalBlock"]{gap:1.15rem}
.alert-heading{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin:3px 0 5px}.alert-heading strong{font-size:19px;color:#f1f5fa}.priority-badge{display:inline-block;padding:5px 10px;border-radius:999px;font-size:13px;font-weight:700;letter-spacing:.01em;border:1px solid}.priority-low{color:#62dfa8;background:#10271f;border-color:#24583f}.priority-medium{color:#ffd074;background:#2d2412;border-color:#66501e}.priority-high{color:#ff8997;background:#311923;border-color:#693040}.alert-category{font-size:14px;color:#aab8ca;margin:3px 0 7px}.alert-title{font-size:16px;font-weight:600;color:#dfe7f1;line-height:1.5}.evidence-text{padding:11px 14px;margin:9px 0 5px;background:#111c29;border-left:3px solid #4f9cff;border-radius:0 8px 8px 0;color:#d1dce9;font-size:15px;line-height:1.65}
.alert-score{display:flex;flex-direction:column;gap:4px;padding:7px 0}.alert-score span,.alert-sentiment{font-size:13px;color:#aab8ca}.alert-score strong{font-size:28px;line-height:1.15;color:#f2f6fb}.alert-score small{font-size:13px;font-weight:500;color:#aab8ca}.alert-status{display:inline-block;width:max-content;padding:6px 11px;border:1px solid;border-radius:999px;font-size:13px;font-weight:700}.status-pending{color:#ffd074;background:#2d2412;border-color:#66501e}.status-reviewed{color:#85b9ff;background:#14263d;border-color:#31577f}.status-dismissed{color:#aeb9c8;background:#1a222e;border-color:#3b4859}.alert-sentiment{margin-top:8px}.alert-sentiment b{color:#e3ebf5}
div[data-testid="stPlotlyChart"]{background:#0c141e;border:1px solid #1e2c3d;border-radius:13px;padding:10px 10px 2px;margin:8px 0 18px}
div[data-testid="stFileUploader"] p{font-size:15px!important}
@media(max-width:900px){.block-container{padding:1.5rem 1.25rem 2rem}section[data-testid="stSidebar"]{min-width:235px}.metric{padding:19px;min-height:112px}.metric-value{font-size:32px}}
@media(max-width:620px){.block-container{padding:1.2rem .9rem 1.5rem}h1{font-size:29px!important}h2{font-size:22px!important}h3{font-size:19px!important}.metric{padding:15px;min-height:100px}.metric-value{font-size:29px}.stMarkdown p{font-size:15px}div[data-testid="stHorizontalBlock"]{gap:.6rem}div[data-testid="stForm"]{padding:14px}.stButton>button,.stFormSubmitButton>button{min-height:44px}}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-mark">◈</span><span class="brand-name">SignalDesk</span><span class="brand-sub">FINANCIAL INTELLIGENCE</span></div>', unsafe_allow_html=True)
    st.caption("DECISION SUPPORT · NLP")
    pages = ["Overview", "News Analyzer", "Risk Intelligence"]
    # Give navigation a stable widget key. Deriving the radio's default index
    # from another state key caused Streamlit to recreate it on form reruns.
    if "navigation_page" not in st.session_state:
        st.session_state.navigation_page = "Overview"
    page = st.radio("Navigation", pages, key="navigation_page", label_visibility="collapsed")
    st.markdown("---")
    st.caption("FinTech research prototype\n\nPotential event indicators are not verified facts or investment advice.")

render(page)
