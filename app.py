import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import re
import time

# ══════════════════════════════════════════════════════════════════════
# MATCHING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def levenshtein(a, b):
    a, b = a.upper(), b.upper()
    al, bl = len(a), len(b)
    if al == 0: return bl
    if bl == 0: return al
    dp = list(range(bl + 1))
    for i in range(1, al + 1):
        prev = i
        for j in range(1, bl + 1):
            tmp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else min(prev + 1, dp[j] + 1, dp[j-1] + 1)
            prev = tmp
    return dp[bl]

def token_sim(a, b):
    a, b = a.upper(), b.upper()
    if a == b: return "exact"
    max_l = max(len(a), len(b))
    len_diff = abs(len(a) - len(b))
    if len_diff > 3: return "none"
    dist = levenshtein(a, b)
    ratio = dist / max_l if max_l > 0 else 0
    if max_l >= 6:
        if dist == 1: return "close"
        if dist == 2 and ratio <= 0.35: return "close"
        if dist == 3 and ratio <= 0.40: return "fuzzy"
    if max_l >= 4:
        if dist == 1: return "close"
        if dist == 2 and ratio <= 0.50: return "fuzzy"
    if max_l >= 3:
        if dist == 1 and len(a) >= 3 and len(b) >= 3: return "fuzzy"
    return "none"

def normalize(s):
    return str(s or "").strip().upper().replace(r'\s+', ' ')

def get_name_token_results(i_name, m_name):
    i_tokens = [t for t in normalize(i_name).split() if len(t) > 1]
    m_tokens = [t for t in normalize(m_name).split() if len(t) > 1]
    perfect_count = 0
    fuzzy_count = 0
    for it in i_tokens:
        best = "none"
        for mt in m_tokens:
            t = token_sim(it, mt)
            if t in ["exact", "close"]: best = "perfect"
            elif t == "fuzzy" and best == "none": best = "fuzzy"
        if best == "perfect": perfect_count += 1
        elif best == "fuzzy": fuzzy_count += 1
    return perfect_count, fuzzy_count

def masked_visible_digits_match(i_clean, m_clean):
    if len(i_clean) < 10 or len(m_clean) < 10: return False
    offset = len(i_clean) - len(m_clean)
    visible_count = 0
    for k in range(len(m_clean)):
        if m_clean[k].upper() != 'X':
            visible_count += 1
            i_idx = offset + k
            if i_idx < 0 or i_idx >= len(i_clean) or m_clean[k] != i_clean[i_idx]:
                return False
    return visible_count >= 2

def compare_name_phone_single(master_names, input_name, master_phones, input_phone,
                               master_udises, input_udise):
    if not input_name or pd.isna(input_name):
        return 0, "", "", "", "No input name", ""
    i_udise = str(input_udise or "").strip()
    i_phone_r = str(input_phone or "").strip()
    i_clean = re.sub(r'\D', '', i_phone_r)
    best_show, best_name, best_phone, best_udise, best_rule, best_details = 0, "", "", "", "", ""
    for idx in range(len(master_names)):
        m_name = str(master_names[idx] or "").strip()
        m_phone_r = str(master_phones[idx] or "").strip()
        m_udise = str(master_udises[idx] or "").strip()
        udise_present = bool(i_udise)
        udise_match = udise_present and m_udise == i_udise
        if udise_present and not udise_match: continue
        m_is_ignored = m_phone_r.upper() == "IGNORE"
        m_is_masked = 'X' in m_phone_r.upper() or 'x' in m_phone_r
        m_clean = re.sub(r'[^0-9X]', '', m_phone_r, flags=re.IGNORECASE)
        m_is_unmasked = not m_is_masked and not m_is_ignored and len(m_clean) >= 10
        if m_is_unmasked and len(i_clean) >= 10:
            if i_clean[-10:] == m_clean[-10:]:
                return (100, m_name, m_phone_r, m_udise, "Stage3 bypass: unmasked phone exact", "S1:30 S2:bypassed S3:100")
        if not udise_match: continue
        stage1 = 30
        perfect_count, fuzzy_count = get_name_token_results(input_name, m_name)
        masked_ok = (m_is_masked and not m_is_ignored and len(i_clean) >= 10
                     and masked_visible_digits_match(i_clean, m_clean))
        addon, rule = 0, ""
        if perfect_count >= 2: addon, rule = 100, "at least 2 tokens perfectly"
        elif masked_ok and perfect_count >= 1: addon, rule = 100, "Phone visible digits + >=1 token perfectly"
        elif masked_ok and fuzzy_count >= 1: addon, rule = 50, "Phone visible digits + >=1 token fuzzy"
        elif perfect_count >= 1 and fuzzy_count >= 1: addon, rule = 50, ">=1 token perfectly + >=1 token fuzzy"
        elif fuzzy_count >= 2: addon, rule = 40, "at least 2 tokens fuzzily"
        elif perfect_count >= 1: addon, rule = 40, "Only >=1 token perfectly"
        elif fuzzy_count >= 1: addon, rule = 30, "Only >=1 token fuzzy"
        else: addon, rule = 0, "No name match"
        show_score = min(100, stage1 + addon)
        if show_score > best_show:
            best_show, best_name, best_phone, best_udise, best_rule, best_details = \
                show_score, m_name, m_phone_r, m_udise, rule, f"S1:{stage1} addon:{addon}"
        if best_show == 100: break
    return best_show, best_name, best_phone, best_udise, best_rule, best_details


# ══════════════════════════════════════════════════════════════════════
# FILE VALIDATION
# ══════════════════════════════════════════════════════════════════════

def validate_master_file(df):
    required_cols = ['UDISE', 'TEACHER_NAME', 'MOBILE_NO']
    if df.shape[1] < 3:
        return False, f"Master file must have at least 3 columns, found {df.shape[1]}"
    actual_cols = [str(c).strip().upper() for c in df.columns[:3]]
    if actual_cols != required_cols:
        return False, f"Master file columns must be: {required_cols}. Found: {list(df.columns[:3])}"
    return True, "Valid"

def validate_user_file(df):
    if df.shape[1] < 6:
        return False, f"User file must have at least 6 columns, found {df.shape[1]}"
    if 'FULL_NAME' not in df.columns:
        return False, "User file must have 'FULL_NAME' column"
    if 'MOBILE_NUMBER' not in df.columns:
        return False, "User file must have 'MOBILE_NUMBER' column"
    if 'UDISE_CODE' not in df.columns:
        return False, "User file must have 'UDISE_CODE' column (should be in column F)"
    return True, "Valid"


# ══════════════════════════════════════════════════════════════════════
# PROCESSING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def process_user_to_master(user_df, master_df, progress_bar, pct_text, status_text, start_pct, end_pct):
    master_names  = master_df['TEACHER_NAME'].values
    master_phones = master_df['MOBILE_NO'].values
    master_udises = master_df['UDISE'].values
    if 'IS_PROVISIONAL' in user_df.columns:
        prov_mask = user_df['IS_PROVISIONAL'].astype(str).str.strip().str.upper() == 'TRUE'
        prov_df = user_df[prov_mask].copy()
    else:
        prov_df = user_df.copy()
    results = []
    total = len(prov_df)
    start_time = time.time()
    for i, (idx, row) in enumerate(prov_df.iterrows()):
        score, matched_name, matched_phone, matched_udise, rule, details = \
            compare_name_phone_single(master_names, row['FULL_NAME'], master_phones,
                                      row['MOBILE_NUMBER'], master_udises, row['UDISE_CODE'])
        results.append({'Score': score, 'Matched_Name': matched_name, 'Matched_Phone': matched_phone,
                        'Matched_UDISE': matched_udise, 'Rule': rule, 'Details': details})
        if total > 0:
            frac = (i + 1) / total
            pct  = int(start_pct + frac * (end_pct - start_pct))
            elapsed   = time.time() - start_time
            remaining = int((elapsed / (i + 1)) * (total - i - 1)) if i > 0 else 0
            progress_bar.progress(pct)
            pct_text.markdown(f"<div class='pct-num'>{pct}%</div>", unsafe_allow_html=True)
            status_text.markdown(
                f"<div class='status-row'>"
                f"<span class='status-step'>Step 1 / 2 &nbsp;&middot;&nbsp; Matching User &rarr; Master</span>"
                f"<span class='status-stat'>{i+1} / {total} rows</span>"
                f"<span class='status-time'>&#9201; ~{remaining}s remaining</span>"
                f"</div>",
                unsafe_allow_html=True
            )
    for key in (results[0].keys() if results else []):
        prov_df[key] = [r[key] for r in results]
    return prov_df


def process_master_to_user(master_df, user_df, progress_bar, pct_text, status_text, start_pct, end_pct):
    user_names  = user_df['FULL_NAME'].values
    user_phones = user_df['MOBILE_NUMBER'].values
    user_udises = user_df['UDISE_CODE'].values
    results = []
    total = len(master_df)
    start_time = time.time()
    for i, (idx, row) in enumerate(master_df.iterrows()):
        score, matched_name, matched_phone, matched_udise, rule, details = \
            compare_name_phone_single(user_names, row['TEACHER_NAME'], user_phones,
                                      row['MOBILE_NO'], user_udises, row['UDISE'])
        results.append({'Score': score, 'Matched_Name': matched_name, 'Matched_Phone': matched_phone,
                        'Matched_UDISE': matched_udise, 'Rule': rule, 'Details': details})
        if total > 0:
            frac = (i + 1) / total
            pct  = int(start_pct + frac * (end_pct - start_pct))
            elapsed   = time.time() - start_time
            remaining = int((elapsed / (i + 1)) * (total - i - 1)) if i > 0 else 0
            progress_bar.progress(pct)
            pct_text.markdown(f"<div class='pct-num'>{pct}%</div>", unsafe_allow_html=True)
            status_text.markdown(
                f"<div class='status-row'>"
                f"<span class='status-step'>Step 2 / 2 &nbsp;&middot;&nbsp; Matching Master &rarr; User</span>"
                f"<span class='status-stat'>{i+1} / {total} rows</span>"
                f"<span class='status-time'>&#9201; ~{remaining}s remaining</span>"
                f"</div>",
                unsafe_allow_html=True
            )
    result_df = master_df.copy()
    for key in (results[0].keys() if results else []):
        result_df[key] = [r[key] for r in results]
    return result_df


def create_excel_download(df, sheet_name="Sheet1"):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output.getvalue()

def create_sample_master():
    df = pd.DataFrame({
        'UDISE': ['27110100101', '27110100102', '27110100103'],
        'TEACHER_NAME': ['RAMESH KUMAR SHARMA', 'SUNITA DEVI', 'MOHAN LAL GUPTA'],
        'MOBILE_NO': ['98XXXXXX01', '87XXXXXX12', '9876543210']
    })
    return create_excel_download(df, "Master_Sample")

def create_sample_user():
    df = pd.DataFrame({
        'COMMUNITY_USER_ID': ['USR001', 'USR002', 'USR003'],
        'FULL_NAME': ['RAMESH KUMAR SHARMA', 'SUNITA DEVI', 'MOHAN LAL GUPTA'],
        'MOBILE_NUMBER': ['9812345601', '8712345612', '9876543210'],
        'EMAIL': ['ramesh@example.com', 'sunita@example.com', 'mohan@example.com'],
        'SCHOOL_NAME': ['GOVT PRIMARY SCHOOL A', 'GOVT PRIMARY SCHOOL B', 'GOVT PRIMARY SCHOOL C'],
        'UDISE_CODE': ['27110100101', '27110100102', '27110100103'],
        'COMMUNITY_NAME': ['Community A', 'Community B', 'Community C'],
        'CIRCLE_NAME': ['Circle 1', 'Circle 1', 'Circle 2'],
        'BLOCK_NAME': ['Block X', 'Block X', 'Block Y'],
        'DISTRICT_NAME': ['District 1', 'District 1', 'District 2'],
        'IS_PROVISIONAL': ['True', 'False', 'True']
    })
    return create_excel_download(df, "User_Sample")


# ══════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stApp"],
    [data-testid="stMain"],
    [data-testid="stHeader"],
    section.main { background: #080c14 !important; }
    * { font-family: 'Plus Jakarta Sans', sans-serif !important; box-sizing: border-box; }
    #MainMenu, footer, header { visibility: hidden; }
    section[data-testid="stSidebar"] { display: none; }
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: #080c14; }
    ::-webkit-scrollbar-thumb { background: #1e2a45; border-radius: 3px; }

    /* HERO */
    .hero {
        position: relative; overflow: hidden;
        background: linear-gradient(135deg, #0d1526 0%, #111c36 50%, #0d1526 100%);
        border-bottom: 1px solid #1a2540;
        padding: 3rem 2rem 2.5rem; text-align: center;
    }
    .hero::before {
        content: ''; position: absolute;
        top: -80px; left: 50%; transform: translateX(-50%);
        width: 700px; height: 360px;
        background: radial-gradient(ellipse, rgba(99,102,241,0.2) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-eyebrow {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(99,102,241,0.12);
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 20px; padding: 4px 14px;
        font-size: 0.7rem; font-weight: 700; color: #818cf8;
        letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 1.1rem;
    }
    .hero-title {
        font-size: 2.5rem; font-weight: 800; color: #f0f4ff;
        letter-spacing: -0.03em; margin: 0 0 0.7rem; line-height: 1.1;
    }
    .hero-title .accent {
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-sub {
        font-size: 0.9rem; color: #4a5f8a; font-weight: 400;
        max-width: 480px; margin: 0 auto; line-height: 1.6;
    }

    /* SECTION LABEL */
    .sec-label {
        font-size: 0.65rem; font-weight: 800; color: #3a4d70;
        text-transform: uppercase; letter-spacing: 0.18em; margin: 2rem 0 1rem;
    }

    /* STEP BADGE */
    .step-badge {
        display: inline-flex; align-items: center; gap: 6px;
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(192,132,252,0.1));
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 20px; padding: 5px 14px;
        font-size: 0.72rem; font-weight: 700; color: #818cf8;
        letter-spacing: 0.06em; margin-bottom: 0.9rem;
    }

    /* FILE UPLOADER */
    [data-testid="stFileUploader"] > label {
        font-size: 1.05rem !important; font-weight: 700 !important;
        color: #c8d6f0 !important; letter-spacing: -0.01em !important;
        margin-bottom: 0.6rem !important;
    }
    [data-testid="stFileUploader"] { background: transparent !important; }
    [data-testid="stFileUploaderDropzone"] {
        background: #0d1526 !important;
        border: 1.5px dashed #1e2e50 !important;
        border-radius: 12px !important;
        transition: border-color 0.2s, background 0.2s !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #6366f1 !important;
        background: rgba(99,102,241,0.04) !important;
    }
    [data-testid="stFileUploaderDropzone"] p,
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] small {
        color: #3a4d70 !important; font-size: 0.82rem !important;
    }
    [data-testid="stFileUploaderDropzone"] svg { color: #3a4d70 !important; }

    /* DOWNLOAD LINK STYLE */
    .stDownloadButton > button {
        background: transparent !important; color: #6366f1 !important;
        border: none !important; padding: 0.25rem 0 !important;
        font-size: 0.78rem !important; font-weight: 600 !important;
        box-shadow: none !important; text-decoration: none !important;
        letter-spacing: 0.01em !important; opacity: 0.8 !important;
        transition: opacity 0.15s !important;
    }
    .stDownloadButton > button:hover {
        opacity: 1 !important; color: #818cf8 !important;
        background: transparent !important; transform: none !important;
    }

    /* EXPANDER */
    .stExpander {
        background: #0d1526 !important; border: 1px solid #1a2540 !important;
        border-radius: 10px !important;
    }
    .stExpander summary { color: #4a5f8a !important; font-size: 0.8rem !important; }
    .stExpander summary:hover { color: #818cf8 !important; }

    /* VALIDATION BANNER */
    .val-banner {
        display: flex; align-items: center; gap: 1rem;
        background: linear-gradient(135deg, #071a10, #0a2015);
        border: 1px solid #0f4028; border-radius: 12px;
        padding: 1rem 1.4rem; margin: 1.2rem 0;
    }
    .val-icon { font-size: 1.4rem; }
    .val-title { color: #34d399; font-size: 0.9rem; font-weight: 700; margin: 0; }
    .val-sub { color: #1a7a4a; font-size: 0.76rem; margin: 0.15rem 0 0; }

    /* INFO BANNER */
    .info-banner {
        background: #0a1020; border: 1px solid #1a2540;
        border-left: 3px solid #6366f1;
        border-radius: 10px; padding: 0.85rem 1.2rem; margin: 0.8rem 0 1.2rem;
    }
    .info-banner p { color: #4a5f8a; font-size: 0.82rem; margin: 0; }
    .info-banner strong { color: #818cf8; }

    /* CTA BUTTON */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: #fff !important; border: none !important;
        border-radius: 12px !important; padding: 0.75rem 2rem !important;
        font-weight: 700 !important; font-size: 0.95rem !important;
        letter-spacing: 0.01em !important;
        box-shadow: 0 4px 24px rgba(99,102,241,0.4) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #7c7ff5 0%, #a78bfa 100%) !important;
        box-shadow: 0 6px 32px rgba(99,102,241,0.55) !important;
        transform: translateY(-1px) !important;
    }

    /* PROGRESS PANEL */
    .progress-panel {
        background: linear-gradient(135deg, #0d1526 0%, #111c36 100%);
        border: 1px solid #1a2540; border-radius: 16px;
        padding: 1.8rem 2rem 1.4rem; margin: 1.5rem 0;
        position: relative; overflow: hidden;
    }
    .progress-panel::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #c084fc);
        border-radius: 16px 16px 0 0;
    }
    .progress-header {
        display: flex; align-items: center; justify-content: space-between;
        margin-bottom: 0.8rem;
    }
    .progress-title {
        font-size: 0.72rem; font-weight: 700; color: #3a4d70;
        text-transform: uppercase; letter-spacing: 0.14em;
    }
    .pct-num {
        font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(135deg, #818cf8, #c084fc);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1; font-family: 'JetBrains Mono', monospace !important;
        margin-bottom: 0.6rem;
    }
    .status-row {
        display: flex; align-items: center; gap: 0.8rem;
        margin-top: 0.7rem; flex-wrap: wrap;
    }
    .status-step { font-size: 0.82rem; color: #6b7a9e; font-weight: 500; }
    .status-stat {
        font-size: 0.75rem; color: #3a4d70;
        background: #0a1020; border: 1px solid #1a2540;
        border-radius: 6px; padding: 2px 8px;
        font-family: 'JetBrains Mono', monospace !important;
    }
    .status-time {
        font-size: 0.78rem; color: #8b5cf6; font-weight: 600;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* STREAMLIT PROGRESS BAR */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 60%, #c084fc 100%) !important;
        border-radius: 8px !important;
        box-shadow: 0 0 14px rgba(99,102,241,0.55) !important;
    }
    .stProgress > div > div > div {
        background: #0d1526 !important; border-radius: 8px !important;
        height: 10px !important; border: 1px solid #1a2540 !important;
    }
    .stProgress { margin: 0.4rem 0 0 !important; }

    /* COMPLETION BANNER */
    .done-banner {
        background: linear-gradient(135deg, #071a10 0%, #0c2318 100%);
        border: 1px solid #1a5c36; border-radius: 14px;
        padding: 1.6rem 2rem; text-align: center;
        margin: 1.5rem 0; position: relative; overflow: hidden;
    }
    .done-banner::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, #34d399, #10b981, #34d399);
    }
    .done-title { font-size: 1.25rem; font-weight: 800; color: #34d399; margin: 0 0 0.3rem; }
    .done-sub { font-size: 0.83rem; color: #1a7a4a; margin: 0; }

    /* METRIC CARDS */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #0d1526 0%, #111c36 100%) !important;
        border: 1px solid #1a2540 !important; border-radius: 14px !important;
        padding: 1.3rem 1.5rem !important; position: relative; overflow: hidden !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important; font-weight: 800 !important;
        color: #c8d6f0 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.68rem !important; font-weight: 700 !important;
        color: #3a4d70 !important; text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
    }
    [data-testid="stMetricDelta"] { font-size: 0.8rem !important; font-weight: 600 !important; }

    /* RESULT DOWNLOAD CARDS */
    .rcard {
        border-radius: 14px; padding: 1.4rem 1.5rem 1.1rem;
        margin-bottom: 0.6rem; position: relative; overflow: hidden;
    }
    .rcard::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    }
    .rcard.g { background: linear-gradient(160deg, #071a10 0%, #091f14 100%); border: 1px solid #0f4028; }
    .rcard.g::before { background: linear-gradient(90deg, #34d399, #10b981); }
    .rcard.a { background: linear-gradient(160deg, #1a1200 0%, #1f1600 100%); border: 1px solid #3d2e00; }
    .rcard.a::before { background: linear-gradient(90deg, #fbbf24, #f59e0b); }
    .rcard.r { background: linear-gradient(160deg, #1a0808 0%, #1f0a0a 100%); border: 1px solid #4a1515; }
    .rcard.r::before { background: linear-gradient(90deg, #f87171, #ef4444); }

    .rcard-icon { font-size: 1.5rem; margin-bottom: 0.45rem; display: block; }
    .rcard-title { font-size: 0.9rem; font-weight: 800; margin-bottom: 0.25rem; }
    .rcard.g .rcard-title { color: #34d399; }
    .rcard.a .rcard-title { color: #fbbf24; }
    .rcard.r .rcard-title { color: #f87171; }
    .rcard-desc { font-size: 0.75rem; line-height: 1.4; margin-bottom: 0.9rem; }
    .rcard.g .rcard-desc { color: #1a7a4a; }
    .rcard.a .rcard-desc { color: #7a5a00; }
    .rcard.r .rcard-desc { color: #7a2a2a; }

    /* Download btn inside result cards */
    .rcard .stDownloadButton > button {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px !important; padding: 0.5rem 1rem !important;
        color: #c8d6f0 !important; font-size: 0.8rem !important;
        font-weight: 600 !important; width: 100% !important;
        text-decoration: none !important; opacity: 1 !important;
        transition: all 0.2s !important;
    }
    .rcard.g .stDownloadButton > button:hover { background: rgba(52,211,153,0.12) !important; color: #34d399 !important; border-color: #34d399 !important; }
    .rcard.a .stDownloadButton > button:hover { background: rgba(251,191,36,0.12) !important; color: #fbbf24 !important; border-color: #fbbf24 !important; }
    .rcard.r .stDownloadButton > button:hover { background: rgba(248,113,113,0.12) !important; color: #f87171 !important; border-color: #f87171 !important; }

    /* TABS */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important; gap: 4px !important;
        border-bottom: 1px solid #1a2540 !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important; color: #3a4d70 !important;
        border-radius: 8px 8px 0 0 !important; font-size: 0.82rem !important;
        font-weight: 600 !important; padding: 10px 18px !important;
        border-bottom: 2px solid transparent !important;
    }
    .stTabs [aria-selected="true"] {
        background: #0d1526 !important; color: #c8d6f0 !important;
        border-bottom: 2px solid #6366f1 !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background: #0d1526 !important; border: 1px solid #1a2540 !important;
        border-top: none !important; border-radius: 0 0 12px 12px !important;
        padding: 1.2rem !important;
    }

    /* GETTING STARTED */
    .gstart {
        background: #0a1020; border: 1.5px dashed #1a2540;
        border-radius: 16px; padding: 3rem 2rem;
        text-align: center; margin: 1.5rem 0 3rem;
    }
    .gstart-icon { font-size: 2.8rem; margin-bottom: 0.8rem; opacity: 0.3; }
    .gstart-title { font-size: 0.95rem; font-weight: 700; color: #2a3a5a; margin-bottom: 0.4rem; }
    .gstart-text { font-size: 0.82rem; color: #1a2540; }
    .gstart-text strong { color: #2a3a5a; }

    /* MISC */
    [data-testid="column"] { padding: 0 0.5rem; }
    .err-box { background: #1a0808; border: 1px solid #4a1515; border-radius: 10px; padding: 0.9rem 1.3rem; margin-top: 1rem; }
    .err-box p { color: #f87171; font-size: 0.86rem; margin: 0; }
    </style>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="Teacher Autoverification Tool",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    inject_css()

    # Session state
    for key in ['processing_done', 'auto_verify', 'not_verified', 'not_registered', 'prov_count', 'master_count']:
        if key not in st.session_state:
            st.session_state[key] = None

    # ━━━ HERO ━━━
    st.markdown("""
        <div class="hero">
            <div class="hero-eyebrow">&#x26A1; Token Fuzzy Matching engiene</div>
            <h1 class="hero-title">Teacher <span class="accent">Smart Autoverification</span> Tool</h1>
            <p class="hero-sub">Upload your master list and user list , and process in one click.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # Reset button
    if st.session_state.processing_done:
        col_r1, col_r2, col_r3 = st.columns([2.5, 1, 2.5])
        with col_r2:
            if st.button("&#x1F504; New Verification", use_container_width=True):
                for key in ['processing_done', 'auto_verify', 'not_verified', 'not_registered', 'prov_count', 'master_count']:
                    st.session_state[key] = None
                st.rerun()
        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

    # ━━━ UPLOAD SECTION ━━━
    if not st.session_state.processing_done:
        col1, col2 = st.columns(2, gap="medium")

        with col1:
            st.markdown("<div class='step-badge'>&#x1F4CA; Step 1 &nbsp;&mdash;&nbsp; Master File</div>", unsafe_allow_html=True)
            with st.expander("ℹ️ View Expected Format"):
                st.markdown("""
                **Required Columns (in order):**
                1. `UDISE` — School UDISE code
                2. `TEACHER_NAME` — Full name of teacher
                3. `MOBILE_NO` — Mobile number (can be masked with X)

                **Example:** `27110100101` · `RAMESH KUMAR SHARMA` · `98XXXXXX01`
                """)
            master_file = st.file_uploader("Upload Master List", type=['xlsx', 'xls'], key='master')
            st.download_button(
                "⬇ Download sample master format",
                data=create_sample_master(),
                file_name="Sample_Master_File.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        with col2:
            st.markdown("<div class='step-badge'>&#x1F465; Step 2 &nbsp;&mdash;&nbsp; User List</div>", unsafe_allow_html=True)
            with st.expander("ℹ️ View Expected Format"):
                st.markdown("""
                **Required Columns:**
                - `FULL_NAME` · `MOBILE_NUMBER` · `UDISE_CODE` (col F) · `IS_PROVISIONAL`

                **Note:** Export directly from ticklinks without any modifications.
                """)
            user_file = st.file_uploader("Upload User List", type=['xlsx', 'xls'], key='user')
            st.download_button(
                "⬇ Download sample user list format",
                data=create_sample_user(),
                file_name="Sample_User_File.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    # ━━━ RESULTS ━━━
    if st.session_state.processing_done:
        auto_verify    = st.session_state.auto_verify
        not_verified   = st.session_state.not_verified
        not_registered = st.session_state.not_registered

        st.markdown("""
            <div class="done-banner">
                <div class="done-title">&#x2705; Verification Complete</div>
                <div class="done-sub">All records matched, scored and categorised &mdash; download your results below</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='sec-label'>Verification Summary</div>", unsafe_allow_html=True)
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Total Processed", st.session_state.prov_count)
        with mc2:
            st.metric("Auto Verified", len(auto_verify),
                      delta=f"{(len(auto_verify)/max(st.session_state.prov_count,1)*100):.1f}%")
        with mc3:
            st.metric("Manual Review", len(not_verified),
                      delta=f"{(len(not_verified)/max(st.session_state.prov_count,1)*100):.1f}%",
                      delta_color="inverse")
        with mc4:
            st.metric("Not Registered", len(not_registered))

        st.markdown("<div class='sec-label'>Download Results</div>", unsafe_allow_html=True)
        dc1, dc2, dc3 = st.columns(3)

        with dc1:
            st.markdown("""
                <div class='rcard g'>
                    <span class='rcard-icon'>&#x2705;</span>
                    <div class='rcard-title'>Auto Verified</div>
                    <div class='rcard-desc'>High-confidence matches ready for automatic approval</div>
                </div>
            """, unsafe_allow_html=True)
            if len(auto_verify) > 0:
                st.download_button(f"Download  \u00b7  {len(auto_verify)} records",
                    data=create_excel_download(auto_verify, "Auto_Verify"),
                    file_name="Auto_Verify_This.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_auto", use_container_width=True)
            else:
                st.markdown("<p style='color:#1a7a4a;font-size:0.78rem;margin:0;'>No auto-verified records</p>", unsafe_allow_html=True)

        with dc2:
            st.markdown("""
                <div class='rcard a'>
                    <span class='rcard-icon'>&#x26A0;&#xFE0F;</span>
                    <div class='rcard-title'>Manual Review</div>
                    <div class='rcard-desc'>Lower-confidence matches &mdash; needs human verification</div>
                </div>
            """, unsafe_allow_html=True)
            if len(not_verified) > 0:
                st.download_button(f"Download  \u00b7  {len(not_verified)} records",
                    data=create_excel_download(not_verified, "Not_Verified"),
                    file_name="Manual_Review_Required.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_notv", use_container_width=True)
            else:
                st.markdown("<p style='color:#7a5a00;font-size:0.78rem;margin:0;'>No records to review</p>", unsafe_allow_html=True)

        with dc3:
            st.markdown("""
                <div class='rcard r'>
                    <span class='rcard-icon'>&#x274C;</span>
                    <div class='rcard-title'>Not Registered</div>
                    <div class='rcard-desc'>In master list but missing from the user database</div>
                </div>
            """, unsafe_allow_html=True)
            if len(not_registered) > 0:
                st.download_button(f"Download  \u00b7  {len(not_registered)} records",
                    data=create_excel_download(not_registered, "Not_Registered"),
                    file_name="Not_Registered.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_notr", use_container_width=True)
            else:
                st.markdown("<p style='color:#7a2a2a;font-size:0.78rem;margin:0;'>All records registered</p>", unsafe_allow_html=True)

        st.markdown("<div class='sec-label'>Data Preview</div>", unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs([
            f"Auto Verified ({len(auto_verify)})",
            f"Manual Review ({len(not_verified)})",
            f"Not Registered ({len(not_registered)})"
        ])
        with tab1:
            if len(auto_verify) > 0:
                st.dataframe(auto_verify.head(50), use_container_width=True, height=380)
                if len(auto_verify) > 50: st.caption(f"Showing first 50 of {len(auto_verify)} records.")
            else: st.caption("No auto-verified records to display.")
        with tab2:
            if len(not_verified) > 0:
                st.dataframe(not_verified.head(50), use_container_width=True, height=380)
                if len(not_verified) > 50: st.caption(f"Showing first 50 of {len(not_verified)} records.")
            else: st.caption("No records requiring manual review.")
        with tab3:
            if len(not_registered) > 0:
                st.dataframe(not_registered.head(50), use_container_width=True, height=380)
                if len(not_registered) > 50: st.caption(f"Showing first 50 of {len(not_registered)} records.")
            else: st.caption("All teachers are registered.")

        return

    # ━━━ PROCESSING TRIGGER ━━━
    if master_file and user_file:
        try:
            master_df = pd.read_excel(master_file)
            user_df   = pd.read_excel(user_file)

            master_valid, master_msg = validate_master_file(master_df)
            user_valid,   user_msg   = validate_user_file(user_df)

            if not master_valid:
                st.markdown(f"<div class='err-box'><p>Master File Error: {master_msg}</p></div>", unsafe_allow_html=True)
                return
            if not user_valid:
                st.markdown(f"<div class='err-box'><p>User File Error: {user_msg}</p></div>", unsafe_allow_html=True)
                return

            st.markdown(f"""
                <div class="val-banner">
                    <span class="val-icon">&#x2705;</span>
                    <div>
                        <p class="val-title">Files validated and loaded successfully</p>
                        <p class="val-sub">Master: {len(master_df)} rows &nbsp;&middot;&nbsp; User List: {len(user_df)} rows</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            if 'IS_PROVISIONAL' in user_df.columns:
                prov_count = (user_df['IS_PROVISIONAL'].astype(str).str.strip().str.upper() == 'TRUE').sum()
                st.markdown(f"""
                    <div class="info-banner">
                        <p>&#x2139;&#xFE0F; &nbsp;Will process <strong>{prov_count}</strong> provisional records out of {len(user_df)} total</p>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
            col_b1, col_b2, col_b3 = st.columns([2, 1.4, 2])
            with col_b2:
                start_clicked = st.button("&#x1F680; Start Verification", use_container_width=True)

            if start_clicked:
                # Progress panel header (static)
                st.markdown("""
                    <div class="progress-panel">
                        <div class="progress-header">
                            <span class="progress-title">&#x26A1; Verification in Progress</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # Live-updating elements
                pct_text     = st.empty()
                progress_bar = st.progress(0)
                status_text  = st.empty()

                pct_text.markdown("<div class='pct-num'>0%</div>", unsafe_allow_html=True)
                status_text.markdown(
                    "<div class='status-row'><span class='status-step'>Initialising matching engine&hellip;</span></div>",
                    unsafe_allow_html=True
                )

                prov_results = process_user_to_master(
                    user_df, master_df, progress_bar, pct_text, status_text, 0, 50
                )

                master_results = process_master_to_user(
                    master_df, user_df, progress_bar, pct_text, status_text, 50, 100
                )

                progress_bar.progress(100)
                pct_text.markdown(
                    "<div class='pct-num' style='background:linear-gradient(135deg,#34d399,#10b981);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;'>100%</div>",
                    unsafe_allow_html=True
                )
                status_text.markdown(
                    "<div class='status-row'><span class='status-step' style='color:#34d399;font-weight:700;'>&#x2705; Processing complete!</span><span class='status-time'>0s remaining</span></div>",
                    unsafe_allow_html=True
                )
                time.sleep(0.7)

                auto_verify    = prov_results[prov_results['Score'] >= 70] if len(prov_results) > 0 else pd.DataFrame()
                not_verified   = prov_results[prov_results['Score'] < 70]  if len(prov_results) > 0 else pd.DataFrame()
                not_registered = master_results[master_results['Score'] < 70]

                st.session_state.processing_done = True
                st.session_state.auto_verify      = auto_verify
                st.session_state.not_verified     = not_verified
                st.session_state.not_registered   = not_registered
                st.session_state.prov_count       = len(prov_results)
                st.session_state.master_count     = len(master_df)
                st.rerun()

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            with st.expander("View Error Details"):
                st.exception(e)

    else:
        st.markdown("""
            <div class="gstart">
                <div class="gstart-icon">&#x2601;&#xFE0F;</div>
                <div class="gstart-title">Ready when you are</div>
                <div class="gstart-text">Upload <strong>Master List</strong> and <strong>User List</strong> above to begin</div>
            </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
