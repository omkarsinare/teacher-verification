import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import re
import time

# ══════════════════════════════════════════════════════════════════════
# MATCHING FUNCTION
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
    if a == b:
        return "exact"
    max_l = max(len(a), len(b))
    len_diff = abs(len(a) - len(b))
    if len_diff > 3:
        return "none"
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
            if t in ["exact", "close"]:
                best = "perfect"
            elif t == "fuzzy" and best == "none":
                best = "fuzzy"
        if best == "perfect":
            perfect_count += 1
        elif best == "fuzzy":
            fuzzy_count += 1
    return perfect_count, fuzzy_count


def masked_visible_digits_match(i_clean, m_clean):
    if len(i_clean) < 10 or len(m_clean) < 10:
        return False
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
    best_show = 0
    best_name = ""
    best_phone = ""
    best_udise = ""
    best_rule = ""
    best_details = ""
    for idx in range(len(master_names)):
        m_name = str(master_names[idx] or "").strip()
        m_phone_r = str(master_phones[idx] or "").strip()
        m_udise = str(master_udises[idx] or "").strip()
        udise_present = bool(i_udise)
        udise_match = udise_present and m_udise == i_udise
        if udise_present and not udise_match:
            continue
        m_is_ignored = m_phone_r.upper() == "IGNORE"
        m_is_masked = 'X' in m_phone_r.upper() or 'x' in m_phone_r
        m_clean = re.sub(r'[^0-9X]', '', m_phone_r, flags=re.IGNORECASE)
        m_is_unmasked = not m_is_masked and not m_is_ignored and len(m_clean) >= 10
        if m_is_unmasked and len(i_clean) >= 10:
            if i_clean[-10:] == m_clean[-10:]:
                return (100, m_name, m_phone_r, m_udise,
                        "Stage3 bypass: unmasked phone exact",
                        "S1:30 S2:bypassed S3:100")
        if not udise_match:
            continue
        stage1 = 30
        perfect_count, fuzzy_count = get_name_token_results(input_name, m_name)
        masked_ok = (m_is_masked and not m_is_ignored and len(i_clean) >= 10
                     and masked_visible_digits_match(i_clean, m_clean))
        addon = 0
        rule = ""
        if perfect_count >= 2:
            addon = 100
            rule = "at least 2 tokens perfectly"
        elif masked_ok and perfect_count >= 1:
            addon = 100
            rule = "Phone visible digits + ≥1 token perfectly"
        elif masked_ok and fuzzy_count >= 1:
            addon = 50
            rule = "Phone visible digits + ≥1 token fuzzy"
        elif perfect_count >= 1 and fuzzy_count >= 1:
            addon = 50
            rule = "≥1 token perfectly + ≥1 token fuzzy"
        elif fuzzy_count >= 2:
            addon = 40
            rule = "at least 2 tokens fuzzily"
        elif perfect_count >= 1:
            addon = 40
            rule = "Only ≥1 token perfectly"
        elif fuzzy_count >= 1:
            addon = 30
            rule = "Only ≥1 token fuzzy"
        else:
            addon = 0
            rule = "No name match"
        final_score = stage1 + addon
        show_score = min(100, final_score)
        if show_score > best_show:
            best_show = show_score
            best_name = m_name
            best_phone = m_phone_r
            best_udise = m_udise
            best_rule = rule
            best_details = f"S1:{stage1} addon:{addon} raw:{final_score}"
        if best_show == 100:
            break
    return best_show, best_name, best_phone, best_udise, best_rule, best_details


# ══════════════════════════════════════════════════════════════════════
# FILE VALIDATION
# ══════════════════════════════════════════════════════════════════════

def validate_master_file(df):
    required_cols = ['UDISE', 'TEACHER_NAME', 'MOBILE_NO']
    if df.shape[1] < 3:
        return False, f"Master file must have at least 3 columns, found {df.shape[1]}"
    actual_cols = [str(c).strip().upper() for c in df.columns[:3]]
    expected_cols = [c.upper() for c in required_cols]
    if actual_cols != expected_cols:
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
# MAIN PROCESSING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def process_user_to_master(user_df, master_df, progress_bar, progress_text, start_pct, end_pct):
    master_names = master_df['TEACHER_NAME'].values
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
        input_name = row['FULL_NAME']
        input_phone = row['MOBILE_NUMBER']
        input_udise = row['UDISE_CODE']

        score, matched_name, matched_phone, matched_udise, rule, details = \
            compare_name_phone_single(
                master_names, input_name,
                master_phones, input_phone,
                master_udises, input_udise
            )

        results.append({
            'Score': score,
            'Matched_Name': matched_name,
            'Matched_Phone': matched_phone,
            'Matched_UDISE': matched_udise,
            'Rule': rule,
            'Details': details
        })

        if total > 0:
            frac = (i + 1) / total
            pct = start_pct + frac * (end_pct - start_pct)
            progress_bar.progress(int(pct))
            elapsed = time.time() - start_time
            remaining = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else 0
            progress_text.markdown(
                f"<p style='color:#a0aec0; font-size:0.9rem; text-align:center;'>"
                f"Step 1/2: Matching User → Master &nbsp;|&nbsp; {i+1}/{total} rows &nbsp;|&nbsp; ⏱ ~{int(remaining)}s remaining"
                f"</p>",
                unsafe_allow_html=True
            )

    for key in results[0].keys() if results else []:
        prov_df[key] = [r[key] for r in results]

    return prov_df


def process_master_to_user(master_df, user_df, progress_bar, progress_text, start_pct, end_pct):
    user_names = user_df['FULL_NAME'].values
    user_phones = user_df['MOBILE_NUMBER'].values
    user_udises = user_df['UDISE_CODE'].values

    results = []
    total = len(master_df)
    start_time = time.time()

    for i, (idx, row) in enumerate(master_df.iterrows()):
        input_name = row['TEACHER_NAME']
        input_phone = row['MOBILE_NO']
        input_udise = row['UDISE']

        score, matched_name, matched_phone, matched_udise, rule, details = \
            compare_name_phone_single(
                user_names, input_name,
                user_phones, input_phone,
                user_udises, input_udise
            )

        results.append({
            'Score': score,
            'Matched_Name': matched_name,
            'Matched_Phone': matched_phone,
            'Matched_UDISE': matched_udise,
            'Rule': rule,
            'Details': details
        })

        if total > 0:
            frac = (i + 1) / total
            pct = start_pct + frac * (end_pct - start_pct)
            progress_bar.progress(int(pct))
            elapsed = time.time() - start_time
            remaining = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else 0
            progress_text.markdown(
                f"<p style='color:#a0aec0; font-size:0.9rem; text-align:center;'>"
                f"Step 2/2: Matching Master → User &nbsp;|&nbsp; {i+1}/{total} rows &nbsp;|&nbsp; ⏱ ~{int(remaining)}s remaining"
                f"</p>",
                unsafe_allow_html=True
            )

    result_df = master_df.copy()
    for key in results[0].keys() if results else []:
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

def main():
    st.set_page_config(
        page_title="Teacher Autoverification Tool",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # ── Dark Theme CSS matching PPT design ──
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

        /* ── Global Reset & Dark Background ── */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
            background-color: #0f1117 !important;
            font-family: 'DM Sans', sans-serif !important;
        }

        [data-testid="stMain"] {
            background-color: #0f1117 !important;
        }

        [data-testid="stHeader"] {
            background-color: #0f1117 !important;
        }

        section[data-testid="stSidebar"] {
            display: none;
        }

        /* ── Hide Streamlit chrome ── */
        #MainMenu, footer, header { visibility: hidden; }

        /* ── Typography ── */
        h1, h2, h3, h4, p, label, span, div {
            font-family: 'DM Sans', sans-serif !important;
        }

        /* ── Page title bar ── */
        .page-topbar {
            background-color: #161b27;
            border-bottom: 1px solid #2d3650;
            padding: 10px 0 6px 0;
            text-align: center;
            margin-bottom: 0;
        }
        .page-topbar-label {
            font-size: 0.7rem;
            color: #6b7a9e;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-weight: 500;
        }

        /* ── Hero header ── */
        .hero-section {
            background: linear-gradient(160deg, #161b27 0%, #1a2035 60%, #161b27 100%);
            border-bottom: 1px solid #2d3650;
            padding: 2.5rem 2rem 2rem 2rem;
            text-align: center;
            margin-bottom: 0;
        }

        .hero-title {
            font-size: 2rem;
            font-weight: 700;
            color: #e8ecf4;
            letter-spacing: -0.02em;
            margin: 0 0 0.4rem 0;
            line-height: 1.2;
        }

        .hero-icon {
            font-size: 1.8rem;
            margin-right: 0.5rem;
            vertical-align: middle;
        }

        .hero-subtitle {
            font-size: 0.88rem;
            color: #6b7a9e;
            font-weight: 400;
            margin: 0;
            max-width: 560px;
            margin: 0 auto;
            line-height: 1.5;
        }

        /* ── Main content area ── */
        .main-content {
            padding: 2rem 2rem 3rem 2rem;
            max-width: 1100px;
            margin: 0 auto;
        }

        /* ── Step badge above uploader ── */
        .step-badge {
            display: inline-block;
            background: #1e2540;
            border: 1px solid #2d3650;
            border-radius: 20px;
            padding: 0.35rem 0.9rem;
            font-size: 0.75rem;
            font-weight: 600;
            color: #818cf8;
            letter-spacing: 0.06em;
            margin-bottom: 0.8rem;
        }

        /* ── File uploader label as heading ── */
        [data-testid="stFileUploader"] > label {
            font-size: 1rem !important;
            font-weight: 700 !important;
            color: #e8ecf4 !important;
            letter-spacing: -0.01em !important;
            margin-bottom: 0.5rem !important;
        }

        /* ── File uploader dark styling ── */
        [data-testid="stFileUploader"] {
            background: #0f1117 !important;
            border: 1.5px dashed #2d3650 !important;
            border-radius: 10px !important;
        }

        [data-testid="stFileUploader"] label {
            color: #a0aec0 !important;
            font-size: 0.85rem !important;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: #0f1117 !important;
            border: none !important;
        }

        [data-testid="stFileUploaderDropzone"] p {
            color: #6b7a9e !important;
            font-size: 0.82rem !important;
        }

        /* ── File loaded pill ── */
        .file-pill {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: #1e2540;
            border: 1px solid #2d3650;
            border-radius: 8px;
            padding: 0.6rem 1rem;
            margin-bottom: 0.5rem;
        }

        .file-pill-name {
            font-size: 0.84rem;
            color: #c8d0e7;
            font-weight: 500;
            font-family: 'DM Mono', monospace !important;
        }

        .file-pill-size {
            font-size: 0.75rem;
            color: #4a5578;
        }

        /* ── Download sample link style ── */
        .stDownloadButton > button {
            background: transparent !important;
            color: #5b7cf6 !important;
            border: none !important;
            padding: 0.3rem 0 !important;
            font-size: 0.82rem !important;
            font-weight: 500 !important;
            box-shadow: none !important;
            text-decoration: underline !important;
            text-underline-offset: 3px !important;
            width: auto !important;
        }

        .stDownloadButton > button:hover {
            color: #7b9bff !important;
            background: transparent !important;
            transform: none !important;
        }

        /* ── Validation success banner ── */
        .validation-banner {
            background: #0e1f1a;
            border: 1px solid #1a4a35;
            border-radius: 10px;
            padding: 1rem 1.4rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin: 1.2rem 0;
        }

        .validation-banner-icon { font-size: 1.1rem; }

        .validation-banner-text {
            font-size: 0.88rem;
            color: #4ade80;
            font-weight: 500;
            margin: 0;
        }

        .validation-banner-sub {
            font-size: 0.78rem;
            color: #2d7a52;
            margin: 0;
        }

        /* ── Info banner ── */
        .info-banner {
            background: #111827;
            border: 1px solid #2d3650;
            border-radius: 10px;
            padding: 0.9rem 1.4rem;
            margin: 0.8rem 0 1.2rem 0;
        }

        .info-banner-text {
            font-size: 0.84rem;
            color: #a0aec0;
            margin: 0;
        }

        .info-banner-text strong {
            color: #c8d0e7;
        }

        /* ── CTA Button ── */
        .stButton > button {
            background: #5b7cf6 !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.7rem 2.5rem !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            letter-spacing: 0.01em !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 4px 20px rgba(91, 124, 246, 0.3) !important;
        }

        .stButton > button:hover {
            background: #7b9bff !important;
            box-shadow: 0 6px 24px rgba(91, 124, 246, 0.45) !important;
            transform: translateY(-1px) !important;
        }

        /* ── Progress section ── */
        .progress-card {
            background: #161b27;
            border: 1px solid #2d3650;
            border-radius: 14px;
            padding: 1.8rem 2rem;
            margin: 1.5rem 0;
        }

        .progress-label {
            font-size: 0.82rem;
            font-weight: 600;
            color: #6b7a9e;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 1.2rem;
        }

        .progress-steps {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }

        /* Streamlit progress bar */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #5b7cf6 0%, #818cf8 100%) !important;
            border-radius: 6px !important;
        }

        .stProgress > div > div > div {
            background: #1e2540 !important;
            border-radius: 6px !important;
        }

        /* ── Completion success ── */
        .completion-banner {
            background: #0e1f1a;
            border: 1px solid #1a4a35;
            border-radius: 12px;
            padding: 1.5rem 2rem;
            text-align: center;
            margin: 1.5rem 0;
        }

        .completion-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #4ade80;
            margin: 0 0 0.3rem 0;
        }

        .completion-sub {
            font-size: 0.85rem;
            color: #2d7a52;
            margin: 0;
        }

        /* ── Section heading ── */
        .section-heading {
            font-size: 0.75rem;
            font-weight: 700;
            color: #6b7a9e;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin: 2rem 0 1rem 0;
        }

        /* ── Metric cards ── */
        .metric-row {
            display: flex;
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        [data-testid="stMetric"] {
            background: #161b27 !important;
            border: 1px solid #2d3650 !important;
            border-radius: 12px !important;
            padding: 1.2rem 1.5rem !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 2rem !important;
            font-weight: 700 !important;
            color: #e8ecf4 !important;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.78rem !important;
            font-weight: 600 !important;
            color: #6b7a9e !important;
            text-transform: uppercase !important;
            letter-spacing: 0.08em !important;
        }

        [data-testid="stMetricDelta"] {
            font-size: 0.82rem !important;
            font-weight: 500 !important;
        }

        /* ── Download result cards ── */
        .result-card {
            border-radius: 12px;
            padding: 1.3rem 1.5rem 1rem 1.5rem;
            margin-bottom: 0.8rem;
        }

        .result-card.green {
            background: #0a1f15;
            border: 1px solid #1a4a35;
        }

        .result-card.amber {
            background: #1a1500;
            border: 1px solid #3d3000;
        }

        .result-card.red {
            background: #1f0a0a;
            border: 1px solid #4a1a1a;
        }

        .result-card-title {
            font-size: 0.88rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }

        .result-card.green .result-card-title { color: #4ade80; }
        .result-card.amber .result-card-title { color: #fbbf24; }
        .result-card.red .result-card-title { color: #f87171; }

        .result-card-desc {
            font-size: 0.78rem;
            line-height: 1.4;
            margin-bottom: 0.8rem;
        }

        .result-card.green .result-card-desc { color: #2d7a52; }
        .result-card.amber .result-card-desc { color: #7a6220; }
        .result-card.red .result-card-desc { color: #7a3030; }

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {
            background: #0f1117 !important;
            gap: 4px !important;
            border-bottom: 1px solid #2d3650 !important;
        }

        .stTabs [data-baseweb="tab"] {
            background: transparent !important;
            color: #6b7a9e !important;
            border-radius: 8px 8px 0 0 !important;
            font-size: 0.85rem !important;
            font-weight: 500 !important;
            padding: 10px 20px !important;
            border-bottom: 2px solid transparent !important;
        }

        .stTabs [aria-selected="true"] {
            background: #161b27 !important;
            color: #e8ecf4 !important;
            border-bottom: 2px solid #5b7cf6 !important;
        }

        .stTabs [data-baseweb="tab-panel"] {
            background: #161b27 !important;
            border: 1px solid #2d3650 !important;
            border-top: none !important;
            border-radius: 0 0 12px 12px !important;
            padding: 1.2rem !important;
        }

        /* ── Dataframe ── */
        .stDataFrame {
            border-radius: 8px !important;
            overflow: hidden !important;
        }

        /* ── Expander ── */
        .stExpander {
            background: #161b27 !important;
            border: 1px solid #2d3650 !important;
            border-radius: 10px !important;
        }

        .stExpander summary {
            color: #a0aec0 !important;
            font-size: 0.84rem !important;
        }

        /* ── Error/Alert ── */
        .stAlert {
            background: #1f0a0a !important;
            border: 1px solid #4a1a1a !important;
            border-radius: 10px !important;
            color: #f87171 !important;
        }

        /* ── Reset button ── */
        .reset-btn-wrap {
            display: flex;
            justify-content: center;
            margin-bottom: 1.5rem;
        }

        /* ── Getting started placeholder ── */
        .getting-started {
            background: #161b27;
            border: 1.5px dashed #2d3650;
            border-radius: 14px;
            padding: 3rem 2rem;
            text-align: center;
            margin: 1.5rem 0 3rem 0;
        }

        .getting-started-title {
            font-size: 1rem;
            font-weight: 600;
            color: #4a5578;
            margin-bottom: 0.5rem;
        }

        .getting-started-text {
            font-size: 0.88rem;
            color: #2d3650;
        }

        /* Column gap fix */
        [data-testid="column"] { padding: 0 0.6rem; }

        /* Dark scrollbar */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: #0f1117; }
        ::-webkit-scrollbar-thumb { background: #2d3650; border-radius: 3px; }
        </style>
    """, unsafe_allow_html=True)

    # ── Session State Init ──
    for key in ['processing_done', 'auto_verify', 'not_verified', 'not_registered',
                'prov_count', 'master_count']:
        if key not in st.session_state:
            st.session_state[key] = None

    # ── Hero Header ──
    st.markdown("""
        <div class='hero-section'>
            <div style='display:flex;align-items:center;justify-content:center;gap:0.5rem;margin-bottom:0.6rem;'>
                <span style='font-size:1.6rem;'>🎓</span>
                <h1 class='hero-title'>Teacher Autoverification Tool</h1>
            </div>
            <p class='hero-subtitle'>
                Intelligent matching system for teacher verification with advanced fuzzy matching algorithms
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── Reset after processing ──
    if st.session_state.processing_done:
        col_r1, col_r2, col_r3 = st.columns([2, 1, 2])
        with col_r2:
            if st.button("🔄 Start New Verification", use_container_width=True):
                for key in ['processing_done', 'auto_verify', 'not_verified', 'not_registered',
                            'prov_count', 'master_count']:
                    st.session_state[key] = None
                st.rerun()
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # UPLOAD SECTION — always visible until processing done
    # ══════════════════════════════════════════════════════════════════

    if not st.session_state.processing_done:
        col1, col2 = st.columns(2, gap="medium")

        with col1:
            st.markdown("<div class='step-badge'>📊 Step 1 &nbsp;—&nbsp; Master File</div>", unsafe_allow_html=True)

            with st.expander("ℹ️ View Expected Format"):
                st.markdown("""
                **Required Columns (in order):**
                1. `UDISE` — School UDISE code
                2. `TEACHER_NAME` — Full name of teacher
                3. `MOBILE_NO` — Mobile number (can be masked with X)

                **Example:**
                - UDISE: `27110100101`
                - TEACHER_NAME: `RAMESH KUMAR SHARMA`
                - MOBILE_NO: `98XXXXXX01`
                """)

            master_file = st.file_uploader(
                "Upload Master List",
                type=['xlsx', 'xls'],
                key='master',
                help="Upload the master teacher database file"
            )

            st.download_button(
                label="⬇ Download Sample Master Format",
                data=create_sample_master(),
                file_name="Sample_Master_File.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        with col2:
            st.markdown("<div class='step-badge'>👥 Step 2 &nbsp;—&nbsp; User List</div>", unsafe_allow_html=True)

            with st.expander("ℹ️ View Expected Format"):
                st.markdown("""
                **Required Columns:**
                - `FULL_NAME` — Teacher's full name
                - `MOBILE_NUMBER` — Contact number
                - `UDISE_CODE` — School UDISE code (Column F)
                - `IS_PROVISIONAL` — Verification status

                **Note:** Upload the file directly from ticklinks without modifications.
                """)

            user_file = st.file_uploader(
                "Upload User List",
                type=['xlsx', 'xls'],
                key='user',
                help="Upload the user list extracted from ticklinks"
            )

            st.download_button(
                label="⬇ Download Sample User List Format",
                data=create_sample_user(),
                file_name="Sample_User_File.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    # ══════════════════════════════════════════════════════════════════
    # RESULTS SECTION (after processing)
    # ══════════════════════════════════════════════════════════════════

    if st.session_state.processing_done:
        auto_verify  = st.session_state.auto_verify
        not_verified = st.session_state.not_verified
        not_registered = st.session_state.not_registered

        # Completion banner
        st.markdown("""
            <div class='completion-banner'>
                <div class='completion-title'>✅ Verification Complete!</div>
                <div class='completion-sub'>Your results are ready for download below</div>
            </div>
        """, unsafe_allow_html=True)

        # ── Summary Metrics ──
        st.markdown("<div class='section-heading'>Verification Summary</div>", unsafe_allow_html=True)

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("📝 Total Processed", st.session_state.prov_count, help="Total provisional rows processed")
        with mc2:
            st.metric("✅ Auto Verified", len(auto_verify),
                      delta=f"{(len(auto_verify)/max(st.session_state.prov_count,1)*100):.1f}%",
                      help="Records with match score ≥ 70")
        with mc3:
            st.metric("⚠️ Manual Review", len(not_verified),
                      delta=f"{(len(not_verified)/max(st.session_state.prov_count,1)*100):.1f}%",
                      delta_color="inverse",
                      help="Records with match score < 70")
        with mc4:
            st.metric("❌ Not Registered", len(not_registered), help="Master records not in user list")

        # ── Download Section ──
        st.markdown("<div class='section-heading'>Download Results</div>", unsafe_allow_html=True)

        dc1, dc2, dc3 = st.columns(3)

        with dc1:
            st.markdown("""
                <div class='result-card green'>
                    <div class='result-card-title'>✅ Auto Verified</div>
                    <div class='result-card-desc'>High confidence matches ready for automatic verification</div>
                </div>
            """, unsafe_allow_html=True)
            if len(auto_verify) > 0:
                st.download_button(
                    label=f"⬇ Download ({len(auto_verify)} records)",
                    data=create_excel_download(auto_verify, "Auto_Verify"),
                    file_name="Auto_Verify_This.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_auto",
                    use_container_width=True
                )
            else:
                st.markdown("<p style='color:#2d7a52;font-size:0.82rem;'>No auto-verified records</p>", unsafe_allow_html=True)

        with dc2:
            st.markdown("""
                <div class='result-card amber'>
                    <div class='result-card-title'>⚠️ Manual Review</div>
                    <div class='result-card-desc'>Records requiring manual verification by admin</div>
                </div>
            """, unsafe_allow_html=True)
            if len(not_verified) > 0:
                st.download_button(
                    label=f"⬇ Download ({len(not_verified)} records)",
                    data=create_excel_download(not_verified, "Not_Verified"),
                    file_name="Manual_Review_Required.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_notv",
                    use_container_width=True
                )
            else:
                st.markdown("<p style='color:#7a6220;font-size:0.82rem;'>No records need review</p>", unsafe_allow_html=True)

        with dc3:
            st.markdown("""
                <div class='result-card red'>
                    <div class='result-card-title'>❌ Not Registered</div>
                    <div class='result-card-desc'>Teachers in master list but absent from user database</div>
                </div>
            """, unsafe_allow_html=True)
            if len(not_registered) > 0:
                st.download_button(
                    label=f"⬇ Download ({len(not_registered)} records)",
                    data=create_excel_download(not_registered, "Not_Registered"),
                    file_name="Not_Registered.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_notr",
                    use_container_width=True
                )
            else:
                st.markdown("<p style='color:#7a3030;font-size:0.82rem;'>All records registered</p>", unsafe_allow_html=True)

        # ── Preview Tabs ──
        st.markdown("<div class='section-heading'>Data Preview</div>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs([
            f"✅ Auto Verified ({len(auto_verify)})",
            f"⚠️ Manual Review ({len(not_verified)})",
            f"❌ Not Registered ({len(not_registered)})"
        ])

        with tab1:
            if len(auto_verify) > 0:
                st.dataframe(auto_verify.head(50), use_container_width=True, height=380)
                if len(auto_verify) > 50:
                    st.markdown(f"<p style='color:#4a5578;font-size:0.8rem;'>Showing first 50 of {len(auto_verify)} records.</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color:#4a5578;font-size:0.85rem;'>No auto-verified records to display.</p>", unsafe_allow_html=True)

        with tab2:
            if len(not_verified) > 0:
                st.dataframe(not_verified.head(50), use_container_width=True, height=380)
                if len(not_verified) > 50:
                    st.markdown(f"<p style='color:#4a5578;font-size:0.8rem;'>Showing first 50 of {len(not_verified)} records.</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color:#4a5578;font-size:0.85rem;'>No records requiring manual review.</p>", unsafe_allow_html=True)

        with tab3:
            if len(not_registered) > 0:
                st.dataframe(not_registered.head(50), use_container_width=True, height=380)
                if len(not_registered) > 50:
                    st.markdown(f"<p style='color:#4a5578;font-size:0.8rem;'>Showing first 50 of {len(not_registered)} records.</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color:#4a5578;font-size:0.85rem;'>All teachers are registered.</p>", unsafe_allow_html=True)

        return  # Stop here — don't show upload UI again

    # ══════════════════════════════════════════════════════════════════
    # PROCESSING TRIGGER (files loaded but not yet run)
    # ══════════════════════════════════════════════════════════════════

    if master_file and user_file:
        try:
            master_df = pd.read_excel(master_file)
            user_df   = pd.read_excel(user_file)

            # Validation
            master_valid, master_msg = validate_master_file(master_df)
            user_valid,   user_msg   = validate_user_file(user_df)

            if not master_valid:
                st.markdown(f"""
                    <div style='background:#1f0a0a;border:1px solid #4a1a1a;border-radius:10px;
                                padding:1rem 1.4rem;margin-top:1rem;'>
                        <p style='color:#f87171;font-size:0.88rem;margin:0;'>❌ Master File Error: {master_msg}</p>
                    </div>
                """, unsafe_allow_html=True)
                return

            if not user_valid:
                st.markdown(f"""
                    <div style='background:#1f0a0a;border:1px solid #4a1a1a;border-radius:10px;
                                padding:1rem 1.4rem;margin-top:1rem;'>
                        <p style='color:#f87171;font-size:0.88rem;margin:0;'>❌ User File Error: {user_msg}</p>
                    </div>
                """, unsafe_allow_html=True)
                return

            # Success banner
            st.markdown(f"""
                <div class='validation-banner'>
                    <span class='validation-banner-icon'>✅</span>
                    <div>
                        <p class='validation-banner-text'>File structure validated and loaded successfully</p>
                        <p class='validation-banner-sub'>Master: {len(master_df)} rows &nbsp;|&nbsp; User List: {len(user_df)} rows</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Provisional count info
            if 'IS_PROVISIONAL' in user_df.columns:
                prov_count = (user_df['IS_PROVISIONAL'].astype(str).str.strip().str.upper() == 'TRUE').sum()
                st.markdown(f"""
                    <div class='info-banner'>
                        <p class='info-banner-text'>
                            <strong>ℹ️ Processing Information:</strong>&nbsp;
                            System will process <strong>{prov_count}</strong> provisional records (out of {len(user_df)} total)
                        </p>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

            # CTA button
            col_b1, col_b2, col_b3 = st.columns([2, 1.5, 2])
            with col_b2:
                start_clicked = st.button("🚀 Start Verification Process", use_container_width=True)

            if start_clicked:
                # Progress card
                st.markdown("""
                    <div class='progress-card'>
                        <div class='progress-label'>⚡ Verification in Progress</div>
                    </div>
                """, unsafe_allow_html=True)

                progress_bar  = st.progress(0)
                progress_text = st.empty()

                progress_text.markdown(
                    "<p style='color:#6b7a9e;font-size:0.85rem;text-align:center;'>Initializing matching engine...</p>",
                    unsafe_allow_html=True
                )

                # Step 1
                prov_results = process_user_to_master(
                    user_df, master_df, progress_bar, progress_text, 0, 50
                )

                # Step 2
                master_results = process_master_to_user(
                    master_df, user_df, progress_bar, progress_text, 50, 100
                )

                progress_bar.progress(100)
                progress_text.markdown(
                    "<p style='color:#4ade80;font-size:0.9rem;font-weight:600;text-align:center;'>✅ 100% completed — 0 sec remaining</p>",
                    unsafe_allow_html=True
                )
                time.sleep(0.8)

                # Categorise
                auto_verify    = prov_results[prov_results['Score'] >= 70] if len(prov_results) > 0 else pd.DataFrame()
                not_verified   = prov_results[prov_results['Score'] < 70]  if len(prov_results) > 0 else pd.DataFrame()
                not_registered = master_results[master_results['Score'] < 70]

                # Persist
                st.session_state.processing_done = True
                st.session_state.auto_verify      = auto_verify
                st.session_state.not_verified     = not_verified
                st.session_state.not_registered   = not_registered
                st.session_state.prov_count       = len(prov_results)
                st.session_state.master_count     = len(master_df)

                st.rerun()

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            with st.expander("View Error Details"):
                st.exception(e)

    else:
        # ── Nothing uploaded yet ──
        st.markdown("""
            <div class='getting-started'>
                <div class='getting-started-title'>👆 Getting Started</div>
                <div class='getting-started-text'>
                    Please upload both <strong>Master File</strong> and <strong>User List</strong>
                    above to begin the verification process
                </div>
            </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
