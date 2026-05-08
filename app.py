import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import re
import time
import gc

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


# ══════════════════════════════════════════════════════════════════════
# FIX FOR SCIENTIFIC NOTATION UDISE
# ══════════════════════════════════════════════════════════════════════

def clean_udise(value):
    try:
        return str(int(float(value))).strip()
    except:
        return str(value).strip()

def build_udise_index(master_df):
    udise_index = {}

    for idx, row in master_df.iterrows():
        udise = clean_udise(row['UDISE'])

        if udise not in udise_index:
            udise_index[udise] = []

        udise_index[udise].append(idx)

    return udise_index


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
                               master_udises, input_udise,candidate_indices=None):
    if not input_name or pd.isna(input_name):
        return 0, "", "", "", "No input name", ""

    # FIXED HERE
    i_udise = clean_udise(input_udise)

    i_phone_r = str(input_phone or "").strip()
    i_clean = re.sub(r'\D', '', i_phone_r)

    best_show = 0
    best_name = ""
    best_phone = ""
    best_udise = ""
    best_rule = ""
    best_details = ""

    if candidate_indices is None:
        candidate_indices = range(len(master_names))

    for idx in candidate_indices:
        m_name = str(master_names[idx] or "").strip()
        m_phone_r = str(master_phones[idx] or "").strip()

        # FIXED HERE
        m_udise = clean_udise(master_udises[idx])

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
                return (
                    100,
                    m_name,
                    m_phone_r,
                    m_udise,
                    "Stage3 bypass: unmasked phone exact",
                    "S1:30 S2:bypassed S3:100"
                )

        if not udise_match:
            continue

        stage1 = 30
        perfect_count, fuzzy_count = get_name_token_results(input_name, m_name)

        masked_ok = (
            m_is_masked and
            not m_is_ignored and
            len(i_clean) >= 10 and
            masked_visible_digits_match(i_clean, m_clean)
        )

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
    """
    Match ONLY Is_Provisional == True rows from User file against Master file.
    Returns (provisional_results_df, full_user_df_with_scores)
    """
    master_names = master_df['TEACHER_NAME'].values
    master_phones = master_df['MOBILE_NO'].values
    master_udises = master_df['UDISE'].values
    master_udise_index = build_udise_index(master_df)

    # Filter: only Is_Provisional == True rows
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

        candidate_indices = master_udise_index.get(clean_udise(input_udise), [])

        score, matched_name, matched_phone, matched_udise, rule, details = \
            compare_name_phone_single(
                master_names,
                input_name,
                master_phones,
                input_phone,
                master_udises,
                input_udise,
                candidate_indices
            )

        results.append({
            'Score': score,
            'Matched_Name': matched_name,
            'Matched_Phone': matched_phone,
            'Matched_UDISE': matched_udise,
            'Rule': rule,
            'Details': details
        })

        # Update progress
        if total > 0 and i % 1000 == 0:
            frac = (i + 1) / total
            pct = start_pct + frac * (end_pct - start_pct)
            progress_bar.progress(int(pct))
            elapsed = time.time() - start_time
            remaining = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else 0
            progress_text.text(f"Step 1/2: Matching User → Master  |  {i+1}/{total} rows  |  ⏱ ~{int(remaining)}s remaining")

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

        if total > 0 and i % 1000 == 0:
            frac = (i + 1) / total
            pct = start_pct + frac * (end_pct - start_pct)
            progress_bar.progress(int(pct))
            elapsed = time.time() - start_time
            remaining = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else 0
            progress_text.text(f"Step 2/2: Matching Master → User  |  {i+1}/{total} rows  |  ⏱ ~{int(remaining)}s remaining")

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
# STREAMLIT UI - IMPROVED DESIGN
# ══════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Teacher Verification System",
        page_icon="📋",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Custom CSS for aesthetic design
    st.markdown("""
        <style>
        /* Global Styles */
        .stApp {
            background: #0f172a;
        }
        
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            background: #1e293b;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            margin-top: 2rem;
        }
        
        /* Header */
        .main-header {
            text-align: center;
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            color: #ffffff !important;
        }
        
        /* Tool Title */
        .tool-title {
            text-align: center;
            font-size: 1.6rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            margin-bottom: 0.4rem;
            color: #a78bfa !important;
            text-transform: uppercase;
        }
        
        .subtitle {
            text-align: center;
            color: #94a3b8 !important;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }
        
        /* Upload Cards */
        .upload-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.8rem;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
            margin-bottom: 1.5rem;
            transition: transform 0.3s ease;
            height: 160px;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        
        .upload-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 30px rgba(102, 126, 234, 0.5);
        }
        
        .upload-card-alt {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 1.8rem;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(245, 87, 108, 0.4);
            margin-bottom: 1.5rem;
            transition: transform 0.3s ease;
            height: 160px;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        
        .upload-card-alt:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 30px rgba(245, 87, 108, 0.5);
        }
        
        .card-title {
            color: white;
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .card-note {
            color: rgba(255, 255, 255, 0.85);
            font-size: 0.92rem;
            font-weight: 500;
            line-height: 1.5;
            margin-top: -0.4rem;
        }
        
        /* Metrics */
        .metric-container {
            background: linear-gradient(135deg, #334155 0%, #475569 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 5px solid #667eea;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            margin-bottom: 1rem;
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 800;
            color: #e0e7ff;
            margin-bottom: 0.3rem;
        }
        
        .metric-label {
            font-size: 0.95rem;
            color: #cbd5e1;
            font-weight: 600;
        }
        
        /* Buttons */
        .stButton > button {
            border-radius: 10px;
            font-weight: 600;
            padding: 0.6rem 1.2rem;
            transition: all 0.3s ease;
            border: none;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        }
        
        .stDownloadButton > button {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            border-radius: 10px;
            font-weight: 600;
            border: none;
        }
        
        .stDownloadButton > button:hover {
            background: linear-gradient(135deg, #059669 0%, #047857 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
        }
        
        /* File Uploader */
        .stFileUploader {
            background: #334155;
            border-radius: 12px;
            padding: 1rem;
        }
        
        .stFileUploader label {
            color: #e2e8f0 !important;
        }
        
        .stFileUploader [data-testid="stFileUploaderDropzone"] {
            background-color: #1e293b;
            border: 2px dashed #475569;
        }
        
        .stFileUploader [data-testid="stFileUploaderDropzone"]:hover {
            border-color: #667eea;
            background-color: #334155;
        }
        
        .stFileUploader small {
            color: #94a3b8 !important;
        }
        
        /* Divider */
        hr {
            margin: 2.5rem 0;
            border: none;
            height: 2px;
            background: linear-gradient(90deg, transparent, #667eea, transparent);
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 3rem;
            border-radius: 10px 10px 0 0;
            padding: 0 2rem;
            font-weight: 600;
        }
        
        /* Progress */
        .stProgress > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Info boxes */
        .info-box {
            background: #1e3a5f;
            border-left: 4px solid #3b82f6;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            color: #bfdbfe;
        }
        
        /* Text colors */
        p, span, div {
            color: #e2e8f0;
        }
        
        /* Dataframe styling */
        .stDataFrame {
            background: #334155;
        }
        
        /* Tab styling for dark theme */
        .stTabs [data-baseweb="tab-list"] {
            background: #1e293b;
        }
        
        .stTabs [data-baseweb="tab"] {
            color: #94a3b8;
        }
        
        .stTabs [aria-selected="true"] {
            color: #667eea;
        }
        </style>
    """, unsafe_allow_html=True)

    # ── Session State Init ──
    for key in ['processing_done', 'auto_verify', 'not_verified', 'not_registered',
                'prov_count', 'master_count']:
        if key not in st.session_state:
            st.session_state[key] = None

    # ── Header ──
    st.markdown('<h1 class="main-header">📋 Teacher Verification System</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="tool-title">Smart Auto Verification Tool</h2>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload files, match records, and download results seamlessly</p>', unsafe_allow_html=True)
    
    # ── Reset Button (Top Right) ──
    if st.session_state.processing_done:
        col_space, col_reset = st.columns([8, 1])
        with col_reset:
            if st.button("🔄 Reset", type="secondary", use_container_width=True):
                for key in ['processing_done', 'auto_verify', 'not_verified', 'not_registered',
                            'prov_count', 'master_count']:
                    st.session_state[key] = None
                st.rerun()

    # ── File Upload Section ──
    if not st.session_state.processing_done:
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown('''
                <div class="upload-card">
                    <div class="card-title">📂 Master File</div>
                    <div class="card-note">Check the format by downloading the sample from below and upload Teacher\'s data in that format strictly.</div>
                </div>
            ''', unsafe_allow_html=True)
            
            master_file = st.file_uploader(
                "Upload Master Excel File",
                type=['xlsx', 'xls'],
                key='master',
                label_visibility="collapsed"
            )
            
            st.download_button(
                label="📥 Download Sample",
                data=create_sample_master(),
                file_name="Sample_Master.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        with col2:
            st.markdown('''
                <div class="upload-card-alt">
                    <div class="card-title">📂 User File</div>
                    <div class="card-note">Extract the User List from Ticklinks for that district and upload it here directly.</div>
                </div>
            ''', unsafe_allow_html=True)
            
            user_file = st.file_uploader(
                "Upload User Excel File",
                type=['xlsx', 'xls'],
                key='user',
                label_visibility="collapsed"
            )
            
            st.download_button(
                label="📥 Download Sample",
                data=create_sample_user(),
                file_name="Sample_User.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    # ── If processing already done, show results directly ──
    if st.session_state.processing_done:
        auto_verify = st.session_state.auto_verify
        not_verified = st.session_state.not_verified
        not_registered = st.session_state.not_registered

        st.markdown("---")
        st.markdown("## 📊 Processing Complete")
        st.markdown('<div class="info-box">✅ All records have been matched and categorized successfully!</div>', unsafe_allow_html=True)

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'''
                <div class="metric-container">
                    <div class="metric-value">{st.session_state.prov_count}</div>
                    <div class="metric-label">Total Processed</div>
                </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
                <div class="metric-container" style="border-left-color: #10b981;">
                    <div class="metric-value" style="color: #6ee7b7;">{len(auto_verify)}</div>
                    <div class="metric-label">Auto Verified</div>
                </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'''
                <div class="metric-container" style="border-left-color: #f59e0b;">
                    <div class="metric-value" style="color: #fbbf24;">{len(not_verified)}</div>
                    <div class="metric-label">Not Verified</div>
                </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            st.markdown(f'''
                <div class="metric-container" style="border-left-color: #ef4444;">
                    <div class="metric-value" style="color: #fca5a5;">{len(not_registered)}</div>
                    <div class="metric-label">Not Registered</div>
                </div>
            ''', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("## 📥 Download Results")

        # Download buttons in a row
        dc1, dc2, dc3 = st.columns(3)
        
        with dc1:
            if len(auto_verify) > 0:
                st.download_button(
                    label="✅ Auto Verify",
                    data=create_excel_download(auto_verify, "Auto_Verify"),
                    file_name="Auto_Verify.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_auto",
                    use_container_width=True
                )
            else:
                st.info("No records")
                
        with dc2:
            if len(not_verified) > 0:
                st.download_button(
                    label="⚠️ Not Verified",
                    data=create_excel_download(not_verified, "Not_Verified"),
                    file_name="Not_Verified.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_notv",
                    use_container_width=True
                )
            else:
                st.info("No records")
                
        with dc3:
            if len(not_registered) > 0:
                st.download_button(
                    label="❌ Not Registered",
                    data=create_excel_download(not_registered, "Not_Registered"),
                    file_name="Not_Registered.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_notr",
                    use_container_width=True
                )
            else:
                st.info("No records")

        st.markdown("---")
        st.markdown("## 👁️ Preview Results")
        
        tab1, tab2, tab3 = st.tabs(["✅ Auto Verify", "⚠️ Not Verified", "❌ Not Registered"])
        
        with tab1:
            if len(auto_verify) > 0:
                st.dataframe(auto_verify.head(50), use_container_width=True, height=400)
            else:
                st.info("No records to display")
                
        with tab2:
            if len(not_verified) > 0:
                st.dataframe(not_verified.head(50), use_container_width=True, height=400)
            else:
                st.info("No records to display")
                
        with tab3:
            if len(not_registered) > 0:
                st.dataframe(not_registered.head(50), use_container_width=True, height=400)
            else:
                st.info("No records to display")

        return  # Don't show processing button again

    # ── Only show processing UI if not done yet ──
    if master_file and user_file:
        try:
            master_df = pd.read_excel(master_file)
            user_df = pd.read_excel(user_file)

            st.markdown(f'<div class="info-box">✅ Files loaded successfully! Master: {len(master_df)} rows | User: {len(user_df)} rows</div>', unsafe_allow_html=True)

            master_valid, master_msg = validate_master_file(master_df)
            user_valid, user_msg = validate_user_file(user_df)

            if not master_valid:
                st.error(f"❌ Master File Error: {master_msg}")
                return
            if not user_valid:
                st.error(f"❌ User File Error: {user_msg}")
                return

            # Show provisional count
            if 'IS_PROVISIONAL' in user_df.columns:
                prov_count = (user_df['IS_PROVISIONAL'].astype(str).str.strip().str.upper() == 'TRUE').sum()
                st.markdown(f'<div class="info-box">ℹ️ Found <strong>{prov_count}</strong> provisional records to process</div>', unsafe_allow_html=True)

            # Centered process button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Start Processing", type="primary", use_container_width=True):
                    progress_bar = st.progress(0)
                    progress_text = st.empty()

                    progress_text.markdown("**Step 1/2:** Matching User → Master...")

                    # Step 1: User → Master (only Is_Provisional == True)
                    prov_results = process_user_to_master(
                        user_df, master_df, progress_bar, progress_text, 0, 50
                    )
                    gc.collect()

                    # Step 2: Master → User (all master rows)
                    progress_text.markdown("**Step 2/2:** Matching Master → User...")
                    master_results = process_master_to_user(
                        master_df, user_df, progress_bar, progress_text, 50, 100
                    )
                    gc.collect()

                    progress_bar.progress(100)
                    progress_text.markdown("✅ **Processing complete!**")

                    # Split results
                    auto_verify = prov_results[prov_results['Score'] >= 70] if len(prov_results) > 0 else pd.DataFrame()
                    not_verified = prov_results[prov_results['Score'] < 70] if len(prov_results) > 0 else pd.DataFrame()
                    not_registered = master_results[master_results['Score'] < 70]

                    # Store in session state
                    st.session_state.processing_done = True
                    st.session_state.auto_verify = auto_verify
                    st.session_state.not_verified = not_verified
                    st.session_state.not_registered = not_registered
                    st.session_state.prov_count = len(prov_results)
                    st.session_state.master_count = len(master_df)

                    st.balloons()
                    st.rerun()

        except Exception as e:
            st.error(f"❌ Error processing files: {str(e)}")
            with st.expander("🔍 View Error Details"):
                st.exception(e)

    else:
        st.markdown('<div class="info-box">👆 Please upload both Master and User files to begin processing</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
