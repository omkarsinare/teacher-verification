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
    """
    Match ONLY Is_Provisional == True rows from User file against Master file.
    Returns (provisional_results_df, full_user_df_with_scores)
    """
    master_names = master_df['TEACHER_NAME'].values
    master_phones = master_df['MOBILE_NO'].values
    master_udises = master_df['UDISE'].values

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

        # Update progress
        if total > 0:
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

        if total > 0:
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
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Teacher Verification System",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Custom CSS for modern aesthetic design
    st.markdown("""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles */
        * {
            font-family: 'Inter', sans-serif;
        }
        
        /* Main container */
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
        }
        
        /* Header styling */
        h1 {
            color: white !important;
            font-weight: 700 !important;
            font-size: 2.8rem !important;
            text-align: center;
            margin-bottom: 0.5rem !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        h2 {
            color: #667eea !important;
            font-weight: 600 !important;
            font-size: 1.5rem !important;
            margin-top: 1.5rem !important;
        }
        
        h3 {
            color: #4a5568 !important;
            font-weight: 600 !important;
            font-size: 1.2rem !important;
        }
        
        /* Card-like containers */
        .stContainer {
            background: white;
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 20px 60px rgba(0,0,0,0.15);
            margin-bottom: 1.5rem;
        }
        
        /* File uploader styling */
        .uploadedFile {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 12px;
            padding: 1rem;
            border: 2px dashed #667eea;
        }
        
        /* Button styling */
        .stButton button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            width: 100%;
        }
        
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }
        
        /* Download button styling */
        .stDownloadButton button {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(56, 239, 125, 0.3);
        }
        
        .stDownloadButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(56, 239, 125, 0.5);
        }
        
        /* Metric styling */
        [data-testid="stMetricValue"] {
            font-size: 2rem !important;
            font-weight: 700 !important;
            color: #667eea !important;
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 0.9rem !important;
            font-weight: 500 !important;
            color: #4a5568 !important;
        }
        
        /* Info boxes */
        .stAlert {
            border-radius: 12px;
            border-left: 4px solid #667eea;
            background: linear-gradient(135deg, #f0f4ff 0%, #e8ecff 100%);
        }
        
        /* Success boxes */
        .element-container:has(.stSuccess) {
            border-radius: 12px;
        }
        
        /* Dataframe styling */
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: transparent;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 12px;
            padding: 12px 24px;
            background: #f7fafc;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        /* Progress bar */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }
        
        /* Remove Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Column padding */
        [data-testid="column"] {
            padding: 0 1rem;
        }
        
        /* Subtitle styling */
        .subtitle {
            text-align: center;
            color: white;
            font-size: 1.1rem;
            font-weight: 400;
            margin-bottom: 2rem;
            opacity: 0.95;
        }
        
        /* Feature cards */
        .feature-card {
            background: white;
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            height: 100%;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.15);
        }
        
        /* Status badges */
        .status-badge {
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9rem;
        }
        
        .status-success {
            background: #d4edda;
            color: #155724;
        }
        
        .status-warning {
            background: #fff3cd;
            color: #856404;
        }
        
        .status-info {
            background: #d1ecf1;
            color: #0c5460;
        }
        </style>
    """, unsafe_allow_html=True)

    # ── Session State Init ──
    for key in ['processing_done', 'auto_verify', 'not_verified', 'not_registered',
                'prov_count', 'master_count']:
        if key not in st.session_state:
            st.session_state[key] = None

    # ── Hero Section ──
    st.markdown("<h1>🎓 Teacher Verification System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Intelligent matching system for teacher verification with advanced fuzzy matching algorithms</p>", unsafe_allow_html=True)

    # ── Reset Button (Only show after processing) ──
    if st.session_state.processing_done:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔄 Start New Verification", type="primary", use_container_width=True):
                for key in ['processing_done', 'auto_verify', 'not_verified', 'not_registered',
                            'prov_count', 'master_count']:
                    st.session_state[key] = None
                st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)

    # ── File Upload Section ──
    if not st.session_state.processing_done:
        # Create a container for better spacing
        with st.container():
            col1, col2 = st.columns(2, gap="large")

            with col1:
                st.markdown("""
                    <div class='feature-card'>
                        <h3>📊 Master File</h3>
                        <p style='color: #718096; margin-bottom: 1rem;'>Upload your master teacher database</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                with st.expander("ℹ️ View Expected Format", expanded=False):
                    st.markdown("""
                    **Required Columns (in order):**
                    1. `UDISE` - School UDISE code
                    2. `TEACHER_NAME` - Full name of teacher
                    3. `MOBILE_NO` - Mobile number (can be masked with X)
                    
                    **Example:**
                    - UDISE: 27110100101
                    - TEACHER_NAME: RAMESH KUMAR SHARMA
                    - MOBILE_NO: 98XXXXXX01
                    """)
                
                master_file = st.file_uploader(
                    "Choose Master Excel File",
                    type=['xlsx', 'xls'],
                    key='master',
                    help="Upload the master teacher database file"
                )
                
                st.download_button(
                    label="📥 Download Sample Master File",
                    data=create_sample_master(),
                    file_name="Sample_Master_File.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Download a sample file to understand the required format",
                    use_container_width=True
                )

            with col2:
                st.markdown("""
                    <div class='feature-card'>
                        <h3>👥 User List</h3>
                        <p style='color: #718096; margin-bottom: 1rem;'>Upload user list from ticklinks</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                with st.expander("ℹ️ View Expected Format", expanded=False):
                    st.markdown("""
                    **Required Columns:**
                    - `FULL_NAME` - Teacher's full name
                    - `MOBILE_NUMBER` - Contact number
                    - `UDISE_CODE` - School UDISE code (Column F)
                    - `IS_PROVISIONAL` - Verification status
                    
                    **Note:** Upload the file directly from ticklinks without any modifications.
                    """)
                
                user_file = st.file_uploader(
                    "Choose User Excel File",
                    type=['xlsx', 'xls'],
                    key='user',
                    help="Upload the user list extracted from ticklinks"
                )
                
                st.download_button(
                    label="📥 Download Sample User List",
                    data=create_sample_user(),
                    file_name="Sample_User_File.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Download a sample file to understand the required format",
                    use_container_width=True
                )

    # ── If processing already done, show results directly ──
    if st.session_state.processing_done:
        auto_verify = st.session_state.auto_verify
        not_verified = st.session_state.not_verified
        not_registered = st.session_state.not_registered

        st.markdown("""
            <div style='background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); 
                        padding: 1.5rem; border-radius: 12px; margin: 2rem 0;
                        border-left: 5px solid #28a745; text-align: center;'>
                <h3 style='color: #155724; margin: 0;'>✅ Verification Complete!</h3>
                <p style='color: #155724; margin: 0.5rem 0 0 0;'>Your results are ready for download</p>
            </div>
        """, unsafe_allow_html=True)

        # ── Summary Metrics ──
        st.markdown("<h2>📊 Verification Summary</h2>", unsafe_allow_html=True)
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric(
                label="📝 Total Processed",
                value=st.session_state.prov_count,
                help="Total provisional rows processed"
            )
        
        with metric_col2:
            st.metric(
                label="✅ Auto Verified",
                value=len(auto_verify),
                delta=f"{(len(auto_verify)/max(st.session_state.prov_count, 1)*100):.1f}%",
                help="Records with match score ≥ 70"
            )
        
        with metric_col3:
            st.metric(
                label="⚠️ Manual Review",
                value=len(not_verified),
                delta=f"{(len(not_verified)/max(st.session_state.prov_count, 1)*100):.1f}%",
                delta_color="inverse",
                help="Records with match score < 70"
            )
        
        with metric_col4:
            st.metric(
                label="❌ Not Registered",
                value=len(not_registered),
                help="Master records not found in user list"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Download Section ──
        st.markdown("<h2>📥 Download Results</h2>", unsafe_allow_html=True)
        
        download_col1, download_col2, download_col3 = st.columns(3)
        
        with download_col1:
            st.markdown("""
                <div style='background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); 
                            padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem;
                            border-left: 5px solid #28a745; height: 120px;'>
                    <h4 style='color: #155724; margin: 0 0 0.5rem 0;'>✅ Auto Verified</h4>
                    <p style='color: #155724; margin: 0; font-size: 0.9rem;'>
                        High confidence matches ready for automatic verification
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            if len(auto_verify) > 0:
                st.download_button(
                    label=f"⬇️ Download ({len(auto_verify)} records)",
                    data=create_excel_download(auto_verify, "Auto_Verify"),
                    file_name="Auto_Verify_This.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_auto",
                    use_container_width=True
                )
            else:
                st.info("No auto-verified records", icon="ℹ️")
        
        with download_col2:
            st.markdown("""
                <div style='background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); 
                            padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem;
                            border-left: 5px solid #ffc107; height: 120px;'>
                    <h4 style='color: #856404; margin: 0 0 0.5rem 0;'>⚠️ Manual Review</h4>
                    <p style='color: #856404; margin: 0; font-size: 0.9rem;'>
                        Records requiring manual verification
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            if len(not_verified) > 0:
                st.download_button(
                    label=f"⬇️ Download ({len(not_verified)} records)",
                    data=create_excel_download(not_verified, "Not_Verified"),
                    file_name="Manual_Review_Required.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_notv",
                    use_container_width=True
                )
            else:
                st.info("No records need review", icon="ℹ️")
        
        with download_col3:
            st.markdown("""
                <div style='background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); 
                            padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem;
                            border-left: 5px solid #dc3545; height: 120px;'>
                    <h4 style='color: #721c24; margin: 0 0 0.5rem 0;'>❌ Not Registered</h4>
                    <p style='color: #721c24; margin: 0; font-size: 0.9rem;'>
                        Teachers in master list but not in user database
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            if len(not_registered) > 0:
                st.download_button(
                    label=f"⬇️ Download ({len(not_registered)} records)",
                    data=create_excel_download(not_registered, "Not_Registered"),
                    file_name="Not_Registered.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_notr",
                    use_container_width=True
                )
            else:
                st.info("All records registered", icon="ℹ️")

        st.markdown("<br><br>", unsafe_allow_html=True)

        # ── Preview Section ──
        st.markdown("<h2>👁️ Data Preview</h2>", unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs([
            f"✅ Auto Verified ({len(auto_verify)})",
            f"⚠️ Manual Review ({len(not_verified)})",
            f"❌ Not Registered ({len(not_registered)})"
        ])
        
        with tab1:
            if len(auto_verify) > 0:
                st.dataframe(
                    auto_verify.head(50),
                    use_container_width=True,
                    height=400
                )
                if len(auto_verify) > 50:
                    st.info(f"Showing first 50 of {len(auto_verify)} records. Download the file to see all records.", icon="ℹ️")
            else:
                st.info("No auto-verified records to display", icon="ℹ️")
        
        with tab2:
            if len(not_verified) > 0:
                st.dataframe(
                    not_verified.head(50),
                    use_container_width=True,
                    height=400
                )
                if len(not_verified) > 50:
                    st.info(f"Showing first 50 of {len(not_verified)} records. Download the file to see all records.", icon="ℹ️")
            else:
                st.info("No records requiring manual review", icon="ℹ️")
        
        with tab3:
            if len(not_registered) > 0:
                st.dataframe(
                    not_registered.head(50),
                    use_container_width=True,
                    height=400
                )
                if len(not_registered) > 50:
                    st.info(f"Showing first 50 of {len(not_registered)} records. Download the file to see all records.", icon="ℹ️")
            else:
                st.info("All teachers are registered", icon="ℹ️")

        return  # Don't show processing button again

    # ── Only show processing UI if not done yet ──
    if master_file and user_file:
        try:
            master_df = pd.read_excel(master_file)
            user_df = pd.read_excel(user_file)

            st.markdown("""
                <div style='background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%); 
                            padding: 1.5rem; border-radius: 12px; margin: 2rem 0;
                            border-left: 5px solid #17a2b8; text-align: center;'>
                    <h4 style='color: #0c5460; margin: 0;'>📂 Files Loaded Successfully</h4>
                    <p style='color: #0c5460; margin: 0.5rem 0 0 0;'>
                        Master: {0} rows | User List: {1} rows
                    </p>
                </div>
            """.format(len(master_df), len(user_df)), unsafe_allow_html=True)

            master_valid, master_msg = validate_master_file(master_df)
            user_valid, user_msg = validate_user_file(user_df)

            if not master_valid:
                st.error(f"❌ Master File Error: {master_msg}", icon="🚫")
                return
            if not user_valid:
                st.error(f"❌ User File Error: {user_msg}", icon="🚫")
                return

            st.success("✅ File structures validated successfully", icon="✅")

            # Show provisional count
            if 'IS_PROVISIONAL' in user_df.columns:
                prov_count = (user_df['IS_PROVISIONAL'].astype(str).str.strip().str.upper() == 'TRUE').sum()
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #e8f4f8 0%, #d4e9f2 100%); 
                                padding: 1.5rem; border-radius: 12px; margin: 1.5rem 0;
                                border-left: 5px solid #3498db;'>
                        <p style='color: #1e5a7d; margin: 0; font-size: 1rem;'>
                            <strong>ℹ️ Processing Information:</strong><br>
                            System will process <strong>{prov_count}</strong> provisional records (out of {len(user_df)} total)
                        </p>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Start Verification Process", type="primary", use_container_width=True):
                    # Create a container for progress
                    progress_container = st.container()
                    
                    with progress_container:
                        st.markdown("""
                            <div style='background: white; padding: 2rem; border-radius: 16px; 
                                        box-shadow: 0 10px 30px rgba(0,0,0,0.1); margin: 2rem 0;'>
                                <h3 style='color: #667eea; text-align: center; margin-bottom: 1.5rem;'>
                                    🔄 Verification in Progress
                                </h3>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        progress_bar = st.progress(0)
                        progress_text = st.empty()

                        progress_text.markdown("""
                            <p style='text-align: center; color: #4a5568; font-size: 1.1rem;'>
                                <strong>Step 1/2:</strong> Matching User → Master (Provisional records only)...
                            </p>
                        """, unsafe_allow_html=True)

                        # Step 1: User → Master (only Is_Provisional == True)
                        prov_results = process_user_to_master(
                            user_df, master_df, progress_bar, progress_text, 0, 50
                        )

                        # Step 2: Master → User (all master rows)
                        progress_text.markdown("""
                            <p style='text-align: center; color: #4a5568; font-size: 1.1rem;'>
                                <strong>Step 2/2:</strong> Matching Master → User...
                            </p>
                        """, unsafe_allow_html=True)
                        
                        master_results = process_master_to_user(
                            master_df, user_df, progress_bar, progress_text, 50, 100
                        )

                        progress_bar.progress(100)
                        progress_text.markdown("""
                            <p style='text-align: center; color: #28a745; font-size: 1.2rem; font-weight: 600;'>
                                ✅ Processing Complete!
                            </p>
                        """, unsafe_allow_html=True)
                        
                        time.sleep(1)  # Brief pause to show completion

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

                        st.rerun()

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}", icon="🚫")
            with st.expander("View Error Details"):
                st.exception(e)

    elif not st.session_state.processing_done:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); 
                        padding: 3rem; border-radius: 16px; margin: 3rem 0;
                        text-align: center; border: 2px dashed #667eea;'>
                <h3 style='color: #1976d2; margin-bottom: 1rem;'>👆 Getting Started</h3>
                <p style='color: #1565c0; font-size: 1.1rem; margin: 0;'>
                    Please upload both <strong>Master File</strong> and <strong>User List</strong> to begin the verification process
                </p>
            </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
