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
        i_token_count = len([t for t in normalize(input_name).split() if len(t) > 1])

        if perfect_count >= 2:
            # >= 2 tokens matching perfectly → addon 100, final 130 → capped 100
            addon = 100; rule = "≥2 tokens perfectly"

        elif masked_ok and perfect_count >= 1:
            # Phone visible digits + >=1 token perfectly → addon 100, final 130 → capped 100
            addon = 100; rule = "Phone visible digits + ≥1 token perfectly"

        elif masked_ok and fuzzy_count >= 2:
            # Phone visible digits + >=2 tokens fuzzy → addon 60, final 90
            addon = 60; rule = "Phone visible digits + ≥2 tokens fuzzy"

        elif masked_ok and fuzzy_count >= 1:
            # Phone visible digits + >=1 token fuzzy → addon 50, final 80
            addon = 50; rule = "Phone visible digits + ≥1 token fuzzy"

        elif perfect_count >= 1 and fuzzy_count >= 1:
            # >=1 token perfectly + >=1 token fuzzy → addon 50, final 80
            addon = 50; rule = "≥1 token perfectly + ≥1 token fuzzy"

        elif fuzzy_count >= 2:
            # >=2 tokens fuzzy → addon 40, final 70
            addon = 40; rule = "≥2 tokens fuzzy"

        elif perfect_count >= 1 and i_token_count == 1:
            # >=1 token perfectly, BUT user input has only 1 token → addon 40, final 70
            addon = 40; rule = "≥1 token perfectly (single token input)"

        elif perfect_count >= 1 and i_token_count > 1:
            # >=1 token perfectly, user input has >1 tokens → weaker signal → addon 20, final 50
            addon = 20; rule = "≥1 token perfectly (multi-token input)"

        elif fuzzy_count >= 1:
            # >=1 token fuzzy only → addon 20, final 50
            addon = 20; rule = "≥1 token fuzzy"

        else:
            addon = 0; rule = "No name match"
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

def process_user_to_master(user_df, master_df, progress_bar=None, status_text=None):
    """
    Match User file against Master file.
    Only processes rows where IS_PROVISIONAL == True.
    """
    master_names = master_df['TEACHER_NAME'].values
    master_phones = master_df['MOBILE_NO'].values
    master_udises = master_df['UDISE'].values

    # Filter to only IS_PROVISIONAL == True rows
    prov_col = None
    for col in user_df.columns:
        if col.upper() == 'IS_PROVISIONAL':
            prov_col = col
            break

    if prov_col:
        provisional_df = user_df[user_df[prov_col].astype(str).str.strip().str.upper() == 'TRUE'].copy()
    else:
        provisional_df = user_df.copy()

    results = []
    total = len(provisional_df)
    start_time = time.time()

    for i, (idx, row) in enumerate(provisional_df.iterrows()):
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

        if progress_bar is not None and total > 0:
            pct = (i + 1) / total
            elapsed = time.time() - start_time
            if pct > 0:
                eta = (elapsed / pct) * (1 - pct)
                eta_str = f"{int(eta)}s remaining" if eta < 60 else f"{int(eta/60)}m {int(eta%60)}s remaining"
            else:
                eta_str = "calculating..."
            progress_bar.progress(pct)
            if status_text:
                status_text.markdown(
                    f"<div style='font-size:13px;color:#aaa;'>Step 1/2 — Row {i+1}/{total} &nbsp;|&nbsp; ⏱ {eta_str}</div>",
                    unsafe_allow_html=True
                )

    if not results:
        # Return empty df with right columns
        result_df = provisional_df.copy()
        for key in ['Score', 'Matched_Name', 'Matched_Phone', 'Matched_UDISE', 'Rule', 'Details']:
            result_df[key] = []
        return result_df

    result_df = provisional_df.copy()
    for key in results[0].keys():
        result_df[key] = [r[key] for r in results]

    return result_df


def process_master_to_user(master_df, user_df, progress_bar=None, status_text=None):
    """Match Master file against User file (reverse direction)"""
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

        if progress_bar is not None and total > 0:
            pct = (i + 1) / total
            elapsed = time.time() - start_time
            if pct > 0:
                eta = (elapsed / pct) * (1 - pct)
                eta_str = f"{int(eta)}s remaining" if eta < 60 else f"{int(eta/60)}m {int(eta%60)}s remaining"
            else:
                eta_str = "calculating..."
            progress_bar.progress(pct)
            if status_text:
                status_text.markdown(
                    f"<div style='font-size:13px;color:#aaa;'>Step 2/2 — Row {i+1}/{total} &nbsp;|&nbsp; ⏱ {eta_str}</div>",
                    unsafe_allow_html=True
                )

    result_df = master_df.copy()
    for key in results[0].keys():
        result_df[key] = [r[key] for r in results]

    return result_df


def create_excel_download(df, sheet_name="Sheet1"):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output.read()


def create_sample_master():
    data = {
        'UDISE': ['27150100101', '27150100102', '27150100103'],
        'TEACHER_NAME': ['RAJESH KUMAR SHARMA', 'PRIYA MEHTA', 'SURESH PATIL'],
        'MOBILE_NO': ['XXXXXX7890', '9876XXXX12', '9123456789']
    }
    return create_excel_download(pd.DataFrame(data), "Master_Sample")


def create_sample_user():
    data = {
        'COMMUNITY_USER_ID': ['U001', 'U002', 'U003'],
        'FULL_NAME': ['Rajesh Kumar Sharma', 'Priya Mehta', 'New Teacher'],
        'MOBILE_NUMBER': ['9999987890', '9876543212', '8888888888'],
        'EMAIL': ['r@example.com', 'p@example.com', 'n@example.com'],
        'SCHOOL_NAME': ['ABC School', 'DEF School', 'GHI School'],
        'UDISE_CODE': ['27150100101', '27150100102', '27150100999'],
        'COMMUNITY_NAME': ['Community A', 'Community B', 'Community C'],
        'CIRCLE_NAME': ['Circle 1', 'Circle 1', 'Circle 2'],
        'BLOCK_NAME': ['Block A', 'Block A', 'Block B'],
        'DISTRICT_NAME': ['Pune', 'Pune', 'Mumbai'],
        'IS_PROVISIONAL': ['True', 'True', 'False']
    }
    return create_excel_download(pd.DataFrame(data), "User_Sample")


# ══════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Teacher Verification System",
        page_icon="📋",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
    <style>
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #e74c3c, #f39c12);
    }
    .reset-note {
        font-size: 12px;
        color: #888;
        margin-top: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("📋 Teacher Verification System")
    st.markdown("---")

    # ── Session state init ──────────────────────────────────────────
    if 'results_ready' not in st.session_state:
        st.session_state.results_ready = False
    if 'auto_verify_bytes' not in st.session_state:
        st.session_state.auto_verify_bytes = None
    if 'not_verified_bytes' not in st.session_state:
        st.session_state.not_verified_bytes = None
    if 'not_registered_bytes' not in st.session_state:
        st.session_state.not_registered_bytes = None
    if 'auto_verify_df' not in st.session_state:
        st.session_state.auto_verify_df = None
    if 'not_verified_df' not in st.session_state:
        st.session_state.not_verified_df = None
    if 'not_registered_df' not in st.session_state:
        st.session_state.not_registered_df = None
    if 'summary' not in st.session_state:
        st.session_state.summary = None

    # ── Reset button (top right) ─────────────────────────────────────
    _, reset_col = st.columns([6, 1])
    with reset_col:
        if st.button("🔄 Reset", type="secondary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # ── File upload section ──────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📁 Upload Master File")
        st.info("Expected columns: UDISE, TEACHER_NAME, MOBILE_NO")
        master_file = st.file_uploader(
            "Upload Master File (Excel)",
            type=['xlsx', 'xls'],
            key='master'
        )
        st.download_button(
            label="📄 Download Sample Master Format",
            data=create_sample_master(),
            file_name="Sample_Master_Format.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="sample_master"
        )

    with col2:
        st.subheader("📁 Upload User File")
        st.info("Must contain: FULL_NAME, MOBILE_NUMBER, UDISE_CODE (column F), IS_PROVISIONAL (column K)")
        user_file = st.file_uploader(
            "Upload User File (Excel)",
            type=['xlsx', 'xls'],
            key='user'
        )
        st.download_button(
            label="📄 Download Sample User Format",
            data=create_sample_user(),
            file_name="Sample_User_Format.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="sample_user"
        )

    # ── If results already computed, show them immediately ───────────
    if st.session_state.results_ready:
        _render_results()
        return

    # ── Process files ────────────────────────────────────────────────
    if master_file and user_file:
        try:
            master_df = pd.read_excel(master_file)
            user_df = pd.read_excel(user_file)

            st.success(f"✅ Files loaded: Master ({len(master_df)} rows), User ({len(user_df)} rows)")

            master_valid, master_msg = validate_master_file(master_df)
            user_valid, user_msg = validate_user_file(user_df)

            if not master_valid:
                st.error(f"❌ Master File Error: {master_msg}")
                return
            if not user_valid:
                st.error(f"❌ User File Error: {user_msg}")
                return

            # Count provisional rows
            prov_col = next((c for c in user_df.columns if c.upper() == 'IS_PROVISIONAL'), None)
            if prov_col:
                prov_count = (user_df[prov_col].astype(str).str.strip().str.upper() == 'TRUE').sum()
                st.success(f"✅ File structures validated successfully — {prov_count} IS_PROVISIONAL=True rows found in User file (Step 1 will only process these)")
            else:
                st.success("✅ File structures validated successfully")
                st.warning("⚠️ IS_PROVISIONAL column not found in User file — processing all rows in Step 1")

            if st.button("🚀 Start Processing", type="primary", use_container_width=True):
                st.markdown("---")

                # ── Step 1 ──────────────────────────────────────────
                st.markdown("**Step 1/2: Matching User file (IS_PROVISIONAL=True) → Master file...**")
                pb1 = st.progress(0)
                st1_text = st.empty()

                user_results = process_user_to_master(user_df, master_df, pb1, st1_text)
                pb1.progress(1.0)
                st1_text.markdown("<div style='font-size:13px;color:#4CAF50;'>✅ Step 1 complete</div>", unsafe_allow_html=True)

                # ── Step 2 ──────────────────────────────────────────
                st.markdown("**Step 2/2: Matching Master file → User file...**")
                pb2 = st.progress(0)
                st2_text = st.empty()

                master_results = process_master_to_user(master_df, user_df, pb2, st2_text)
                pb2.progress(1.0)
                st2_text.markdown("<div style='font-size:13px;color:#4CAF50;'>✅ Step 2 complete</div>", unsafe_allow_html=True)

                # ── Split & store ────────────────────────────────────
                auto_verify = user_results[user_results['Score'] >= 70]
                not_verified = user_results[user_results['Score'] < 70]
                not_registered = master_results[master_results['Score'] < 70]

                st.session_state.auto_verify_bytes = create_excel_download(auto_verify, "Auto_Verify")
                st.session_state.not_verified_bytes = create_excel_download(not_verified, "Not_Verified")
                st.session_state.not_registered_bytes = create_excel_download(not_registered, "Not_Registered")
                st.session_state.auto_verify_df = auto_verify
                st.session_state.not_verified_df = not_verified
                st.session_state.not_registered_df = not_registered
                st.session_state.summary = (len(auto_verify), len(not_verified), len(not_registered))
                st.session_state.results_ready = True

                st.rerun()

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

    else:
        st.info("👆 Please upload both Master and User files to begin")


def _render_results():
    """Render results section from session state (persists across downloads)."""
    n_auto, n_not_v, n_not_r = st.session_state.summary

    st.success("✅ Processing completed!")
    st.markdown("---")
    st.subheader("📊 Summary")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Auto Verify (≥70)", n_auto)
    with col2:
        st.metric("Not Verified (<70)", n_not_v)
    with col3:
        st.metric("Not Registered (<70)", n_not_r)

    st.markdown("---")
    st.subheader("📥 Download Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        if n_auto > 0:
            st.download_button(
                label="⬇️ Download Auto Verify",
                data=st.session_state.auto_verify_bytes,
                file_name="Auto_verify_This.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_auto"
            )
        else:
            st.warning("No records with score ≥70")

    with col2:
        if n_not_v > 0:
            st.download_button(
                label="⬇️ Download Not Verified",
                data=st.session_state.not_verified_bytes,
                file_name="Not_Verified.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_not_v"
            )
        else:
            st.warning("No records with score <70")

    with col3:
        if n_not_r > 0:
            st.download_button(
                label="⬇️ Download Not Registered",
                data=st.session_state.not_registered_bytes,
                file_name="Not_Registered.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_not_r"
            )
        else:
            st.warning("No master records with score <70")

    st.markdown("---")
    st.subheader("👁️ Preview Results")

    tab1, tab2, tab3 = st.tabs(["Auto Verify", "Not Verified", "Not Registered"])

    with tab1:
        if n_auto > 0:
            st.dataframe(st.session_state.auto_verify_df.head(50), use_container_width=True)
        else:
            st.info("No records")

    with tab2:
        if n_not_v > 0:
            st.dataframe(st.session_state.not_verified_df.head(50), use_container_width=True)
        else:
            st.info("No records")

    with tab3:
        if n_not_r > 0:
            st.dataframe(st.session_state.not_registered_df.head(50), use_container_width=True)
        else:
            st.info("No records")


if __name__ == "__main__":
    main()