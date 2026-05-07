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


# KEEP THE REST OF YOUR FILE EXACTLY SAME


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
        if total > 0 and i % 50 == 0:
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

        if total > 0 and i % 50 == 0:
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
        page_icon="📋",
        layout="wide"
    )

    # ── Session State Init ──
    for key in ['processing_done', 'auto_verify', 'not_verified', 'not_registered',
                'prov_count', 'master_count']:
        if key not in st.session_state:
            st.session_state[key] = None

    st.title("📋 Teacher Verification System")
    st.markdown("---")

    # ── Reset Button ──
    if st.session_state.processing_done:
        if st.button("🔄 Reset", type="secondary"):
            for key in ['processing_done', 'auto_verify', 'not_verified', 'not_registered',
                        'prov_count', 'master_count']:
                st.session_state[key] = None
            st.rerun()

    # ── File Upload Section ──
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
            file_name="Sample_Master_File.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download a sample Master file to understand the required format"
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
            file_name="Sample_User_File.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download a sample User file to understand the required format"
        )

    # ── If processing already done, show results directly ──
    if st.session_state.processing_done:
        auto_verify = st.session_state.auto_verify
        not_verified = st.session_state.not_verified
        not_registered = st.session_state.not_registered

        st.success("✅ Processing completed! Results are ready below.")
        st.markdown("---")
        st.subheader("📊 Summary")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Provisional Rows Processed", st.session_state.prov_count)
        with c2:
            st.metric("Auto Verify (≥70)", len(auto_verify))
        with c3:
            st.metric("Not Verified (<70)", len(not_verified))
        with c4:
            st.metric("Not Registered (<70)", len(not_registered))

        st.markdown("---")
        st.subheader("📥 Download Results")

        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            if len(auto_verify) > 0:
                st.download_button(
                    label="⬇️ Download Auto Verify",
                    data=create_excel_download(auto_verify, "Auto_Verify"),
                    file_name="Auto_verify_This.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_auto"
                )
            else:
                st.warning("No records with score ≥70")
        with dc2:
            if len(not_verified) > 0:
                st.download_button(
                    label="⬇️ Download Not Verified",
                    data=create_excel_download(not_verified, "Not_Verified"),
                    file_name="Not_Verified.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_notv"
                )
            else:
                st.warning("No records with score <70")
        with dc3:
            if len(not_registered) > 0:
                st.download_button(
                    label="⬇️ Download Not Registered",
                    data=create_excel_download(not_registered, "Not_Registered"),
                    file_name="Not_Registered.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_notr"
                )
            else:
                st.warning("No master records with score <70")

        st.markdown("---")
        st.subheader("👁️ Preview Results")
        tab1, tab2, tab3 = st.tabs(["Auto Verify", "Not Verified", "Not Registered"])
        with tab1:
            if len(auto_verify) > 0:
                st.dataframe(auto_verify.head(50), use_container_width=True)
            else:
                st.info("No records")
        with tab2:
            if len(not_verified) > 0:
                st.dataframe(not_verified.head(50), use_container_width=True)
            else:
                st.info("No records")
        with tab3:
            if len(not_registered) > 0:
                st.dataframe(not_registered.head(50), use_container_width=True)
            else:
                st.info("No records")

        return  # Don't show processing button again

    # ── Only show processing UI if not done yet ──
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

            st.success("✅ File structures validated successfully")

            # Show provisional count
            if 'IS_PROVISIONAL' in user_df.columns:
                prov_count = (user_df['IS_PROVISIONAL'].astype(str).str.strip().str.upper() == 'TRUE').sum()
                st.info(f"ℹ️ Step 1 will process **{prov_count}** rows where IS_PROVISIONAL = True (out of {len(user_df)} total)")

            if st.button("🚀 Start Processing", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                progress_text = st.empty()

                progress_text.text("Step 1/2: Matching User → Master (Is_Provisional=True only)...")

                # Step 1: User → Master (only Is_Provisional == True)
                prov_results = process_user_to_master(
                    user_df, master_df, progress_bar, progress_text, 0, 50
                )

                # Step 2: Master → User (all master rows)
                progress_text.text("Step 2/2: Matching Master → User...")
                master_results = process_master_to_user(
                    master_df, user_df, progress_bar, progress_text, 50, 100
                )

                progress_bar.progress(100)
                progress_text.text("✅ Processing complete!")

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

                st.success("✅ Results ready below")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

    else:
        st.info("👆 Please upload both Master and User files to begin")


if __name__ == "__main__":
    main()
