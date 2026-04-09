import pandas as pd

# load the data
input_dir = "/data/projects/CSC/code/XTC/07_regression_model/Input/processed/01_with_id"
X_pre = pd.read_csv(f"{input_dir}/X_pre_with_id.csv")
X_delta = pd.read_csv(f"{input_dir}/X_delta_with_id.csv")
X_pre_delta = pd.read_csv(f"{input_dir}/X_pre_delta_with_id.csv")
y_df_final = pd.read_csv(f"{input_dir}/y_with_id.csv")

# Standardize ID columns
for df in [X_pre, X_delta, X_pre_delta, y_df_final]:
    df["cohort"] = pd.to_numeric(df["cohort"], errors="coerce").astype("Int64")
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")

print("Loaded processed files with IDs.")
print("X_pre:", X_pre.shape)
print("X_delta:", X_delta.shape)
print("X_pre_delta:", X_pre_delta.shape)
print("y_df_final:", y_df_final.shape)

# =========================================================
# Set the paired sessions used to create these files
# =========================================================
pre_session = 1
post_session = 3   # change to 2 if your paired data are session1->session2

# =========================================================
# 1. QC exclusion lists
# =========================================================
failed_qc = {
    1: {"I031", "I051", "I061", "I062", "I078", "I087", "I162", "I169", "I170",
        "I196", "I206", "II389", "II745", "II1025", "III027", "III037", "III038", "IV003"},
    2: {"I107", "I123"},
    3: {"I033", "I037", "I054", "I123", "I162", "I166", "I170"},
}

manual_exclusions = {
    1: {"I104", "I244"},
    3: {"I244", "I180"},
}

exclude_by_session = {}
for sess in [1, 2, 3]:
    exclude_by_session[sess] = failed_qc.get(sess, set()) | manual_exclusions.get(sess, set())

# =========================================================
# 2. Helper to build subject_code from cohort + id
# =========================================================
roman_map = {1: "I", 2: "II", 3: "III", 4: "IV"}

def add_subject_code(df):
    df = df.copy()
    df["cohort"] = pd.to_numeric(df["cohort"], errors="coerce").astype("Int64")
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df["subject_code"] = (
        df["cohort"].map(roman_map) +
        df["id"].astype(str).str.zfill(3)
    )
    return df

# =========================================================
# 3. Filter using fixed paired sessions
# =========================================================
def apply_qc_filter_fixed_sessions(df, pre_sess, post_sess):
    df = add_subject_code(df)

    exclude_mask = df["subject_code"].isin(exclude_by_session.get(pre_sess, set())) | \
                   df["subject_code"].isin(exclude_by_session.get(post_sess, set()))

    excluded_df = df.loc[exclude_mask].copy()
    kept_df = df.loc[~exclude_mask].copy()

    kept_df["session_pre"] = pre_sess
    kept_df["session_post"] = post_sess

    excluded_df["session_pre"] = pre_sess
    excluded_df["session_post"] = post_sess

    return kept_df, excluded_df

# =========================================================
# 4. Apply to all datasets
# =========================================================
X_pre_filt, X_pre_excluded = apply_qc_filter_fixed_sessions(X_pre, pre_session, post_session)
X_delta_filt, X_delta_excluded = apply_qc_filter_fixed_sessions(X_delta, pre_session, post_session)
X_pre_delta_filt, X_pre_delta_excluded = apply_qc_filter_fixed_sessions(X_pre_delta, pre_session, post_session)
y_filt, y_excluded = apply_qc_filter_fixed_sessions(y_df_final, pre_session, post_session)

# =========================================================
# 5. Check consistency
# =========================================================
print("Before filtering:")
print("X_pre:", X_pre.shape)
print("X_delta:", X_delta.shape)
print("X_pre_delta:", X_pre_delta.shape)
print("y:", y_df_final.shape)

print("\nAfter filtering:")
print("X_pre_filt:", X_pre_filt.shape)
print("X_delta_filt:", X_delta_filt.shape)
print("X_pre_delta_filt:", X_pre_delta_filt.shape)
print("y_filt:", y_filt.shape)

x_keys = set(map(tuple, X_pre_delta_filt[["cohort", "id"]].drop_duplicates().values))
y_keys = set(map(tuple, y_filt[["cohort", "id"]].drop_duplicates().values))

print("\nShared filtered keys:", len(x_keys & y_keys))
print("X only:", len(x_keys - y_keys))
print("Y only:", len(y_keys - x_keys))

# =========================================================
# 6. Save filtered datasets
# =========================================================
out_dir = "/data/projects/CSC/code/XTC/07_regression_model/Input/processed"

X_pre_filt.to_csv(f"{out_dir}/X_pre_with_id_filteredQC.csv", index=False)
X_delta_filt.to_csv(f"{out_dir}/X_delta_with_id_filteredQC.csv", index=False)
X_pre_delta_filt.to_csv(f"{out_dir}/X_pre_delta_with_id_filteredQC.csv", index=False)
y_filt.to_csv(f"{out_dir}/y_with_id_filteredQC.csv", index=False)

X_pre_excluded.to_csv(f"{out_dir}/X_pre_excludedQC.csv", index=False)
X_delta_excluded.to_csv(f"{out_dir}/X_delta_excludedQC.csv", index=False)
X_pre_delta_excluded.to_csv(f"{out_dir}/X_pre_delta_excludedQC.csv", index=False)
y_excluded.to_csv(f"{out_dir}/y_excludedQC.csv", index=False)

print("\nSaved filtered and excluded CSVs.")