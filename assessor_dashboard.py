# assessor_dashboard.py
import re
from typing import List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st


DOMAIN_COLUMNS = ["PC", "MK", "SBP", "PBLI", "Prof", "ICS"]
ALL_SCORE_COLUMNS = DOMAIN_COLUMNS + ["Overall"]

ASSESSOR_CANDIDATES = ["Name of Evaluator", "Assessor", "Evaluator", "Faculty", "Faculty Name"]
RESIDENT_CANDIDATES = ["Resident Name", "Resident", "SR Name", "Trainee"]
LEVEL_CANDIDATES = ["SR Level", "Senior Resident Level", "Training Level", "Level", "PGY", "Residency Year"]
ASSESSMENT_DATE_CANDIDATES = ["Assessment Date", "Date of Assessment", "Feedback Date", "Date of Feedback", "Date"]
SUBMISSION_DATE_CANDIDATES = ["Submission Date", "Date Submitted", "Submitted Date", "Timestamp", "Created Date"]
COMMENT_CANDIDATES = [
    "Comments",
    "Comment",
    "Remarks",
    "Remark",
    "Feedback",
    "Qualitative Feedback",
    "Narrative Feedback",
    "Additional Comments",
]

ACTIONABILITY_TERMS = [
    "action",
    "consider",
    "develop",
    "focus",
    "improve",
    "next time",
    "plan",
    "practice",
    "recommend",
    "should",
    "try",
    "work on",
]


def norm_text_series(series: pd.Series) -> pd.Series:
    return series.astype("string").str.replace(r"\s+", " ", regex=True).str.strip()


def column_tokens(name: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(name).lower())


def norm_column_name(name: str) -> str:
    return " ".join(column_tokens(name))


def pick_first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    normalized = {norm_column_name(c): c for c in df.columns}
    for candidate in candidates:
        match = normalized.get(norm_column_name(candidate))
        if match is not None:
            return match
    return None


def pick_comment_column(df: pd.DataFrame) -> Optional[str]:
    exact = pick_first_present(df, COMMENT_CANDIDATES)
    if exact:
        return exact

    for col in df.columns:
        tokens = set(column_tokens(col))
        is_comment = tokens.intersection({"comment", "comments", "remark", "remarks", "feedback"})
        is_metadata = tokens.intersection({"date", "month", "time", "timestamp"})
        if is_comment and not is_metadata:
            return col
    return None


def clean_scores(df: pd.DataFrame, score_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if score_cols:
        df[score_cols] = df[score_cols].mask(df[score_cols].eq(0))
    return df


def detect_context(df: pd.DataFrame) -> dict:
    score_cols = [col for col in ALL_SCORE_COLUMNS if col in df.columns]
    domain_cols = [col for col in DOMAIN_COLUMNS if col in df.columns]
    return {
        "assessor_col": pick_first_present(df, ASSESSOR_CANDIDATES),
        "resident_col": pick_first_present(df, RESIDENT_CANDIDATES),
        "level_col": pick_first_present(df, LEVEL_CANDIDATES),
        "assessment_date_col": pick_first_present(df, ASSESSMENT_DATE_CANDIDATES),
        "submission_date_col": pick_first_present(df, SUBMISSION_DATE_CANDIDATES),
        "score_cols": score_cols,
        "domain_cols": domain_cols,
    }


def load_workbook(uploaded_file):
    df_quant = pd.read_excel(uploaded_file, sheet_name="Quantitative")
    uploaded_file.seek(0)
    try:
        df_qual = pd.read_excel(uploaded_file, sheet_name="Qualitative")
    except Exception:
        df_qual = pd.DataFrame()
    return df_quant, df_qual


def prepare_quantitative(df_quant: pd.DataFrame, context: dict) -> pd.DataFrame:
    df = df_quant.copy()
    for col in [context["assessor_col"], context["resident_col"], context["level_col"]]:
        if col and col in df.columns:
            df[col] = norm_text_series(df[col])
    df = clean_scores(df, context["score_cols"])
    if "Assessment Type" in df.columns and context["score_cols"]:
        is_gm = df["Assessment Type"].astype(str).str.contains("GM", case=False, na=False)
        df.loc[is_gm, context["score_cols"]] = df.loc[is_gm, context["score_cols"]] / 2.0
    return df


def make_long_scores(df: pd.DataFrame, assessor_col: str, score_cols: List[str], extra_cols=None) -> pd.DataFrame:
    extra_cols = extra_cols or []
    id_vars = [assessor_col] + [col for col in extra_cols if col and col in df.columns]
    long_df = df.melt(
        id_vars=id_vars,
        value_vars=score_cols,
        var_name="Competency",
        value_name="Score",
    )
    long_df = long_df.rename(columns={assessor_col: "Assessor"})
    return long_df.dropna(subset=["Assessor", "Score"])


def median_submission_delay_days(df: pd.DataFrame, assessment_date_col: Optional[str], submission_date_col: Optional[str]):
    if not assessment_date_col or not submission_date_col or assessment_date_col == submission_date_col:
        return pd.Series(dtype="float64")

    assessment_dates = pd.to_datetime(df[assessment_date_col], errors="coerce")
    submission_dates = pd.to_datetime(df[submission_date_col], errors="coerce")
    delay_days = (submission_dates - assessment_dates).dt.days
    return delay_days.mask(delay_days.lt(0))


def assessor_contribution_table(df: pd.DataFrame, context: dict) -> pd.DataFrame:
    assessor_col = context["assessor_col"]
    resident_col = context["resident_col"]
    domain_cols = context["domain_cols"]

    rows = []
    delay = median_submission_delay_days(
        df,
        context["assessment_date_col"],
        context["submission_date_col"],
    )

    for assessor, group in df.groupby(assessor_col, dropna=True):
        competencies = [col for col in domain_cols if group[col].notna().any()]
        delay_value = delay.loc[group.index].median() if not delay.empty else pd.NA
        rows.append(
            {
                "Assessor": assessor,
                "Forms submitted": int(group.shape[0]),
                "Residents assessed": int(group[resident_col].nunique()) if resident_col else pd.NA,
                "Competencies covered": len(competencies),
                "Coverage detail": ", ".join(competencies) if competencies else "None",
                "Median submission delay": delay_value,
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values(["Forms submitted", "Residents assessed"], ascending=False)


def score_distribution_summary(long_scores: pd.DataFrame) -> pd.DataFrame:
    if long_scores.empty:
        return pd.DataFrame()
    return (
        long_scores.groupby("Assessor")["Score"]
        .agg(Score_count="count", Median="median", Min="min", Max="max")
        .round(2)
        .reset_index()
    )


def adjusted_stringency_index(df: pd.DataFrame, context: dict) -> pd.DataFrame:
    assessor_col = context["assessor_col"]
    level_col = context["level_col"]
    score_cols = context["domain_cols"] or context["score_cols"]

    extra_cols = [level_col] if level_col else []
    long_df = make_long_scores(df, assessor_col, score_cols, extra_cols=extra_cols)
    if long_df.empty:
        return pd.DataFrame()

    expected_keys = [level_col, "Competency"] if level_col else ["Competency"]
    expected = (
        long_df.groupby(expected_keys, dropna=False)["Score"]
        .mean()
        .rename("Expected score")
        .reset_index()
    )
    adjusted = long_df.merge(expected, on=expected_keys, how="left")
    adjusted["Adjusted difference"] = adjusted["Score"] - adjusted["Expected score"]

    summary = (
        adjusted.groupby("Assessor")["Adjusted difference"]
        .agg(Adjusted_index="mean", Median_difference="median", Score_count="count")
        .round(3)
        .reset_index()
        .sort_values("Adjusted_index", ascending=False)
    )
    return summary


def competency_coverage(df: pd.DataFrame, context: dict) -> pd.DataFrame:
    assessor_col = context["assessor_col"]
    domain_cols = context["domain_cols"]
    long_df = make_long_scores(df, assessor_col, domain_cols)
    if long_df.empty:
        return pd.DataFrame()
    coverage = (
        long_df.groupby(["Assessor", "Competency"])["Score"]
        .count()
        .rename("Assessments")
        .reset_index()
    )
    return coverage.pivot(index="Assessor", columns="Competency", values="Assessments").fillna(0).astype(int)


def word_count(text) -> int:
    if pd.isna(text):
        return 0
    return len(re.findall(r"\b\w+\b", str(text)))


def is_actionable(text) -> bool:
    if pd.isna(text):
        return False
    value = str(text).lower()
    return any(term in value for term in ACTIONABILITY_TERMS)


def feedback_quality_summary(df_quant: pd.DataFrame, df_qual: pd.DataFrame, context: dict) -> pd.DataFrame:
    feedback_df = df_qual.copy() if not df_qual.empty else df_quant.copy()
    comment_col = pick_comment_column(feedback_df)
    if not comment_col:
        return pd.DataFrame()

    assessor_col = pick_first_present(feedback_df, ASSESSOR_CANDIDATES) or context["assessor_col"]
    if not assessor_col or assessor_col not in feedback_df.columns:
        feedback_df["Assessor"] = "All assessors"
        assessor_col = "Assessor"

    feedback_df[assessor_col] = norm_text_series(feedback_df[assessor_col])
    feedback_df[comment_col] = norm_text_series(feedback_df[comment_col])
    feedback_df["Has comment"] = feedback_df[comment_col].notna() & feedback_df[comment_col].str.len().gt(0)
    feedback_df["Word count"] = feedback_df[comment_col].apply(word_count)
    feedback_df["Actionable"] = feedback_df[comment_col].apply(is_actionable)

    summary = (
        feedback_df.groupby(assessor_col)
        .agg(
            Feedback_rows=(comment_col, "size"),
            Comment_presence=("Has comment", "mean"),
            Median_word_count=("Word count", "median"),
            Mean_word_count=("Word count", "mean"),
            Actionable_comments=("Actionable", "mean"),
        )
        .reset_index()
        .rename(columns={assessor_col: "Assessor"})
    )
    summary["Comment presence"] = (summary["Comment_presence"] * 100).round(1)
    summary["Actionability"] = (summary["Actionable_comments"] * 100).round(1)
    summary["Median word count"] = summary["Median_word_count"].round(1)
    summary["Mean word count"] = summary["Mean_word_count"].round(1)
    return summary[
        ["Assessor", "Feedback_rows", "Comment presence", "Median word count", "Mean word count", "Actionability"]
    ].sort_values("Feedback_rows", ascending=False)


st.set_page_config(page_title="Assessor Dashboard", layout="wide")
st.title("Assessor Dashboard")

uploaded_file = st.file_uploader("Upload your EPA Excel file", type=["xlsx"])

if not uploaded_file:
    st.info("Upload an EPA Excel file to review assessor contribution, scoring patterns, and feedback quality.")
else:
    try:
        df_quant_raw, df_qual = load_workbook(uploaded_file)
    except Exception as exc:
        st.error(f"Could not load workbook: {exc}")
        st.stop()

    context = detect_context(df_quant_raw)
    if not context["assessor_col"]:
        st.error("No assessor column found. Expected one of: Name of Evaluator, Assessor, Evaluator.")
        st.stop()
    if not context["score_cols"]:
        st.error("No score columns found. Expected one or more of: PC, MK, SBP, PBLI, Prof, ICS, Overall.")
        st.stop()

    df_quant = prepare_quantitative(df_quant_raw, context)

    st.success("Data loaded. Zero, blank, NA, and non-numeric scores are excluded from score summaries.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Contribution",
            "Score Distribution",
            "Stringency Index",
            "Competency Coverage",
            "Feedback Quality",
        ]
    )

    with tab1:
        st.subheader("Assessor Contribution")
        contribution = assessor_contribution_table(df_quant, context)
        if contribution.empty:
            st.info("No assessor contribution data available.")
        else:
            st.dataframe(contribution, use_container_width=True)

    with tab2:
        st.subheader("Assessor Score Distribution")
        long_scores = make_long_scores(df_quant, context["assessor_col"], context["score_cols"])
        if long_scores.empty:
            st.info("No valid scores available for distribution plotting.")
        else:
            distribution = score_distribution_summary(long_scores)
            fig = px.box(
                long_scores,
                x="Assessor",
                y="Score",
                color="Assessor",
                points="all",
                hover_data=["Competency"],
                title="Score Distribution by Assessor",
            )
            fig.update_layout(
                yaxis=dict(range=[0, 5.5], title="Score"),
                xaxis_tickangle=-35,
                showlegend=False,
                margin=dict(l=20, r=20, t=55, b=100),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(distribution, use_container_width=True)

    with tab3:
        st.subheader("Adjusted Stringency-Leniency Index")
        st.caption("Positive values suggest more lenient scoring; negative values suggest more stringent scoring.")
        stringency = adjusted_stringency_index(df_quant, context)
        if stringency.empty:
            st.info("No valid scores available for adjusted stringency analysis.")
        else:
            fig = px.bar(
                stringency,
                x="Assessor",
                y="Adjusted_index",
                color="Adjusted_index",
                color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
                title="Actual Score Minus Expected Score",
                labels={"Adjusted_index": "Adjusted Index"},
            )
            fig.update_layout(xaxis_tickangle=-35, margin=dict(l=20, r=20, t=55, b=100))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(stringency, use_container_width=True)

    with tab4:
        st.subheader("Competency Heatmap")
        coverage = competency_coverage(df_quant, context)
        if coverage.empty:
            st.info("No competency coverage data available.")
        else:
            fig = px.imshow(
                coverage,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Blues",
                title="Assessor x Competency Coverage",
                labels={"x": "Competency", "y": "Assessor", "color": "Assessments"},
            )
            fig.update_layout(margin=dict(l=20, r=20, t=55, b=20))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(coverage, use_container_width=True)

    with tab5:
        st.subheader("Narrative Feedback Quality")
        quality = feedback_quality_summary(df_quant, df_qual, context)
        if quality.empty:
            st.info("No narrative feedback column found in the workbook.")
        else:
            st.dataframe(quality, use_container_width=True)
            fig = px.scatter(
                quality,
                x="Median word count",
                y="Actionability",
                size="Feedback_rows",
                color="Comment presence",
                hover_name="Assessor",
                title="Feedback Depth and Actionability",
                labels={
                    "Actionability": "Actionable comments (%)",
                    "Comment presence": "Comment presence (%)",
                },
            )
            fig.update_layout(margin=dict(l=20, r=20, t=55, b=20))
            st.plotly_chart(fig, use_container_width=True)
