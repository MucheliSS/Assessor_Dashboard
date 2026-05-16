# assessor_dashboard.py
import io
import re
import textwrap
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

METRIC_EXPLAINERS = {
    "Forms submitted": "Number of assessment forms submitted by this assessor.",
    "Residents assessed": "Number of distinct residents assessed; higher values suggest broader sampling.",
    "Competencies covered": "Number of EPA competency domains with at least one valid score.",
    "Median submission delay": "Typical days between assessment and submission; shorter delays usually preserve detail.",
    "Median score": "Middle score across this assessor's valid ratings after GM normalization.",
    "Scores recorded": "Number of non-zero, non-blank numeric scores contributing to score summaries.",
    "Adjusted stringency-leniency index": "Actual score minus expected score for comparable SR level and competency mix.",
    "Comment presence": "Percentage of feedback rows with a non-empty narrative comment.",
    "Median word count": "Typical length of narrative comments when measured by word count.",
    "Actionability": "Percentage of comments containing action-oriented language such as improve, focus, should, or try.",
}


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


def safe_number(value):
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_metric(value, suffix: str = "", decimals: int = 1) -> str:
    number = safe_number(value)
    if number is None:
        return "NA"
    if float(number).is_integer() and decimals == 0:
        return f"{int(number)}{suffix}"
    return f"{number:.{decimals}f}{suffix}"


def selected_value(table: pd.DataFrame, assessor: str, column: str):
    if table.empty or column not in table.columns:
        return pd.NA
    rows = table[table["Assessor"] == assessor]
    if rows.empty:
        return pd.NA
    return rows.iloc[0][column]


def cohort_median(table: pd.DataFrame, column: str):
    if table.empty or column not in table.columns:
        return pd.NA
    return pd.to_numeric(table[column], errors="coerce").median()


def metric_delta(table: pd.DataFrame, assessor: str, column: str, suffix: str = "", decimals: int = 1) -> str:
    value = safe_number(selected_value(table, assessor, column))
    median = safe_number(cohort_median(table, column))
    if value is None or median is None:
        return ""
    difference = value - median
    sign = "+" if difference > 0 else ""
    return f"{sign}{difference:.{decimals}f}{suffix} vs cohort median"


def comparison_row(
    table: pd.DataFrame,
    assessor: str,
    column: str,
    label: str,
    suffix: str = "",
    decimals: int = 1,
):
    value = selected_value(table, assessor, column)
    median = cohort_median(table, column)
    return {
        "Measure": label,
        "Selected assessor": format_metric(value, suffix=suffix, decimals=decimals),
        "Cohort median": format_metric(median, suffix=suffix, decimals=decimals),
        "Difference": metric_delta(table, assessor, column, suffix=suffix, decimals=decimals),
        "How to read it": METRIC_EXPLAINERS.get(label, ""),
    }


def build_report_card_table(
    assessor: str,
    contribution: pd.DataFrame,
    distribution: pd.DataFrame,
    stringency: pd.DataFrame,
    quality: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        comparison_row(contribution, assessor, "Forms submitted", "Forms submitted", decimals=0),
        comparison_row(contribution, assessor, "Residents assessed", "Residents assessed", decimals=0),
        comparison_row(contribution, assessor, "Competencies covered", "Competencies covered", decimals=0),
        comparison_row(contribution, assessor, "Median submission delay", "Median submission delay", suffix=" days"),
        comparison_row(distribution, assessor, "Median", "Median score"),
        comparison_row(distribution, assessor, "Score_count", "Scores recorded", decimals=0),
        comparison_row(stringency, assessor, "Adjusted_index", "Adjusted stringency-leniency index", decimals=3),
        comparison_row(quality, assessor, "Comment presence", "Comment presence", suffix="%"),
        comparison_row(quality, assessor, "Median word count", "Median word count", decimals=0),
        comparison_row(quality, assessor, "Actionability", "Actionability", suffix="%"),
    ]
    return pd.DataFrame(rows)


def selected_vs_cohort_scores(long_scores: pd.DataFrame, assessor: str) -> pd.DataFrame:
    if long_scores.empty:
        return pd.DataFrame()
    scores = long_scores.copy()
    scores["Comparison group"] = scores["Assessor"].where(
        scores["Assessor"] == assessor,
        "Other assessors",
    )
    return scores


def selected_competency_profile(long_scores: pd.DataFrame, assessor: str) -> pd.DataFrame:
    if long_scores.empty:
        return pd.DataFrame()
    profile = (
        long_scores.assign(
            Comparison=long_scores["Assessor"].where(long_scores["Assessor"] == assessor, "Other assessors")
        )
        .groupby(["Comparison", "Competency"], as_index=False)["Score"]
        .mean()
    )
    return profile


def pdf_safe_text(value) -> str:
    text = "" if pd.isna(value) else str(value)
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00b7": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def escape_pdf_text(value) -> str:
    text = pdf_safe_text(value)
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def wrap_pdf_line(label: str, value, width: int = 95) -> List[str]:
    text = f"{label}: {pdf_safe_text(value)}" if label else pdf_safe_text(value)
    return textwrap.wrap(text, width=width) or [""]


def report_card_pdf_lines(
    assessor: str,
    report_table: pd.DataFrame,
    selected_quality: pd.DataFrame,
    selected_stringency: pd.DataFrame,
    selected_coverage: pd.DataFrame,
) -> List[str]:
    lines = [
        f"Assessor Report Card: {pdf_safe_text(assessor)}",
        "Comparisons are descriptive and should be interpreted with case mix, SR level, and competency mix in mind.",
        "",
        "Cohort comparison",
    ]

    for _, row in report_table.iterrows():
        summary = (
            f"{row['Measure']} | Selected: {row['Selected assessor']} | "
            f"Cohort median: {row['Cohort median']} | Difference: {row['Difference'] or 'NA'}"
        )
        lines.extend(wrap_pdf_line("", summary))
        if row.get("How to read it"):
            lines.extend(wrap_pdf_line("Meaning", row["How to read it"]))
        lines.append("")

    if not selected_coverage.empty:
        lines.append("Competency coverage")
        coverage_row = selected_coverage.iloc[0]
        coverage_text = ", ".join(f"{col}: {coverage_row[col]}" for col in selected_coverage.columns)
        lines.extend(wrap_pdf_line("", coverage_text))
        lines.append("")

    if not selected_quality.empty:
        lines.append("Feedback quality")
        for col in selected_quality.columns:
            if col != "Assessor":
                lines.extend(wrap_pdf_line(col, selected_quality.iloc[0][col]))
        lines.append("")

    if not selected_stringency.empty:
        lines.append("Stringency-leniency")
        for col in selected_stringency.columns:
            if col != "Assessor":
                lines.extend(wrap_pdf_line(col, selected_stringency.iloc[0][col]))

    return lines


def build_simple_pdf(lines: List[str]) -> bytes:
    pages = []
    max_lines_per_page = 44
    for start in range(0, len(lines), max_lines_per_page):
        pages.append(lines[start:start + max_lines_per_page])

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        None,
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    page_object_numbers = []

    for page_lines in pages:
        content = io.StringIO()
        content.write("BT\n/F1 15 Tf\n50 760 Td\n")
        title = page_lines[0] if page_lines else "Assessor Report Card"
        content.write(f"({escape_pdf_text(title)}) Tj\n0 -24 Td\n/F1 9 Tf\n12 TL\n")
        for line in page_lines[1:]:
            content.write(f"({escape_pdf_text(line)}) Tj\nT*\n")
        content.write("ET\n")
        stream = content.getvalue().encode("latin-1", errors="replace")
        content_object = len(objects) + 1
        page_object = len(objects) + 2
        objects.append(b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"endstream")
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_object} 0 R >>".encode("ascii")
        )
        page_object_numbers.append(page_object)

    kids = " ".join(f"{num} 0 R" for num in page_object_numbers)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_object_numbers)} >>".encode("ascii")

    output = io.BytesIO()
    output.write(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(output.tell())
        output.write(f"{index} 0 obj\n".encode("ascii"))
        output.write(obj)
        output.write(b"\nendobj\n")

    xref_start = output.tell()
    output.write(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.write(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.write(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_start}\n%%EOF\n".encode("ascii")
    )
    return output.getvalue()


def report_card_pdf_bytes(
    assessor: str,
    report_table: pd.DataFrame,
    selected_quality: pd.DataFrame,
    selected_stringency: pd.DataFrame,
    selected_coverage: pd.DataFrame,
) -> bytes:
    lines = report_card_pdf_lines(
        assessor,
        report_table,
        selected_quality,
        selected_stringency,
        selected_coverage,
    )
    return build_simple_pdf(lines)


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    return cleaned or "assessor"


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

    contribution = assessor_contribution_table(df_quant, context)
    long_scores = make_long_scores(df_quant, context["assessor_col"], context["score_cols"])
    distribution = score_distribution_summary(long_scores)
    stringency = adjusted_stringency_index(df_quant, context)
    coverage = competency_coverage(df_quant, context)
    quality = feedback_quality_summary(df_quant, df_qual, context)

    assessor_options = sorted(df_quant[context["assessor_col"]].dropna().unique().tolist())

    tab_report, tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Report Card",
            "Contribution",
            "Score Distribution",
            "Stringency Index",
            "Competency Coverage",
            "Feedback Quality",
        ]
    )

    with tab_report:
        st.subheader("Assessor Report Card")
        if not assessor_options:
            st.info("No assessors available for report-card view.")
        else:
            selected_assessor = st.selectbox("Choose Assessor", assessor_options)
            st.caption("Comparisons are descriptive and should be interpreted with case mix, SR level, and competency mix in mind.")

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            with c1:
                st.metric(
                    "Forms",
                    format_metric(selected_value(contribution, selected_assessor, "Forms submitted"), decimals=0),
                    metric_delta(contribution, selected_assessor, "Forms submitted", decimals=0),
                )
            with c2:
                st.metric(
                    "Residents",
                    format_metric(selected_value(contribution, selected_assessor, "Residents assessed"), decimals=0),
                    metric_delta(contribution, selected_assessor, "Residents assessed", decimals=0),
                )
            with c3:
                st.metric(
                    "Competencies",
                    format_metric(selected_value(contribution, selected_assessor, "Competencies covered"), decimals=0),
                    metric_delta(contribution, selected_assessor, "Competencies covered", decimals=0),
                )
            with c4:
                st.metric(
                    "Delay",
                    format_metric(selected_value(contribution, selected_assessor, "Median submission delay"), suffix=" days"),
                    metric_delta(contribution, selected_assessor, "Median submission delay", suffix=" days"),
                )
            with c5:
                st.metric(
                    "Comment Presence",
                    format_metric(selected_value(quality, selected_assessor, "Comment presence"), suffix="%"),
                    metric_delta(quality, selected_assessor, "Comment presence", suffix="%"),
                )
            with c6:
                st.metric(
                    "Adjusted Index",
                    format_metric(selected_value(stringency, selected_assessor, "Adjusted_index"), decimals=3),
                    metric_delta(stringency, selected_assessor, "Adjusted_index", decimals=3),
                )

            st.write("### Cohort Comparison")
            report_table = build_report_card_table(
                selected_assessor,
                contribution,
                distribution,
                stringency,
                quality,
            )
            st.dataframe(report_table, use_container_width=True)

            selected_quality = quality[quality["Assessor"] == selected_assessor] if not quality.empty else pd.DataFrame()
            selected_stringency = (
                stringency[stringency["Assessor"] == selected_assessor]
                if not stringency.empty
                else pd.DataFrame()
            )
            selected_coverage = (
                coverage.loc[[selected_assessor]]
                if not coverage.empty and selected_assessor in coverage.index
                else pd.DataFrame()
            )

            pdf_bytes = report_card_pdf_bytes(
                selected_assessor,
                report_table,
                selected_quality,
                selected_stringency,
                selected_coverage,
            )
            st.download_button(
                "Download Report Card PDF",
                data=pdf_bytes,
                file_name=f"assessor_report_card_{safe_filename(selected_assessor)}.pdf",
                mime="application/pdf",
            )

            st.write("### Score Use")
            comparison_scores = selected_vs_cohort_scores(long_scores, selected_assessor)
            if comparison_scores.empty:
                st.info("No valid scores available for this assessor.")
            else:
                fig = px.box(
                    comparison_scores,
                    x="Comparison group",
                    y="Score",
                    color="Comparison group",
                    points="all",
                    hover_data=["Assessor", "Competency"],
                    title=f"{selected_assessor} vs Other Assessors",
                )
                fig.update_layout(
                    yaxis=dict(range=[0, 5.5], title="Score"),
                    xaxis_title="",
                    showlegend=False,
                    margin=dict(l=20, r=20, t=55, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

            left, right = st.columns(2)
            with left:
                st.write("### Competency Profile")
                competency_profile = selected_competency_profile(long_scores, selected_assessor)
                if competency_profile.empty:
                    st.info("No competency score profile available.")
                else:
                    fig = px.bar(
                        competency_profile,
                        x="Competency",
                        y="Score",
                        color="Comparison",
                        barmode="group",
                        title="Average Score by Competency",
                    )
                    fig.update_layout(
                        yaxis=dict(range=[0, 5], title="Average Score"),
                        margin=dict(l=20, r=20, t=55, b=20),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                if not selected_coverage.empty:
                    st.dataframe(selected_coverage, use_container_width=True)

            with right:
                st.write("### Feedback Quality")
                if selected_quality.empty:
                    st.info("No feedback quality data available for this assessor.")
                else:
                    st.dataframe(selected_quality, use_container_width=True)

                if not selected_stringency.empty:
                    st.write("### Stringency-Leniency")
                    st.dataframe(selected_stringency, use_container_width=True)

    with tab1:
        st.subheader("Assessor Contribution")
        if contribution.empty:
            st.info("No assessor contribution data available.")
        else:
            st.dataframe(contribution, use_container_width=True)

    with tab2:
        st.subheader("Assessor Score Distribution")
        if long_scores.empty:
            st.info("No valid scores available for distribution plotting.")
        else:
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
