# Assessor Dashboard

Streamlit dashboard for reviewing assessor-level EPA assessment patterns from Excel.

## Outputs

- Assessor contribution table: forms submitted, residents assessed, competencies covered, median submission delay.
- Assessor score distribution: shows whether each assessor uses the full rating scale.
- Adjusted stringency-leniency index: actual score minus expected score by SR level and competency.
- Competency heatmap: assessor by competency coverage.
- Narrative feedback quality: comment presence, word count, and actionability flag.

## Excel Format

Required sheet:

- `Quantitative`

Optional sheet:

- `Qualitative`

Useful columns:

- Assessor: `Name of Evaluator`, `Assessor`, or `Evaluator`
- Resident: `Resident Name`
- SR level: `SR Level`, `Training Level`, `Level`, or similar
- Domain scores: `PC`, `MK`, `SBP`, `PBLI`, `Prof`, `ICS`, `Overall`
- Dates: `Assessment Date` and `Submission Date`
- Comments: `Comments`, `Remarks`, `Feedback`, or similar

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run assessor_dashboard.py
```

