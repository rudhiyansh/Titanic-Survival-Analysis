# Titanic Survival Analysis

**Turning 891 rows of passenger data into a clear, evidence-backed story about who survived the Titanic — and why.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?logo=pandas&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4C72B0)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Plotting-11557C)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Build-Passing-brightgreen)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture / How It Works](#architecture--how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Contributing](#contributing)
- [License](#license)
- [Author / Contact](#author--contact)

---

## Overview

The Titanic dataset is one of the most widely referenced datasets in data science, but most public notebooks that use it stop at surface-level charts without explaining the reasoning behind each cleaning decision or drawing conclusions that hold up to scrutiny. This project treats the dataset the way a working analyst would treat a real business dataset: audit it, clean it deliberately, document every transformation, and only then move to interpretation. The goal is not just to produce plots, but to answer a specific question — which factors most strongly predicted survival on the Titanic — using a reproducible, well-reasoned workflow. Every step, from handling missing `Age` and `Embarked` values to bucketing passengers into age groups, is done with a stated rationale rather than a default pandas call. The end result is a notebook that reads less like an exercise and more like a short analytical report, which is exactly the standard this project holds itself to.

## Features

- **Structural data audit** — Uses `.info()`, `.shape`, `.describe()`, and `.columns` to fully understand the dataset's shape, types, and distributions before any cleaning begins, rather than jumping straight into transformations.
- **Deliberate missing-data strategy** — The `Cabin` column is dropped outright due to its high proportion of missing values, `Age` is imputed with the column mean to preserve sample size, and the two rows missing `Embarked` are dropped since imputing a categorical port of origin would introduce more noise than value.
- **Duplicate and consistency checks** — Explicitly verifies there are no duplicate rows and inspects the unique values in categorical fields like `Sex` to catch encoding issues before they silently affect results.
- **Survival rate breakdowns** — Computes precise survival percentages overall, and segmented by gender and passenger class, giving quantitative backing to every visual claim instead of relying on the chart alone.
- **Age-group binning** — Converts continuous `Age` values into meaningful categories (Child, Teen, Young Adult, Adult, Senior) using `pd.cut`, making age-based survival trends interpretable at a glance rather than buried in a scatter of raw numbers.
- **Multi-variable visualization** — Goes beyond single-factor analysis with a combined class-and-gender survival plot, surfacing interaction effects (e.g., first-class women vs. third-class men) that single-variable charts miss entirely.
- **Distribution analysis** — Includes a KDE-overlaid histogram of passenger age to characterize the underlying population, not just the outcome variable.
- **Narrative conclusions** — Closes with a written summary that ties every chart back to a real historical explanation (the "women and children first" protocol), grounding the statistics in context instead of leaving them to speak for themselves.

## Tech Stack

- **Python** was chosen as the base language because it remains the standard for data analysis work, with a mature ecosystem and syntax that keeps the focus on the data rather than boilerplate.
- **Pandas** handles all data loading, cleaning, and aggregation. It was chosen over raw Python data structures because operations like `groupby`, `fillna`, and `dropna` express cleaning logic in a single readable line instead of manual loops, which matters when every transformation needs to be auditable.
- **NumPy** underpins the numerical operations pandas relies on and is used directly where explicit array-level operations are clearer than a pandas equivalent.
- **Matplotlib** provides the low-level plotting control needed to set precise figure sizes, titles, and axis labels — details that make the difference between a chart that's readable and one that isn't.
- **Seaborn** is layered on top of Matplotlib for the statistical plots (`countplot`, `barplot`, `histplot`) because it produces cleaner default aesthetics and handles categorical grouping (via `hue`) with far less code than Matplotlib alone.
- **Jupyter Notebook** was the natural choice for this project's format since analysis work benefits from interleaving code, output, and commentary in one linear, inspectable document — exactly how the conclusions in this project were derived and should be reviewed.

## Architecture / How It Works

The notebook follows a linear, four-stage pipeline, and each stage only begins once the previous one is verified complete:

1. **Ingest** — The raw `train.csv` file (the classic Kaggle Titanic training set) is loaded into a pandas DataFrame.
2. **Audit & Clean** — The DataFrame is inspected for shape, types, and nulls. Based on that audit, `Cabin` is dropped, `Age` nulls are mean-imputed, `Embarked` nulls are row-dropped, and duplicate rows are confirmed absent. Each decision is made independently based on how much signal that column contributes versus how much missing data it carries.
3. **Feature Derivation** — A new `AgeGroup` categorical feature is derived from the continuous `Age` column using fixed bin edges, enabling group-level survival comparisons that raw ages can't provide directly.
4. **Analysis & Visualization** — Survival rate is computed overall and sliced by `Sex`, `Pclass`, and `AgeGroup`, with each numerical breakdown paired with a corresponding Seaborn visualization. The notebook closes by cross-tabulating `Pclass` and `Sex` together against survival to capture interaction effects that single-variable analysis would miss, then synthesizes all findings into a written conclusion.

Because each stage depends on the output of the one before it, running the notebook top to bottom reproduces the full analysis deterministically — there are no hidden branches or out-of-order cell dependencies.

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/titanic-survival-analysis.git
cd titanic-survival-analysis

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install pandas numpy matplotlib seaborn jupyter

# 4. Launch the notebook
jupyter notebook titanic_survival_analysis.ipynb
```

The dataset itself (`train.csv`) is the standard Kaggle Titanic training set. Download it from the [Kaggle Titanic competition page](https://www.kaggle.com/c/titanic/data) and place it in a `data/` folder in the project root, then update the file path in the first code cell to match.

## Usage

Once the notebook is running, execute the cells sequentially. A few representative examples of what each stage outputs:

**Checking overall survival rate:**

```python
survival_rate = df["Survived"].mean() * 100
print(f"Overall survival rate: {survival_rate:.2f}%")
```
```
Overall survival rate: 40.19%
```

**Survival rate by gender:**

```python
print(df.groupby("Sex")["Survived"].mean() * 100)
```
```
Sex
female    74.20
male      18.89
Name: Survived, dtype: float64
```

**Deriving age groups and comparing survival:**

```python
bins = [0, 12, 18, 35, 60, 80]
labels = ["Child", "Teen", "Young Adult", "Adult", "Senior"]
df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels)

print(df.groupby("AgeGroup", observed=True)["Survived"].mean() * 100)
```

Each analytical cell is immediately followed by a Seaborn visualization (`countplot` or `barplot`) so the numeric result and its visual counterpart are always presented together — the numbers justify the chart, and the chart makes the numbers intuitive.

## Folder Structure

```
titanic-survival-analysis/
│
├── data/
│   └── train.csv                    # Raw Kaggle Titanic dataset (not committed)
│
├── titanic_survival_analysis.ipynb  # Main analysis notebook
├── README.md                        # Project documentation
├── requirements.txt                 # Pinned dependency versions
└── LICENSE                          # MIT License
```

## Contributing

Contributions are welcome, particularly around extending the analysis with additional cleaning strategies or new segment comparisons. To contribute:

1. Fork the repository and create a feature branch (`git checkout -b feature/fare-vs-survival`).
2. Keep each analytical addition in its own notebook cell with a preceding markdown cell explaining the question it answers — this project's core value is that every step is explainable, and new contributions should hold to that same standard.
3. If you change an existing cleaning decision (for example, using median instead of mean imputation for `Age`), state the reasoning in a markdown cell rather than silently swapping the method.
4. Run the full notebook top to bottom before submitting to confirm there are no broken cell dependencies.
5. Open a pull request with a clear description of what was added or changed and why it improves the analysis.
6. Keep visualizations consistent with the existing style (figure size `(10,6)`, titled axes, and a labeled legend) so the notebook reads as one coherent document rather than a patchwork of styles.

Issues and suggestions are just as valuable as code — if you spot a flawed assumption in the existing analysis, opening an issue with your reasoning is genuinely useful.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details. You are free to use, modify, and distribute this work with attribution.

## Author / Contact

**Rudhiyansh Vijay Sandanshiv**

Data analysis enthusiast building a portfolio of well-reasoned, reproducible EDA projects on the path toward a data analyst career.

- GitHub: [github.com/your-username](https://github.com/your-rudhiyansh)

If this analysis was useful or you spot something worth discussing, feel free to open an issue or reach out directly.
