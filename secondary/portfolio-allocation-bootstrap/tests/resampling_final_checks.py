import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CODE_PATH = BASE_DIR / "resampling_code.py"
SLIDES_CODE_PATH = BASE_DIR / "resampling_slides_code.py"
DATA_PATH = BASE_DIR / "resampling_data.csv"
RESULTS_PATH = BASE_DIR / "resampling_bootstrap.csv"
PDF_PATH = BASE_DIR / "resampling_slides.pdf"
SUMMARY_PATH = BASE_DIR / "resampling_summary.txt"
EXPLANATION_PATH = BASE_DIR / "resampling_solution_explanation.md"


def check(condition, message):
    ### Stops the final checks with a clear message if a requirement fails
    if not condition:
        raise AssertionError(message)


def run_script(script_name):
    ### Runs an project script from the project folder
    completed = subprocess.run(
        [sys.executable, script_name],
        cwd=BASE_DIR,
        text=True,
        capture_output=True,
        check=False,
    )
    check(
        completed.returncode == 0,
        f"{script_name} failed\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}",
    )
    return completed.stdout.strip()


def source_text(path):
    ### Reads a Python source file for static submission checks
    return path.read_text(encoding="utf-8")


#===========================================================
#   CHECKING REQUIRED INPUT AND SOURCE FILES

for required_path in [DATA_PATH, CODE_PATH, SLIDES_CODE_PATH, SUMMARY_PATH, EXPLANATION_PATH]:
    check(required_path.exists(), f"missing required file: {required_path.name}")

data = pd.read_csv(DATA_PATH)
check(list(data.columns) == ["asset1_return", "asset2_return"], "data columns are not as expected")
check(len(data) == 100, "data should contain 100 paired return rows")

#===========================================================
#   CHECKING STATIC CODE REQUIREMENTS

code_text = source_text(CODE_PATH)
slides_code_text = source_text(SLIDES_CODE_PATH)
combined_source = code_text + "\n" + slides_code_text

for bad_pattern in [
    "np.random." + "choice",
    "random." + "choice",
    "random." + "sample",
    "sklearn.utils." + "resample",
    ".sample" + "(",
]:
    check(bad_pattern not in combined_source, f"forbidden resampling helper found: {bad_pattern}")

check("np.floor(n * u)" in code_text, "bootstrap index conversion floor(n * U) not found")
check("rng.random((b, n))" in code_text, "uniform bootstrap draws not found")
check("SUMMARY_PATH" not in code_text, "summary writing should not be part of the main code script")
check("#===========================================================" in code_text, "main code is missing section separators")
check("#===========================================================" in slides_code_text, "slides code is missing section separators")
check("resampling_bootstrap.csv" not in slides_code_text, "slides code should not write the result CSV")

for path_literal in ["resampling_data.csv", "resampling_bootstrap.csv"]:
    check(not os.path.isabs(path_literal), f"path should be relative: {path_literal}")

#===========================================================
#   RUNNING REPRODUCIBILITY SCRIPTS

print(run_script("resampling_code.py"))
print(run_script("resampling_slides_code.py"))

#===========================================================
#   CHECKING SUBMISSION OUTPUTS

results = pd.read_csv(RESULTS_PATH)
check(list(results.columns) == ["alpha_hat", "ci_lower", "ci_upper"], "result columns are incorrect")
check(results.shape == (1, 3), "result CSV should contain exactly one row and three columns")

alpha_hat = float(results.loc[0, "alpha_hat"])
ci_lower = float(results.loc[0, "ci_lower"])
ci_upper = float(results.loc[0, "ci_upper"])

check(np.isfinite(alpha_hat), "alpha_hat is not finite")
check(np.isfinite(ci_lower), "ci_lower is not finite")
check(np.isfinite(ci_upper), "ci_upper is not finite")
check(ci_lower < alpha_hat < ci_upper, "alpha_hat should lie inside the bootstrap interval")
check(ci_lower < ci_upper, "confidence interval endpoints are not ordered")

summary = SUMMARY_PATH.read_text(encoding="utf-8").strip()
check("minimum-variance allocation" in summary, "summary should explain the model")
check("alpha_hat = 0.706418" in summary, "summary does not match the final alpha estimate")

check(PDF_PATH.exists(), "slide PDF was not created")
check(PDF_PATH.stat().st_size > 50000, "slide PDF looks too small")

try:
    import fitz

    doc = fitz.open(PDF_PATH)
    check(doc.page_count == 3, "slide PDF should contain exactly 3 pages")
except ImportError:
    print("PyMuPDF is not installed; skipped PDF page-count check.")

print("All final checks passed.")
