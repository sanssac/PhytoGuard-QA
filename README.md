# PhytoGuard — QA & Automated Testing Pipeline

![CI](https://github.com/sanssac/Phyto-Gaurd/actions/workflows/qa-pipeline.yml/badge.svg)

Automated quality assurance pipeline for the PhytoGuard plant disease detection system. Built to validate model inputs, preprocessing logic, inference outputs, and edge cases using Pytest and GitHub Actions CI/CD.

---

## Test Coverage

| Area | Tests |
|---|---|
| Input Validation | Shape, dtype, pixel range, NaN/Inf checks |
| Preprocessing | Normalization, resize, batch dimension |
| Model Output | Output shape, softmax sum, class range, confidence |
| Edge Cases | Blank image, noisy image, grayscale detection |

---

## Run Locally

```bash
# Install dependencies
pip install pytest pytest-html numpy opencv-python-headless

# Run tests
pytest tests/ -v

# Run with HTML report
pytest tests/ -v --html=reports/test-report.html --self-contained-html
```

---

## CI/CD Pipeline

Every push to `main` or `dev` automatically:
1. Sets up Python 3.10 environment
2. Installs all dependencies
3. Runs the full test suite
4. Generates and uploads an HTML test report as a build artifact

---

## Project Structure

```
phytoguard-qa/
├── tests/
│   └── test_model.py        # Full Pytest test suite
├── .github/
│   └── workflows/
│       └── qa-pipeline.yml  # GitHub Actions CI workflow
├── reports/                 # Auto-generated test reports
└── README.md
```

---

## Tech Stack
- Python 3.10
- Pytest
- OpenCV
- NumPy
- GitHub Actions
