# Contributing to uproot

Thank you for your interest in contributing to **uproot**! uproot is a community-driven project, and we welcome contributions of all kinds, including bug reports, feature requests, documentation improvements, and code contributions.

This guide will help you get started with contributing.

---

## 🚀 Quick Start

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone git@github.com:YOUR_USERNAME/uproot5.git
   cd uproot
   ```

3. **Set up the development environment**:

   We recommend using a conda environment. You can reproduce the full developer environment with:

   ```bash
   conda create -n uproot-py313 python=3.13
   conda activate uproot-py313

   # Add conda-forge channel and prioritize it
   conda config --add channels conda-forge
   conda config --set channel_priority strict

   # Install dependencies
   conda install xrootd
   conda install root
   conda install pandas
   conda install dask

   # pip-only dependencies
   pip install scikit-hep-testdata
   pip install rangehttpserver
   pip install boost_histogram
   pip install hist
   pip install dask_awkward
   pip install pyarrow
   pip install pytest-timeout
   pip install fsspec-xrootd
   pip install uv

   # Run local HTTP server (if needed for test data)
   python -m RangeHTTPServer
   ```

4. **Install Uproot in editable mode**:
   ```bash
   uv pip install -e. --group=dev
   ```

---

## 🌿 Branching and Naming Conventions

- Always work on a **feature branch**:
  ```bash
  git checkout -b YOUR_USERNAME/my-cool-feature
  ```
- Use descriptive names, such as:
  - `fix_streamer-parsing`
  - `feature_custom-interpretation-api`
  - `docs_improve-tutorials`

---

## 🎨 Code Style

- Use **pre-commit** for automatic linting, style issues, and formatting:
  ```bash
  pre-commit run --all

---

## ✅ Running Tests

Uproot uses **pytest** for testing.

To run the full test suite:
```bash
python -m pytest tests -ss -vv
```

To run a specific test module:
```bash
pytest tests/test_my_module.py
```

Some tests may depend on having ROOT or XRootD installed. These are covered in the environment setup.

---

## 🔃 Submitting a Pull Request

1. Make sure all tests pass and your code is cleanly formatted.
2. Push your changes to your fork:
   ```bash
   git push origin YOUR_USERNAME/my-cool-feature
   ```
3. Open a pull request (PR) from your fork to the `main` branch of [scikit-hep/uproot](https://github.com/scikit-hep/uproot).
4. Fill in the PR template and explain what you did and why.
5. Be responsive to feedback from reviewers—we’re all here to help!

---

## 🤖 AI-assisted contributions

We welcome the use of AI tools as part of the development process. They can be valuable aids for drafting, refactoring, documentation, and exploration. However, contributions to Uproot require human judgment, contextual understanding, and familiarity with the project’s structure, goals, and standards.

When using AI tools:

* **You remain the author of the contribution.** Review, understand, and test all AI-assisted code or documentation before submitting it under your name. You should be able to explain and defend the changes on request.
* **Avoid fully automated submissions.** Issues or pull requests generated end-to-end by automated tools, without meaningful human review or intent, are not appropriate.
* **Be respectful of reviewers’ time.** Ensure that both the content of the PR and its description reflect your own understanding. Reviewers should not be expected to infer authorship or unknowingly interact with an AI during review.
* **Disclose significant AI assistance.** If AI tools were used for a substantial portion of the contribution, please note this in the PR description. (This guide is an example of that: we used ChatGPT to help with the writing, and in this comment, we acknowledge that fact.)

Contributors are responsible for the correctness, maintainability, and long-term impact of all submitted changes, regardless of whether AI tools were used.

Pull requests that do not meet these expectations may be closed without review.

---

## 🐛 Reporting Bugs

1. Check if the bug is already reported on the [issue tracker](https://github.com/scikit-hep/uproot/issues).
2. If not, [open a new issue](https://github.com/scikit-hep/uproot/issues/new/choose) and provide:
   - A minimal reproducible example
   - Expected vs. actual behavior
   - Version info (`python -m uproot --version`)
   - Any relevant stack trace or logs

---

## 💡 Requesting Features

1. Search the issues to see if a similar feature has been discussed.
2. If not, open a **Feature Request** issue and describe:
   - What problem the feature solves
   - A suggested implementation or interface (if applicable)
   - Any related prior art in other libraries or experiments

---

## 📚 Improving Documentation

Documentation lives in the `docs-sphinx/` folder and is built using **Sphinx**. To build locally
make sure you have Sphinx and the ReadTheDocs theme installed in your virtualenv:
```bash
pip install sphinx sphinx-rtd-theme
```
Navigate to your docs folder and invoke the Sphinx builder to produce HTML in the _build/html directory:
```
cd docs-sphinx
sphinx-build -b html . _build/html
```
Once it finishes, open:
```
open _build/html/index.html
```

You can also suggest improvements to examples, tutorials, and API references.

---

## 🙌 Thanks!

uproot thrives on its community. Whether you're fixing a typo, contributing a feature, or suggesting a design&mdash;you're making a difference!
