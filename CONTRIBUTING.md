# Contributing to aatm

Thank you for your interest in improving `aatm`! We welcome all contributions, including bug fixes, new features, documentation enhancements, and issue triaging.

---

## Local Development Setup

We use modern packaging standards with configurations centralized in `pyproject.toml`. 

### Prerequisites

* Python 3.10+
* [uv](https://github.com) (recommended) or `pip` / `virtualenv`

### Step-by-Step Environment Setup

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/{your-username}/aatm.git
   cd aatm
   ```
3. **Create and activate a virtual environment**:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```
4. **Install development dependencies** in editable mode:
   ```bash
   uv pip install -e ".[dev]"
   ```

---

## Code Quality Standards

To maintain code consistency, we enforce strict linting and formatting rules on our CI/CD pipeline.

* **Formatting & Linting**: We use [Ruff](https://github.com). Run checks locally via:
  ```bash
  ruff format .
  ruff check .
  ```
* **Type Hinting**: All new public APIs must include static type hints verified by `mypy`.
  ```bash
  mypy aatm/
  ```
* **Style Constraints**: Use `snake_case` for function and variable names, and `PascalCase` for classes.

---

## Testing Guidelines

We use [pytest](https://pytest.org) for our test suite. 

* Every feature addition or bug fix must include corresponding tests.
* We target a minimum test coverage of **80%**.
* Run the test suite locally with coverage reporting:
  ```bash
  pytest tests/
  ```

---

## Documentation

Clear documentation keeps our codebase maintainable. 

* **Docstrings**: Follow [PEP 257](https://python.org) style standards using NumPy or Google style layouts.
* **Building Docs Locally**: Our documentation is compiled using MkDocs. Build it to preview your changes:
  ```bash
  mkdocs build
  mkdocs serve
  ```

---

## Pull Request Process

Ready to submit your changes? Follow this standard workflow:

1. **Create a feature branch** off the main branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Commit your changes** using clean, descriptive commit messages. We prefer [Conventional Commits](https://conventionalcommits.org) (e.g., `feat: add parallel processing to extractor`).
3. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
4. **Open a Pull Request (PR)** against our `main` branch.
5. **Ensure all CI checks pass** (tests, linting, type checks). A maintainer will review your code shortly.

---

## Reporting Bugs & Suggesting Features

Before opening a new tracker item, search our existing GitHub Issues to see if the topic has been discussed.
* **Bugs**: Provide a minimal reproducible example, your operating system details, and Python package versions.
* **Features**: Open a feature tracking issue to discuss major scope enhancements before writing code.
