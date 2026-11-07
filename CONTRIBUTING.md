# Contributing to aatm

Thank you for your interest in improving `aatm`! We welcome all contributions, including bug fixes, new features, documentation enhancements, and issue triaging.

---

## Local Development Setup

We use modern packaging standards with configurations centralized in `pyproject.toml`. 

### Prerequisites

* Python 3.11+
* [uv](https://github.com/astral-sh/uv) (recommended) or `pip` / `virtualenv`

### Step-by-Step Environment Setup

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/{your-username}/aatm.git
   cd aatm
   ```
3. **Install the project and development dependencies** from the lockfile
   (the same environment CI uses):
   ```bash
   uv sync --all-extras
   ```
   This installs the CUDA 12.8 build of PyTorch. On a machine without an
   NVIDIA GPU, use the CPU-only build instead:
   ```bash
   uv sync --all-extras --no-group cu128 --group cpu
   ```
4. **Before pushing**, run the same checks as CI:
   ```bash
   sh scripts/precommit.sh
   ```

---

## Code Quality Standards

To maintain code consistency, we enforce strict linting and formatting rules on our CI/CD pipeline.

* **Formatting & Linting**: We use [Ruff](https://github.com/astral-sh/ruff). Run checks locally via:
  ```bash
  uv run ruff format aatm/ tests/
  uv run ruff check aatm/ tests/
  ```
* **Type Hinting**: All new public APIs must include static type hints verified by `mypy`.
  ```bash
  uv run mypy aatm/
  ```
* **Style Constraints**: Use `snake_case` for function and variable names, and `PascalCase` for classes.

---

## Testing Guidelines

We use [pytest](https://pytest.org) for our test suite. 

* Every feature addition or bug fix must include corresponding tests.
* We target a test coverage of **80%**. CI fails if total coverage drops below the
  floor set in `[tool.coverage.report] fail_under` in `pyproject.toml`; raise that
  floor when your change increases coverage.
* Run the test suite locally with coverage reporting:
  ```bash
  uv run pytest --cov
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

## Releasing (maintainers)

Docs are deployed to GitHub Pages automatically on every push to `main`.
To publish a new version to PyPI, bump `version` in `pyproject.toml`, merge
it to `main`, then push a matching tag:

```bash
git tag v0.1.8
git push origin v0.1.8
```

---

## Reporting Bugs & Suggesting Features

Before opening a new tracker item, search our existing GitHub Issues to see if the topic has been discussed.
* **Bugs**: Provide a minimal reproducible example, your operating system details, and Python package versions.
* **Features**: Open a feature tracking issue to discuss major scope enhancements before writing code.
