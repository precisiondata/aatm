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

Docs are versioned with [mike](https://github.com/jimporter/mike) and
deployed to GitHub Pages by `.github/workflows/docs.yml`:

* every push to `main` updates `dev`, the latest unreleased docs;
* every `v*` tag publishes that version, aliases it `latest` and makes it
  the default the site root redirects to;
* older tags can be (re)built from the Actions tab: run **Docs** manually
  with the tag, ticking *latest* only for the newest release.

To preview the version selector locally, run `mike serve` after a
`mike deploy dev` (this only touches your local `gh-pages` branch).
PyPI releases are published by `.github/workflows/release.yml` when a `v*`
tag is pushed. Only repository admins can create `v*` tags.

### Bumping the version

The version lives only in `pyproject.toml` (and is mirrored in `uv.lock`).
We follow [Semantic Versioning](https://semver.org): `patch` for fixes,
`minor` for new features, `major` for breaking changes.

1. **Branch off an up-to-date `main`:**
   ```bash
   git checkout main
   git pull
   git checkout -b chore/release-X.Y.Z
   ```
2. **Bump the version** with uv, which updates both `pyproject.toml` and
   `uv.lock`:
   ```bash
   uv version --bump patch   # or minor / major, or an exact version: uv version X.Y.Z
   ```
3. **Commit, push and open a PR:**
   ```bash
   git commit -am "chore: bump version to X.Y.Z"
   git push -u origin chore/release-X.Y.Z
   gh pr create --title "chore: release X.Y.Z"
   ```
   `main` is protected: the PR needs passing CI and one approval (admins
   can bypass).
4. **After the PR merges, tag the merge commit on `main` and push the tag:**
   ```bash
   git checkout main
   git pull
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```
5. **Approve the publish.** The release workflow runs CI and checks that the
   tagged commit is on `main` and that the tag matches the project version.
   It then builds the package and pauses at the `pypi` environment. Open the
   run in the Actions tab, click **Review deployments** and approve it. The
   new version then appears on <https://pypi.org/project/aatm/>.

If the workflow fails before publishing, fix the problem on a new PR.
Then delete and re-push the tag (`git tag -d vX.Y.Z`,
`git push origin :refs/tags/vX.Y.Z`, then repeat step 4). Once a version
has been uploaded to PyPI it can't be reused: bump to the next version
instead.

---

## Reporting Bugs & Suggesting Features

Before opening a new tracker item, search our existing GitHub Issues to see if the topic has been discussed.
* **Bugs**: Provide a minimal reproducible example, your operating system details, and Python package versions.
* **Features**: Open a feature tracking issue to discuss major scope enhancements before writing code.
