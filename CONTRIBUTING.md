# Contributing to DocVision AI

We welcome contributions from the machine learning and computer vision developer community! Please review the guidelines below to submit improvements.

---

## 1. Code Style Guidelines

To keep the codebase modular, legible, and easy to maintain, we follow these python standards:
- **Style Formatter**: We use [Black](https://github.com/psf/black) (`black .` format).
- **Linter Quality Checks**: We use [Flake8](https://flake8.pycqa.org/) to detect syntax warnings and unused imports.
- **Type Annotations**: Ensure all service methods and utility functions specify typing properties (e.g. `List`, `Dict`, `Tuple`, `Optional`).

---

## 2. Development Workflow

1. **Fork** and create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-awesome-feature
   ```
2. Make your edits inside modular classes (`backend/services/`, `utils/`, etc.).
3. Add automated tests inside `tests/` matching your changes.
4. Verify that the **entire test suite** passes successfully:
   ```bash
   python -m unittest discover -s tests
   ```

---

## 3. Pull Request Submission

- Provide a clear, detailed PR description describing what changes were made and how they were verified.
- Confirm that type checks and Black formatter runs do not raise errors.
- PRs must be approved by the core repository maintainers before merging.
