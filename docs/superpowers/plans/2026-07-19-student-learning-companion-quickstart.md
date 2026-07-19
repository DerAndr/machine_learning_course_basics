# Student Learning Companion Quickstart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give students copy-ready prompts and accurate instructions for using the repository learning-companion skills locally or installing the generic skill for other projects.

**Architecture:** Add one canonical student guide and link it from the three existing entry points. Keep detailed authoring mechanics in the operational guide and keep the ML-course adapter repository-scoped.

**Tech Stack:** Markdown, pytest, pathlib

## Global Constraints

- Preserve the generic-core plus repository-adapter architecture.
- Use the official Codex skill locations: repository skills in `.agents/skills` and personal skills in `$HOME/.agents/skills`.
- Do not package a plugin in this change.
- Do not modify the skill implementations.

---

### Task 1: Student quickstart contract and guide

**Files:**
- Create: `docs/student-learning-companion-quickstart.md`
- Modify: `README.md`
- Modify: `docs/student-quickstart.md`
- Modify: `docs/interactive-lecture-learning-assistant.md`
- Modify: `tests/test_interactive_learning_assistant_docs.py`

**Interfaces:**
- Consumes: the existing generic core and ML-course adapter skill names and paths.
- Produces: a canonical student guide linked from all student and authoring entry points.

- [ ] **Step 1: Write the failing documentation test**

Add a test that requires the new guide, its local and personal-installation sections, generic and course-specific prompt examples, both platform copy commands, and links from the three existing entry points.

- [ ] **Step 2: Run the targeted test and verify it fails**

Run:

```powershell
uv run pytest tests/test_interactive_learning_assistant_docs.py -q
```

Expected: failure because `docs/student-learning-companion-quickstart.md` and its entry-point links do not exist.

- [ ] **Step 3: Write the student guide and entry-point links**

Create the guide with:

- a no-install route to prepared reviews;
- repository-local Codex discovery and `$skill-name` invocation instructions;
- copy-ready ML-course prompts;
- a reusable generic prompt template and a concrete non-ML example;
- Windows and macOS/Linux personal-installation commands for the complete generic skill directory;
- troubleshooting and source-safety guidance;
- an official OpenAI Codex skills documentation link.

Link the guide from `README.md`, `docs/student-quickstart.md`, and `docs/interactive-lecture-learning-assistant.md`.

- [ ] **Step 4: Run the targeted test and verify it passes**

Run:

```powershell
uv run pytest tests/test_interactive_learning_assistant_docs.py -q
```

Expected: all tests in the file pass.

- [ ] **Step 5: Run repository verification**

Run:

```powershell
uv run pytest
uv run ruff check tests/test_interactive_learning_assistant_docs.py
```

Expected: the full suite and Ruff pass.

- [ ] **Step 6: Commit**

```powershell
git add README.md docs/student-quickstart.md docs/student-learning-companion-quickstart.md docs/interactive-lecture-learning-assistant.md tests/test_interactive_learning_assistant_docs.py docs/superpowers/plans/2026-07-19-student-learning-companion-quickstart.md
git commit -m "docs: add student learning companion quickstart"
```
