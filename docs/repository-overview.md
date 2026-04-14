# Repository Overview

This repository stores the authoring version of the machine learning course.

## Responsibilities

- Maintain lecture-ready assignments and post-class solutions.
- Keep reusable code outside notebooks.
- Preserve a clean migration path from legacy materials.
- Provide the metadata needed to publish a student-facing repository later.

## Main Areas

- `lectures/` contains one directory per lecture.
- `src/mlcourse/` contains shared helper code.
- `publish/lectures.yaml` tracks assignment readiness and solution release state.
- `legacy_import/` is reserved for raw legacy materials during migration.

## Current State

All 14 lectures are in place with lecture notes, slide decks, example notebooks, and practical sessions.
The normalization and bug-audit pass for lectures `01` to `14` is complete.
Lecture-specific dependency groups are available for heavier libraries (ensembles, time series, HPO/AutoML, neural networks, XAI/PiML, ML in production).
