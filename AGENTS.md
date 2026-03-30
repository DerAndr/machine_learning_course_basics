# Agent Guide

This repository is being staged for a minimal first content commit.

## Scope of This Commit Layer

Treat the following as the only in-scope content:

- `README.md`
- `lectures/README.md`
- `lectures/<lecture_slug>/README.md`
- `lectures/<lecture_slug>/lecture_notes.md`
- `lectures/<lecture_slug>/links.yaml`
- `lectures/<lecture_slug>/slides/lecture.pdf`
- `lectures/<lecture_slug>/assignment/practice.ipynb`

## Canonical Traversal Order

1. `README.md`
2. `lectures/README.md`
3. `lectures/<lecture_slug>/README.md`
4. `lectures/<lecture_slug>/lecture_notes.md`
5. `lectures/<lecture_slug>/links.yaml`
6. `lectures/<lecture_slug>/slides/lecture.pdf`
7. `lectures/<lecture_slug>/assignment/practice.ipynb`

## Intentional Exclusions

For this commit, do not depend on:

- solutions
- tooling files
- source code under `src/`
- docs outside the minimal root and lecture readmes
- `incoming_materials/`
- `legacy_import/`
- `publish/`

## Meaning of Core Files

- `lecture_notes.md`: student-facing recap and revision notes
- `links.yaml`: compact lecture metadata with lecture number, slug, title, YouTube, and slides URL
- `slides/lecture.pdf`: canonical lecture deck for this commit layer
- `assignment/practice.ipynb`: practical notebook for the lecture
- `README.md`: navigation and purpose, not theory content

## Editing Guidance

If asked to refine the first commit layer:

- keep the structure simple
- prefer English-only repository text
- keep one canonical PDF and one canonical practice notebook per lecture
- do not re-introduce raw source duplication into this layer
