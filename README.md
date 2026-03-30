# Machine Learning Course

This repository is being prepared as a clean first content commit for the course.

The commit is intentionally minimal. It contains only the canonical lecture layer:

- lecture structure,
- lecture PDFs,
- practice notebooks,
- Markdown explanations.

## Included in This Commit Layer

- `AGENTS.md`
- `lectures/README.md`
- `lectures/lecture_xx_slug/README.md`
- `lectures/lecture_xx_slug/lecture_notes.md`
- `lectures/lecture_xx_slug/links.yaml`
- `lectures/lecture_xx_slug/slides/lecture.pdf`
- `lectures/lecture_xx_slug/assignment/practice.ipynb`

## What Is Not Included Yet

- solutions
- tooling and environment files
- reusable Python helpers
- publishing manifests
- raw imports and migration history
- extra source formats such as `pptx`

## Directory Layout

```text
lectures/
├── README.md
├── lecture_01_eda/
│   ├── README.md
│   ├── lecture_notes.md
│   ├── links.yaml
│   ├── slides/
│   │   └── lecture.pdf
│   └── assignment/
│       └── practice.ipynb
└── ...
```

## How to Read the Repository

For humans:

1. `README.md`
2. `lectures/README.md`
3. `lectures/<lecture_slug>/README.md`
4. `lectures/<lecture_slug>/lecture_notes.md`
5. `lectures/<lecture_slug>/links.yaml`
6. `lectures/<lecture_slug>/slides/lecture.pdf`
7. `lectures/<lecture_slug>/assignment/practice.ipynb`

For agents:

1. `AGENTS.md`
2. `lectures/README.md`
3. `lectures/<lecture_slug>/README.md`
4. `lectures/<lecture_slug>/lecture_notes.md`
5. `lectures/<lecture_slug>/links.yaml`
6. `lectures/<lecture_slug>/assignment/practice.ipynb`
