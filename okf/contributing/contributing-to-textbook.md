---
type: Reference
title: Contributing to the Interactive Textbook
description: A contribution workflow for improving the OKF textbook as both student-facing content and agent-readable knowledge.
tags: [foundations]
timestamp: 2026-06-28T00:00:00Z
status: draft
difficulty: introductory
estimated_reading_minutes: 6
related_concepts:
  - /learning-paths/classification-part-1.md
source_materials:
  - /docs/contributing-to-textbook.md
---

# Contributing to the Interactive Textbook

## Core idea

The textbook is a knowledge system, not only a website. Contributions should improve three surfaces at the same time:

- the student-facing concept or lab;
- the OKF metadata and relationships that organize the knowledge;
- the agent-readable manifest that lets agents navigate and improve the course.

## Contribution loop

- Start from a real learner problem.
- Read the relevant lecture notes and existing OKF pages.
- Improve or add the smallest useful concept, lab, or learning path.
- Include formulas and assumptions when the topic requires them.
- Keep descriptions, indexes, skills, and learning objectives synchronized.
- Validate the OKF bundle and rebuild the textbook preview.
- Check the deployed page and manifest after merge.

## Quality bar

Do not add shallow summaries. A good contribution should help a student answer a precise question and should help an agent locate the concept later.

For mathematical topics, formulas are not optional decoration. They are often the clearest version of the idea. Explain each formula in words and connect it to model behavior or decision-making.

## Go deeper

Read the full [contribution guide](https://github.com/DerAndr/machine_learning_course_basics/blob/main/docs/contributing-to-textbook.md) before opening a larger textbook change.
