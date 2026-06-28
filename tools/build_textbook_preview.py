"""Build a small interactive textbook preview from the OKF bundle."""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlsplit

from mlcourse.okf_validation import parse_markdown, validate_bundle

REPOSITORY_URL = "https://github.com/DerAndr/machine_learning_course_basics"
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)")


@dataclass(frozen=True)
class Page:
    """One rendered textbook page."""

    source: Path
    output: Path
    title: str
    description: str
    body: str
    metadata: dict[str, object]
    okf_path: str


def _is_url(target: str) -> bool:
    return urlsplit(target).scheme in {"http", "https", "mailto"}


def _route_for(markdown_path: Path, bundle: Path, output: Path) -> Path:
    relative = markdown_path.relative_to(bundle)
    if relative.name == "index.md":
        return output / relative.parent / "index.html"
    return output / relative.with_suffix(".html")


def _okf_path_for(markdown_path: Path, bundle: Path) -> str:
    return "/" + markdown_path.relative_to(bundle).as_posix()


def _title_from_body(body: str, fallback: str) -> str:
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def _resolve_markdown_target(source: Path, target: str, bundle: Path) -> Path | None:
    clean = target.split("#", 1)[0].split("?", 1)[0]
    if not clean or _is_url(clean) or clean.startswith("#"):
        return None
    candidate = bundle / clean.lstrip("/") if clean.startswith("/") else source.parent / clean
    if clean.endswith("/") or candidate.is_dir():
        candidate /= "index.md"
    return candidate.resolve()


def _relative_href(from_output: Path, to_output: Path) -> str:
    href = to_output.relative_to(from_output.parent, walk_up=True).as_posix()
    return quote(href, safe="/#-.")


def _rewrite_markdown_link(match: re.Match[str], source: Path, bundle: Path, output: Path) -> str:
    label, target = match.groups()
    if _is_url(target) or target.startswith("#"):
        return f"[{label}]({target})"
    fragment = ""
    if "#" in target:
        fragment = "#" + target.split("#", 1)[1]
    resolved = _resolve_markdown_target(source, target, bundle)
    if resolved is None:
        return f"[{label}]({target})"
    route = _route_for(resolved, bundle, output)
    current = _route_for(source, bundle, output)
    return f"[{label}]({_relative_href(current, route)}{fragment})"


def _inline_markdown(text: str) -> str:
    escaped = html.escape(text)
    return MARKDOWN_LINK_RE.sub(
        lambda match: (
            f'<a href="{html.escape(match.group(2), quote=True)}">{html.escape(match.group(1))}</a>'
        ),
        escaped,
    )


def _markdown_to_html(markdown: str) -> str:
    blocks: list[str] = []
    list_items: list[str] = []
    paragraph: list[str] = []
    in_code = False
    code_lines: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append(f"<p>{_inline_markdown(' '.join(paragraph))}</p>")
            paragraph.clear()

    def flush_list() -> None:
        if list_items:
            items = "\n".join(f"<li>{_inline_markdown(item)}</li>" for item in list_items)
            blocks.append(f"<ul>\n{items}\n</ul>")
            list_items.clear()

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        if line.startswith("```"):
            flush_paragraph()
            flush_list()
            if in_code:
                blocks.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
                code_lines.clear()
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_lines.append(raw_line)
            continue
        if not line.strip():
            flush_paragraph()
            flush_list()
            continue
        if line.startswith("#"):
            flush_paragraph()
            flush_list()
            level = len(line) - len(line.lstrip("#"))
            text = line[level:].strip()
            if 1 <= level <= 6 and text:
                blocks.append(f"<h{level}>{_inline_markdown(text)}</h{level}>")
                continue
        if line.startswith(("- ", "* ")):
            flush_paragraph()
            list_items.append(line[2:].strip())
            continue
        paragraph.append(line.strip())

    flush_paragraph()
    flush_list()
    if in_code:
        blocks.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
    return "\n".join(blocks)


def _load_pages(bundle: Path, output: Path) -> list[Page]:
    pages: list[Page] = []
    for source in sorted(bundle.rglob("*.md")):
        parsed = parse_markdown(source.read_text(encoding="utf-8-sig"))
        metadata = parsed.metadata or {}
        title = str(metadata.get("title") or _title_from_body(parsed.body, source.stem))
        description = str(metadata.get("description") or "")
        rewritten_body = MARKDOWN_LINK_RE.sub(
            lambda match, source=source: _rewrite_markdown_link(match, source, bundle, output),
            parsed.body,
        )
        pages.append(
            Page(
                source=source,
                output=_route_for(source, bundle, output),
                title=title,
                description=description,
                body=rewritten_body,
                metadata=metadata,
                okf_path=_okf_path_for(source, bundle),
            )
        )
    return pages


def _page_lookup(pages: list[Page]) -> dict[str, Page]:
    return {page.okf_path: page for page in pages}


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _metadata_badges_html(page: Page) -> str:
    badges = []
    page_type = page.metadata.get("type")
    if isinstance(page_type, str):
        badges.append(page_type)
    status = page.metadata.get("status")
    if isinstance(status, str):
        badges.append(status)
    badges.extend(_string_list(page.metadata.get("tags")))
    if not badges:
        return ""
    return (
        '<ul class="metadata-badges" aria-label="Page metadata">'
        + "".join(f"<li>{html.escape(item)}</li>" for item in badges)
        + "</ul>"
    )


def _learning_objectives_html(page: Page) -> str:
    objectives = _string_list(page.metadata.get("learning_objectives"))
    if not objectives:
        return ""
    return (
        '<section class="objectives"><h2>Learning objective</h2><ul>'
        + "".join(f"<li>{html.escape(item)}</li>" for item in objectives)
        + "</ul></section>"
    )


def _relationship_cards_html(
    page: Page,
    pages: list[Page],
    output: Path,
) -> str:
    lookup = _page_lookup(pages)
    groups = (
        ("Learning route", "related_concepts"),
        ("Try next", "related_labs"),
        ("Prerequisites", "prerequisites"),
    )
    sections = []
    for title, field in groups:
        cards = []
        for target in _string_list(page.metadata.get(field)):
            target_page = lookup.get(target)
            if target_page is None:
                continue
            href = _relative_href(page.output, target_page.output)
            cards.append(
                '<a class="relationship-card" '
                f'href="{html.escape(href, quote=True)}">'
                f"<span>{html.escape(str(target_page.metadata.get('type') or 'Page'))}</span>"
                f"<strong>{html.escape(target_page.title)}</strong>"
                f"<small>{html.escape(target_page.description)}</small>"
                "</a>"
            )
        if cards:
            sections.append(
                f'<section class="relationship-section"><h2>{title}</h2>'
                f'<div class="relationship-grid">{"".join(cards)}</div></section>'
            )
    return "".join(sections)


def _manifest_entry(page: Page, output: Path) -> dict[str, object]:
    return {
        "id": page.okf_path.removeprefix("/").removesuffix(".md"),
        "okf_path": page.okf_path,
        "title": page.title,
        "description": page.description,
        "type": page.metadata.get("type"),
        "status": page.metadata.get("status"),
        "tags": _string_list(page.metadata.get("tags")),
        "learning_objectives": _string_list(page.metadata.get("learning_objectives")),
        "prerequisites": _string_list(page.metadata.get("prerequisites")),
        "related_concepts": _string_list(page.metadata.get("related_concepts")),
        "related_labs": _string_list(page.metadata.get("related_labs")),
        "source_materials": _string_list(page.metadata.get("source_materials")),
        "textbook_path": page.output.relative_to(output, walk_up=True).as_posix(),
    }


def _write_manifest(pages: list[Page], output: Path) -> None:
    concepts = [
        _manifest_entry(page, output)
        for page in pages
        if page.source.name != "index.md" and page.metadata
    ]
    payload = {
        "schema": "ml-course-okf-manifest-v1",
        "description": "Agent-readable index for the interactive textbook preview.",
        "concept_count": len(concepts),
        "concepts": concepts,
    }
    (output / "okf-manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _source_materials_html(page: Page) -> str:
    sources = page.metadata.get("source_materials")
    if not isinstance(sources, list):
        return ""
    items = []
    for source in sources:
        if not isinstance(source, str):
            continue
        if source.startswith("/"):
            href = f"{REPOSITORY_URL}/blob/main/{quote(source.lstrip('/'), safe='/#-.')}"
        else:
            href = source
        items.append(
            f'<li><a href="{html.escape(href, quote=True)}">{html.escape(source)}</a></li>'
        )
    if not items:
        return ""
    return (
        '<section class="source-materials"><h2>Source materials</h2><ul>'
        + "".join(items)
        + "</ul></section>"
    )


def _lab_fallback_table(data_path: Path) -> str:
    data = json.loads(data_path.read_text(encoding="utf-8"))
    thresholds = [0.3, 0.5, 0.7]
    rows = []
    for threshold in thresholds:
        counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        for item in data["examples"]:
            predicted = item["score"] >= threshold
            actual = bool(item["label"])
            if predicted and actual:
                counts["tp"] += 1
            elif predicted and not actual:
                counts["fp"] += 1
            elif not predicted and actual:
                counts["fn"] += 1
            else:
                counts["tn"] += 1
        precision = (
            counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0
        )
        recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        rows.append(
            "<tr>"
            f"<td>{threshold:.2f}</td><td>{counts['tp']}</td><td>{counts['fp']}</td>"
            f"<td>{counts['tn']}</td><td>{counts['fn']}</td>"
            f"<td>{precision:.2f}</td><td>{recall:.2f}</td><td>{f1:.2f}</td>"
            "</tr>"
        )
    return (
        '<section class="lab-fallback"><h2>Static fallback</h2>'
        "<p>This table works without JavaScript and shows the same score set "
        "at three thresholds.</p>"
        "<table><thead><tr><th>Threshold</th><th>TP</th><th>FP</th><th>TN</th><th>FN</th>"
        "<th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></section>"
    )


def _interactive_lab_html(page: Page, output: Path, data_path: Path) -> str:
    if (
        page.source.as_posix()
        .replace("\\", "/")
        .endswith("labs/classification-threshold-explorer.md")
    ):
        data_href = _relative_href(page.output, output / "data" / data_path.name)
        return (
            '<section class="interactive-lab" data-threshold-lab '
            f'data-url="{html.escape(data_href, quote=True)}">'
            "<h2>Interactive threshold explorer</h2>"
            '<label for="threshold-slider">Threshold: '
            '<output id="threshold-value">0.50</output></label>'
            '<input id="threshold-slider" type="range" min="0" max="1" step="0.01" value="0.5">'
            '<div class="metric-grid" aria-live="polite"></div>'
            '<div class="example-table"></div>'
            "</section>" + _lab_fallback_table(data_path)
        )
    return ""


def _navigation_html(pages: list[Page], current: Page) -> str:
    links = []
    for page in pages:
        href = _relative_href(current.output, page.output)
        label = html.escape(page.title)
        marker = ' aria-current="page"' if page.output == current.output else ""
        links.append(f'<li><a href="{html.escape(href, quote=True)}"{marker}>{label}</a></li>')
    return (
        '<nav class="sidebar" aria-label="Textbook pages"><h2>Pages</h2><ul>'
        + "".join(links)
        + "</ul></nav>"
    )


def _render_page(page: Page, pages: list[Page], output: Path, data_path: Path) -> str:
    css_href = _relative_href(page.output, output / "assets" / "textbook.css")
    js_href = _relative_href(page.output, output / "assets" / "threshold-lab.js")
    home_href = _relative_href(page.output, output / "index.html")
    body = _markdown_to_html(page.body)
    badges = _metadata_badges_html(page)
    objectives = _learning_objectives_html(page)
    relationships = _relationship_cards_html(page, pages, output)
    lab = _interactive_lab_html(page, output, data_path)
    sources = _source_materials_html(page)
    manifest_href = _relative_href(page.output, output / "okf-manifest.json")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(page.title)} · ML Course Interactive Textbook</title>
  <meta name="description" content="{html.escape(page.description, quote=True)}">
  <link rel="stylesheet" href="{html.escape(css_href, quote=True)}">
</head>
<body>
  <header class="site-header">
    <a href="{html.escape(home_href, quote=True)}">ML Course Interactive Textbook</a>
    <a class="manifest-link" href="{html.escape(manifest_href, quote=True)}">Agent manifest</a>
  </header>
  <div class="layout">
    {_navigation_html(pages, page)}
    <main>
      {badges}
      {body}
      {objectives}
      {relationships}
      {lab}
      {sources}
    </main>
  </div>
  <script src="{html.escape(js_href, quote=True)}" defer></script>
</body>
</html>
"""


def build_textbook_preview(
    bundle: Path = Path("okf"),
    site: Path = Path("site"),
    output: Path | None = None,
) -> Path:
    """Build the textbook preview and return the output directory."""

    repository_root = Path.cwd()
    result = validate_bundle(bundle, repository_root=repository_root)
    if result.errors or result.warnings:
        details = "\n".join(
            f"{item.severity.upper()} {item.code} {item.path}: {item.message}"
            for item in result.diagnostics
        )
        raise SystemExit(f"OKF validation must pass before rendering.\n{details}")

    build_dir = output or site / "_build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)

    shutil.copytree(site / "assets", build_dir / "assets")
    shutil.copytree(site / "data", build_dir / "data")

    pages = _load_pages(bundle.resolve(), build_dir.resolve())
    _write_manifest(pages, build_dir.resolve())
    data_path = (site / "data" / "classification-threshold-scores.json").resolve()
    for page in pages:
        page.output.parent.mkdir(parents=True, exist_ok=True)
        page.output.write_text(
            _render_page(page, pages, build_dir.resolve(), data_path), encoding="utf-8"
        )

    return build_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=Path("okf"), type=Path)
    parser.add_argument("--site", default=Path("site"), type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = build_textbook_preview(args.bundle, args.site, args.output)
    print(f"Built textbook preview at {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
