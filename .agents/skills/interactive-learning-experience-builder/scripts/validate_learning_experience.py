import argparse
import json
import re
from html.parser import HTMLParser
from pathlib import Path

SETTINGS = {"difficulty", "focus", "color-blind", "break-prompts"}
NATIVE_CONTROLS = {"button", "input", "select", "textarea"}
INTERACTIVE_ROLES = {
    "button",
    "checkbox",
    "combobox",
    "radio",
    "slider",
    "spinbutton",
    "switch",
}
KEYBOARD_HANDLERS = {"onkeydown", "onkeypress", "onkeyup"}
VOID_ELEMENTS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
NETWORK_URL = re.compile(r"^(?:https?:)?//", re.IGNORECASE)
VISIBLE_FOCUS = re.compile(
    r":focus-visible\s*\{[^}]*(?:outline|box-shadow)\s*:\s*(?!none\b|0(?:\D|$))",
    re.IGNORECASE | re.DOTALL,
)
NETWORK_CAPABILITIES = (
    re.compile(r"\bfetch\s*\(", re.IGNORECASE),
    re.compile(r"\b(?:XMLHttpRequest|WebSocket|EventSource)\b", re.IGNORECASE),
    re.compile(r"\bnavigator\s*\.\s*sendBeacon\s*\(", re.IGNORECASE),
    re.compile(r"\bimport\s*\(", re.IGNORECASE),
    re.compile(
        r"\bdocument\s*\.\s*createElement\s*\(\s*['\"]"
        r"(?:script|link|img|iframe|audio|video|source)['\"]",
        re.IGNORECASE,
    ),
    re.compile(
        r"\.\s*(?:src|href)\s*=\s*['\"]"
        r"(?:(?:https?|wss?):)?//",
        re.IGNORECASE,
    ),
)
CONTENT_DATA_SCRIPT = re.compile(
    r"^\s*const\s+CONTENT\s*=\s*(\{.*\})\s*;\s*$",
    re.DOTALL,
)
STICKY_PROGRESS_STYLE = re.compile(
    r"\.progress-panel\s*\{[^}]*\bposition\s*:\s*sticky\b",
    re.IGNORECASE | re.DOTALL,
)


class _ContractParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.external_resources: list[str] = []
        self.settings: set[str] = set()
        self.has_noscript = False
        self.graph_fallbacks: list[list[str]] = []
        self.has_main = False
        self.has_viewport = False
        self.has_progress_panel = False
        self._inside_style = False
        self._inside_script = False
        self._fallback_depth = 0
        self.style_text: list[str] = []
        self.scripts: list[list[str]] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        if tag == "noscript":
            self.has_noscript = True
        elif tag == "main" and attributes.get("id") == "main-content":
            self.has_main = True
        elif (
            tag == "meta"
            and attributes.get("name", "").lower() == "viewport"
            and attributes.get("content")
        ):
            self.has_viewport = True
        elif tag == "style":
            self._inside_style = True
        elif tag == "script":
            self._inside_script = True
            self.scripts.append([])

        setting = attributes.get("data-setting")
        if setting and self._is_usable_control(tag, attributes):
            self.settings.add(setting)

        classes = set(attributes.get("class", "").split())
        if "progress-panel" in classes:
            self.has_progress_panel = True
        if "graph-fallback" in classes or "data-graph-fallback" in attributes:
            self.graph_fallbacks.append([])
            self._fallback_depth = 1
        elif self._fallback_depth and tag not in VOID_ELEMENTS:
            self._fallback_depth += 1

        resource = self._external_resource(tag, attributes)
        if resource:
            self.external_resources.append(f"<{tag}> {resource}")

        style = attributes.get("style", "")
        if NETWORK_URL.search(style) or re.search(
            r"url\(\s*['\"]?(?:https?:)?//", style, re.IGNORECASE
        ):
            self.external_resources.append(f"<{tag}> inline style")

    def handle_endtag(self, tag: str) -> None:
        if tag == "style":
            self._inside_style = False
        elif tag == "script":
            self._inside_script = False
        if self._fallback_depth:
            self._fallback_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._inside_style:
            self.style_text.append(data)
        if self._inside_script:
            self.scripts[-1].append(data)
        if self._fallback_depth:
            self.graph_fallbacks[-1].append(data)

    @staticmethod
    def _is_usable_control(tag: str, attrs: dict[str, str | None]) -> bool:
        if tag in NATIVE_CONTROLS:
            return "disabled" not in attrs and not (
                tag == "input" and (attrs.get("type") or "").lower() == "hidden"
            )

        role = (attrs.get("role") or "").lower()
        tabindex = attrs.get("tabindex")
        try:
            is_focusable = tabindex is not None and int(tabindex) >= 0
        except ValueError:
            is_focusable = False
        has_keyboard_handler = any(handler in attrs for handler in KEYBOARD_HANDLERS)
        return role in INTERACTIVE_ROLES and is_focusable and has_keyboard_handler

    @staticmethod
    def _external_resource(tag: str, attrs: dict[str, str | None]) -> str | None:
        candidates = [attrs.get("src"), attrs.get("srcset")]
        if tag == "link":
            rel = set((attrs.get("rel") or "").lower().split())
            if "stylesheet" in rel or (
                "preload" in rel and (attrs.get("as") or "").lower() == "font"
            ):
                candidates.append(attrs.get("href"))

        return next((candidate for candidate in candidates if candidate), None)


def _is_content_data_script(script: str) -> bool:
    match = CONTENT_DATA_SCRIPT.fullmatch(script)
    if not match:
        return False
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict)


def validate_html(path: Path) -> list[str]:
    """Return every offline, settings, and accessibility contract violation."""
    try:
        html = path.read_text(encoding="utf-8")
    except OSError as error:
        return [f"cannot read {path}: {error}"]

    parser = _ContractParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception as error:
        return [f"cannot parse HTML: {error}"]

    errors: list[str] = []
    for resource in parser.external_resources:
        errors.append(f"external runtime resource is not portable: {resource}")

    style_text = "\n".join(parser.style_text)
    if re.search(
        r"(?:@import|url\()\s*['\"]?(?:https?:)?//",
        style_text,
        re.IGNORECASE,
    ):
        errors.append("external style or font resource is not portable")

    executable_scripts = (
        "".join(parts) for parts in parser.scripts if not _is_content_data_script("".join(parts))
    )
    if any(
        pattern.search(script) for script in executable_scripts for pattern in NETWORK_CAPABILITIES
    ):
        errors.append("inline script contains a network-capable runtime operation")

    for setting in sorted(SETTINGS - parser.settings):
        errors.append(f"missing settings control: {setting}")
    if not parser.has_noscript:
        errors.append("missing noscript static-content fallback")
    if "prefers-reduced-motion" not in style_text.lower():
        errors.append("missing prefers-reduced-motion style")
    if not VISIBLE_FOCUS.search(style_text):
        errors.append("missing visible :focus-visible rule")
    if not parser.graph_fallbacks or any(
        not "".join(fallback).strip() for fallback in parser.graph_fallbacks
    ):
        errors.append("missing readable graph fallback")
    if not parser.has_main:
        errors.append("missing main-content landmark")
    if not parser.has_viewport:
        errors.append("missing viewport metadata")
    if not parser.has_progress_panel:
        errors.append("missing progress panel")
    if not STICKY_PROGRESS_STYLE.search(style_text):
        errors.append("missing sticky progress style")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate an offline interactive lecture review site."
    )
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    errors = validate_html(args.path)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"VALID: {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
