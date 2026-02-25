"""Browser automation tool using Playwright/Chromium.

Handles JS-rendered pages and sites that block plain HTTP requests (403 Forbidden).
Requires Playwright + Chromium binary. System libs are auto-resolved via LD_LIBRARY_PATH.
"""

import json
import os
import re

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.web import _normalize, _strip_tags, _validate_url

# Inject system libs extracted to user dir (no sudo required).
_PLAYWRIGHT_DEPS = os.path.join(os.path.expanduser("~"), ".local", "playwright-deps", "usr", "lib")
for _lib_dir in (
    os.path.join(_PLAYWRIGHT_DEPS, "aarch64-linux-gnu"),
    _PLAYWRIGHT_DEPS,
):
    if os.path.isdir(_lib_dir):
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        if _lib_dir not in existing:
            os.environ["LD_LIBRARY_PATH"] = f"{_lib_dir}:{existing}" if existing else _lib_dir

# Realistic desktop UA — avoids many bot-detection triggers.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/132.0.0.0 Safari/537.36"
)
_VIEWPORT = {"width": 1280, "height": 800}
_DEFAULT_MAX_CHARS = 50_000
_DEFAULT_TIMEOUT_MS = 30_000  # 30 s page-load timeout


class WebBrowseTool(Tool):
    """Browse a URL with a real headless browser (Playwright/Chromium).

    Use this instead of web_fetch when:
    - The site requires JavaScript to render content
    - web_fetch returns 403 Forbidden or empty content
    - You need content from automotive marketplaces (Carvana, Cars.com, Edmunds, etc.)
    """

    name = "web_browse"
    description = (
        "Fetch a URL using a real headless browser (bypasses JS requirements and 403 blocks). "
        "Returns page text. Use when web_fetch fails or returns 403."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to visit"},
            "waitUntil": {
                "type": "string",
                "enum": ["domcontentloaded", "networkidle", "load"],
                "description": "When to consider page loaded. Use 'networkidle' for heavy JS sites (slower).",
            },
            "selector": {
                "type": "string",
                "description": "Optional CSS selector — wait for this element before extracting.",
            },
            "maxChars": {
                "type": "integer",
                "minimum": 100,
                "description": "Max characters to return (default 50000).",
            },
        },
        "required": ["url"],
    }

    def __init__(self, max_chars: int = _DEFAULT_MAX_CHARS):
        self.max_chars = max_chars

    async def execute(
        self,
        url: str,
        waitUntil: str = "domcontentloaded",
        selector: str | None = None,
        maxChars: int | None = None,
        **kwargs,
    ) -> str:
        max_chars = maxChars or self.max_chars

        # Validate URL first.
        ok, err = _validate_url(url)
        if not ok:
            return json.dumps({"error": f"URL validation failed: {err}", "url": url})

        try:
            from playwright.async_api import async_playwright, TimeoutError as PWTimeout
        except ImportError:
            return json.dumps({"error": "Playwright not installed. Run: pip install playwright", "url": url})

        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-setuid-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-blink-features=AutomationControlled",
                        "--disable-http2",  # WSL2 HTTP2 sometimes fails
                    ],
                )
                ctx = await browser.new_context(
                    user_agent=_USER_AGENT,
                    viewport=_VIEWPORT,
                    # Mimic a real browser accepting HTML
                    extra_http_headers={
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.9",
                        "Accept-Encoding": "gzip, deflate, br",
                        "Upgrade-Insecure-Requests": "1",
                        "Sec-Fetch-Dest": "document",
                        "Sec-Fetch-Mode": "navigate",
                        "Sec-Fetch-Site": "none",
                    },
                )
                # Hide navigator.webdriver from JS fingerprinting
                await ctx.add_init_script(
                    "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
                )

                page = await ctx.new_page()
                try:
                    resp = await page.goto(
                        url,
                        wait_until=waitUntil,
                        timeout=_DEFAULT_TIMEOUT_MS,
                    )
                    status = resp.status if resp else 0

                    # Optionally wait for a specific element.
                    if selector:
                        await page.wait_for_selector(selector, timeout=10_000)

                    html = await page.content()
                    final_url = page.url

                finally:
                    await ctx.close()
                    await browser.close()

        except Exception as exc:
            return json.dumps({"error": str(exc), "url": url})

        # Convert HTML → readable text.
        text = _html_to_text(html)
        truncated = len(text) > max_chars
        if truncated:
            text = text[:max_chars]

        return json.dumps({
            "url": url,
            "finalUrl": final_url,
            "status": status,
            "extractor": "playwright",
            "truncated": truncated,
            "length": len(text),
            "text": text,
        }, ensure_ascii=False)


def _html_to_text(html: str) -> str:
    """Best-effort HTML → readable text conversion."""
    try:
        from readability import Document
        doc = Document(html)
        title = doc.title() or ""
        body = _to_markdown(doc.summary())
        return f"# {title}\n\n{body}" if title else body
    except Exception:
        return _normalize(_strip_tags(html))


def _to_markdown(html: str) -> str:
    """Convert HTML to plain markdown-ish text."""
    text = re.sub(
        r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>([\s\S]*?)</a>',
        lambda m: f'[{_strip_tags(m[2])}]({m[1]})',
        html, flags=re.I,
    )
    text = re.sub(
        r'<h([1-6])[^>]*>([\s\S]*?)</h\1>',
        lambda m: f'\n{"#" * int(m[1])} {_strip_tags(m[2])}\n',
        text, flags=re.I,
    )
    text = re.sub(r'<li[^>]*>([\s\S]*?)</li>', lambda m: f'\n- {_strip_tags(m[1])}', text, flags=re.I)
    text = re.sub(r'</(p|div|section|article)>', '\n\n', text, flags=re.I)
    text = re.sub(r'<(br|hr)\s*/?>', '\n', text, flags=re.I)
    return _normalize(_strip_tags(text))
