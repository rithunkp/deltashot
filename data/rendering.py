from __future__ import annotations

from pathlib import Path


class HtmlRenderer:
    """Owns one headless Chromium instance for a batch of static page renders."""

    def __init__(self, width: int = 512, height: int = 512) -> None:
        self.width = width
        self.height = height
        self._playwright = None
        self._browser = None

    async def __aenter__(self) -> HtmlRenderer:
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch()
        return self

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self._browser is not None:
            await self._browser.close()
        if self._playwright is not None:
            await self._playwright.stop()

    async def render(self, html: str, output_path: Path) -> None:
        if self._browser is None:
            raise RuntimeError("HtmlRenderer must be entered before rendering.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        page = await self._browser.new_page(viewport={"width": self.width, "height": self.height}, device_scale_factor=1)
        try:
            async def block_nondeterministic_assets(route: object) -> None:
                request = route.request
                if request.resource_type in {"font", "image", "media"}:
                    await route.abort()
                else:
                    await route.continue_()

            await page.route("**/*", block_nondeterministic_assets)
            await page.set_content(html, wait_until="domcontentloaded", timeout=30_000)
            await page.wait_for_timeout(500)
            await page.screenshot(path=str(output_path), full_page=False)
        finally:
            await page.close()


async def render_html_to_png(html: str, output_path: Path, width: int = 512, height: int = 512) -> None:
    """Render one HTML document to a screenshot with Playwright."""
    async with HtmlRenderer(width=width, height=height) as renderer:
        await renderer.render(html, output_path)
