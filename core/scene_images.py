"""Scene image download: AI + free stock + optional user URLs.

Recommendations (legal / reliable):
  ✅ Pollinations AI — free text→image (best for original cartoon scenes)
  ✅ Wikimedia Commons — free educational/stock illustrations
  ✅ Pexels / Unsplash — high quality photos (set API key in env)
  ✅ Direct image URLs you own or have rights to use
  ❌ Do NOT scrape Instagram / Facebook / TikTok / YouTube thumbnails
     (ToS + copyright; broken URLs; no commercial rights)
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

# UI labels → internal source codes
IMAGE_SOURCE_CHOICES = {
    "AI cartoon (recommended)": "ai",
    "AI + stock mix (better variety)": "mix",
    "Stock photos (Wikimedia / Pexels)": "stock",
    "My image links only": "urls",
}

USER_AGENT = "video-to-khmer-kids/1.4 (scene images; educational)"


def _http_get(url: str, *, timeout: int = 60, headers: dict | None = None) -> bytes | None:
    hdrs = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    if headers:
        hdrs.update(headers)
    try:
        req = urllib.request.Request(url, headers=hdrs)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        return data if data else None
    except Exception:
        return None


def _save_jpeg(data: bytes, out_path: Path, width: int, height: int) -> bool:
    try:
        if not data or len(data) < 4000:
            return False
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img = img.resize((width, height), Image.Resampling.LANCZOS)
        img = ImageEnhance.Contrast(img).enhance(1.10)
        img = ImageEnhance.Color(img).enhance(1.12)
        img = ImageEnhance.Sharpness(img).enhance(1.08)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, "JPEG", quality=94)
        return out_path.stat().st_size > 10000
    except Exception:
        return False


def _fallback_gradient(out_path: Path, width: int, height: int, seed: str, style: str) -> Path:
    h = int(hashlib.md5(seed.encode("utf-8")).hexdigest()[:8], 16)
    palettes = {
        "kids": [(255, 200, 80), (120, 210, 255), (255, 140, 180), (100, 220, 140)],
        "storybook": [(255, 236, 210), (255, 180, 140), (120, 170, 200)],
        "cinematic": [(20, 24, 48), (80, 40, 60), (30, 60, 90)],
    }
    colors = palettes.get(style, palettes["kids"])
    c0 = colors[h % len(colors)]
    c1 = colors[(h // 7) % len(colors)]
    c2 = colors[(h // 13) % len(colors)]
    img = Image.new("RGB", (width, height), c0)
    px = img.load()
    for y in range(height):
        t = y / max(1, height - 1)
        r = int(c0[0] * (1 - t) + c1[0] * t)
        g = int(c0[1] * (1 - t) + c1[1] * t)
        b = int(c0[2] * (1 - t) + c1[2] * t)
        for x in range(0, width, 2):
            u = x / max(1, width - 1)
            px[x, y] = (
                int(r * (1 - u) + c2[0] * u),
                int(g * (1 - u) + c2[1] * u),
                int(b * (1 - u) + c2[2] * u),
            )
            if x + 1 < width:
                px[x + 1, y] = px[x, y]
    img = img.filter(ImageFilter.GaussianBlur(radius=1.0))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "JPEG", quality=90)
    return out_path


def boost_kids_prompt(prompt: str) -> str:
    """Stronger preschool cartoon lock so AI images look more nursery-like."""
    p = re.sub(r"\s+", " ", (prompt or "").strip())
    if not p:
        p = "happy toddlers singing nursery rhyme"
    lock = (
        "bright colorful 3D preschool cartoon animation still, cute toddler JJ-style "
        "big round eyes soft cheeks, soft claymation-like 3D, highly saturated candy colors, "
        "sunny wholesome nursery rhyme scene, Pixar-like kids film lighting, "
        "family-friendly, sharp focus, no text, no watermark, no logo, no letters"
    )
    # Keep subject first, style second
    core = p.split(", no text")[0][:220]
    return f"{core}, {lock}"[:380]


def stock_search_query(prompt: str, style: str = "kids") -> str:
    """Short search string for stock / Commons."""
    p = re.sub(r"[^\w\s'-]", " ", (prompt or "").lower())
    stop = {
        "the", "and", "with", "from", "that", "this", "scene", "illustrating",
        "cinematic", "photorealistic", "sharp", "focus", "watermark", "text",
        "letters", "logo", "no", "a", "an", "of", "in", "on", "for", "to",
    }
    words = [w for w in p.split() if len(w) > 2 and w not in stop][:6]
    if style == "kids":
        words = (words + ["children", "illustration", "cartoon", "nursery"])[:7]
    return " ".join(words) or "children nursery rhyme illustration"


def parse_image_urls(text: str) -> list[str]:
    """Extract http(s) image URLs from a multiline / comma-separated paste."""
    raw = text or ""
    found = re.findall(r"https?://[^\s<>\"']+", raw, flags=re.I)
    out: list[str] = []
    for u in found:
        u = u.rstrip(").,];}'\"")
        low = u.lower()
        # Skip obvious non-image social page links (user should paste direct CDN)
        if any(x in low for x in ("instagram.com/p/", "tiktok.com/", "facebook.com/", "fb.watch")):
            continue
        out.append(u)
    # de-dupe preserve order
    seen: set[str] = set()
    uniq: list[str] = []
    for u in out:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


def fetch_from_url(url: str, out_path: Path, width: int, height: int, timeout: int = 45) -> bool:
    data = _http_get(url, timeout=timeout)
    return _save_jpeg(data or b"", out_path, width, height)


def fetch_pollinations_ai(
    prompt: str,
    out_path: Path,
    *,
    width: int = 1280,
    height: int = 720,
    seed: int = 1,
    style: str = "kids",
    timeout: int = 90,
) -> bool:
    """Try new gen.pollinations.ai then legacy image.pollinations.ai."""
    prompt_use = boost_kids_prompt(prompt) if style == "kids" else re.sub(r"\s+", " ", prompt)[:360]
    encoded = urllib.parse.quote(prompt_use)
    api_key = (os.environ.get("POLLINATIONS_API_KEY") or "").strip()

    urls = [
        (
            f"https://gen.pollinations.ai/image/{encoded}"
            f"?model=flux&width={width}&height={height}&seed={seed}&nologo=true"
        ),
        (
            f"https://image.pollinations.ai/prompt/{encoded}"
            f"?width={width}&height={height}&seed={seed}&nologo=true&enhance=false&model=flux"
        ),
        # Second seed / slight enhance for kids only
        (
            f"https://image.pollinations.ai/prompt/{urllib.parse.quote(prompt_use[:280])}"
            f"?width={width}&height={height}&seed={seed + 17}&nologo=true&enhance=true&model=flux"
        ),
    ]
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    for url in urls:
        data = _http_get(url, timeout=timeout, headers=headers or None)
        if _save_jpeg(data or b"", out_path, width, height):
            return True
    return False


def fetch_wikimedia(
    query: str,
    out_path: Path,
    *,
    width: int = 1280,
    height: int = 720,
    seed: int = 0,
    timeout: int = 45,
) -> bool:
    """Free illustration/photo from Wikimedia Commons search."""
    q = (query or "children illustration").strip()[:120]
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrnamespace": "6",
            "gsrlimit": "12",
            "gsrsearch": q,
            "prop": "imageinfo",
            "iiprop": "url|mime",
            "iiurlwidth": str(max(640, min(width, 1600))),
        }
    )
    api = f"https://commons.wikimedia.org/w/api.php?{params}"
    data = _http_get(api, timeout=timeout)
    if not data:
        return False
    try:
        payload = json.loads(data.decode("utf-8", errors="ignore"))
        pages = (payload.get("query") or {}).get("pages") or {}
        items = list(pages.values())
        if not items:
            return False
        # Pick by seed for variety across scenes
        items.sort(key=lambda p: p.get("pageid", 0))
        pick = items[seed % len(items)]
        info = (pick.get("imageinfo") or [{}])[0]
        mime = (info.get("mime") or "").lower()
        if mime and not mime.startswith("image/"):
            # try next
            for i, p in enumerate(items):
                info = (p.get("imageinfo") or [{}])[0]
                mime = (info.get("mime") or "").lower()
                if mime.startswith("image/"):
                    pick = p
                    break
            else:
                return False
            info = (pick.get("imageinfo") or [{}])[0]
        url = info.get("thumburl") or info.get("url")
        if not url:
            return False
        return fetch_from_url(url, out_path, width, height, timeout=timeout)
    except Exception:
        return False


def fetch_pexels(
    query: str,
    out_path: Path,
    *,
    width: int = 1280,
    height: int = 720,
    seed: int = 0,
    timeout: int = 45,
) -> bool:
    """Pexels stock (needs PEXELS_API_KEY). Free key: https://www.pexels.com/api/"""
    key = (os.environ.get("PEXELS_API_KEY") or "").strip()
    if not key:
        return False
    q = urllib.parse.quote((query or "kids")[:80])
    api = f"https://api.pexels.com/v1/search?query={q}&per_page=15&orientation=landscape"
    data = _http_get(api, timeout=timeout, headers={"Authorization": key})
    if not data:
        return False
    try:
        payload = json.loads(data.decode("utf-8", errors="ignore"))
        photos = payload.get("photos") or []
        if not photos:
            return False
        photo = photos[seed % len(photos)]
        src = (photo.get("src") or {})
        url = src.get("large2x") or src.get("large") or src.get("original")
        if not url:
            return False
        return fetch_from_url(url, out_path, width, height, timeout=timeout)
    except Exception:
        return False


def fetch_unsplash(
    query: str,
    out_path: Path,
    *,
    width: int = 1280,
    height: int = 720,
    seed: int = 0,
    timeout: int = 45,
) -> bool:
    """Unsplash stock (needs UNSPLASH_ACCESS_KEY). Free: https://unsplash.com/developers"""
    key = (os.environ.get("UNSPLASH_ACCESS_KEY") or "").strip()
    if not key:
        return False
    q = urllib.parse.quote((query or "children")[:80])
    api = f"https://api.unsplash.com/search/photos?query={q}&per_page=15&orientation=landscape"
    data = _http_get(
        api,
        timeout=timeout,
        headers={"Authorization": f"Client-ID {key}"},
    )
    if not data:
        return False
    try:
        payload = json.loads(data.decode("utf-8", errors="ignore"))
        results = payload.get("results") or []
        if not results:
            return False
        photo = results[seed % len(results)]
        urls = photo.get("urls") or {}
        url = urls.get("regular") or urls.get("full") or urls.get("small")
        if not url:
            return False
        return fetch_from_url(url, out_path, width, height, timeout=timeout)
    except Exception:
        return False


def fetch_scene_image(
    prompt: str,
    out_path: Path,
    *,
    width: int = 1280,
    height: int = 720,
    style: str = "kids",
    seed: int | None = None,
    source: str = "ai",
    ref_urls: list[str] | None = None,
    timeout: int = 90,
) -> Path:
    """
    Download one scene image.

    source: ai | mix | stock | urls
    ref_urls: optional direct image links (cycled by seed)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base_seed = (
        int(seed)
        if seed is not None
        else int(hashlib.md5((prompt or "x").encode("utf-8")).hexdigest()[:8], 16) % 100000
    )
    src = (source or "ai").strip().lower()
    if src not in ("ai", "mix", "stock", "urls"):
        src = "ai"
    refs = [u for u in (ref_urls or []) if u]

    # 1) User-provided URLs (always tried first if present)
    if refs and src in ("urls", "mix", "ai", "stock"):
        url = refs[base_seed % len(refs)]
        if fetch_from_url(url, out_path, width, height, timeout=min(45, timeout)):
            return out_path
        if src == "urls":
            # try all refs once
            for u in refs:
                if fetch_from_url(u, out_path, width, height, timeout=min(45, timeout)):
                    return out_path

    if src == "urls":
        return _fallback_gradient(out_path, width, height, prompt or "kids", style)

    query = stock_search_query(prompt, style=style)

    # 2) AI path
    if src in ("ai", "mix"):
        if fetch_pollinations_ai(
            prompt,
            out_path,
            width=width,
            height=height,
            seed=base_seed,
            style=style,
            timeout=timeout,
        ):
            return out_path

    # 3) Stock path
    if src in ("stock", "mix", "ai"):
        # Prefer Pexels/Unsplash when keys exist, then Wikimedia
        if fetch_pexels(query, out_path, width=width, height=height, seed=base_seed, timeout=45):
            return out_path
        if fetch_unsplash(query, out_path, width=width, height=height, seed=base_seed, timeout=45):
            return out_path
        if fetch_wikimedia(query, out_path, width=width, height=height, seed=base_seed, timeout=45):
            return out_path
        # Broader Commons query for kids
        if style == "kids" and fetch_wikimedia(
            "children cartoon illustration nursery",
            out_path,
            width=width,
            height=height,
            seed=base_seed + 3,
            timeout=45,
        ):
            return out_path

    # 4) Last AI retry with boosted kids prompt
    if src != "stock" and fetch_pollinations_ai(
        boost_kids_prompt(prompt),
        out_path,
        width=width,
        height=height,
        seed=base_seed + 99,
        style="kids",
        timeout=timeout,
    ):
        return out_path

    return _fallback_gradient(out_path, width, height, prompt or "kids", style)
