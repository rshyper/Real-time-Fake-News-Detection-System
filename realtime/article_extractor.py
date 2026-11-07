from urllib.parse import urlparse
import json, trafilatura

def extract_article(url):
    p = urlparse((url or "").strip())
    if p.scheme not in ("http","https") or not p.netloc or p.username or p.password:
        raise ValueError("Enter a valid public http:// or https:// article URL.")
    safe_url = p.geturl()
    downloaded = trafilatura.fetch_url(safe_url)
    if not downloaded:
        raise RuntimeError("Could not download this page. It may block automated access or require JavaScript.")
    raw = trafilatura.extract(downloaded, url=safe_url, output_format="json",
                              with_metadata=True, include_comments=False, include_tables=False)
    if not raw:
        raise RuntimeError("Could not extract readable article text.")
    data = json.loads(raw)
    text = (data.get("text") or "").strip()
    if len(text.split()) < 40:
        raise RuntimeError("Extracted text is too short. Try another URL or paste article text manually.")
    return {"url":safe_url, "title":data.get("title") or "", "author":data.get("author") or "",
            "date":data.get("date") or "", "sitename":data.get("sitename") or "", "text":text}
