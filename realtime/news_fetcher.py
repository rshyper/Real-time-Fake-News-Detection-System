from urllib.parse import quote
import feedparser, requests

def search_live_news(query, limit=10):
    if not query or not query.strip():
        return []
    url = "https://news.google.com/rss/search?q={}&hl=en-IN&gl=IN&ceid=IN:en".format(quote(query.strip()))
    r = requests.get(url, timeout=20, headers={"User-Agent":"Mozilla/5.0"})
    r.raise_for_status()
    feed = feedparser.parse(r.content)
    out = []
    for e in feed.entries[:limit]:
        src = getattr(e, "source", {}) or {}
        out.append({"title":getattr(e,"title","Untitled"), "url":getattr(e,"link",""),
                    "published":getattr(e,"published","Date unavailable"),
                    "summary":getattr(e,"summary",""),
                    "source":src.get("title","Source shown in article link")})
    return out
