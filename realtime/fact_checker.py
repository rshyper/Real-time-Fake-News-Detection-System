import os, requests
from dotenv import load_dotenv
load_dotenv()
ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

def search_fact_checks(query, page_size=5, language_code="en"):
    key = os.getenv("GOOGLE_FACT_CHECK_API_KEY","").strip()
    if not key:
        return {"configured":False, "message":"Google Fact Check API key is not configured in .env.", "claims":[]}
    if not query or not query.strip():
        return {"configured":True, "message":"Enter a claim or headline.", "claims":[]}
    params = {"key":key, "query":query.strip()[:500], "languageCode":language_code,
              "pageSize":max(1,min(int(page_size),10))}
    r = requests.get(ENDPOINT, params=params, timeout=20)
    if r.status_code == 403:
        raise RuntimeError("Google API access denied. Check API enablement, key restrictions and quota.")
    if r.status_code == 400:
        raise RuntimeError("Google rejected the request. Check the API key and API configuration.")
    r.raise_for_status()
    results = []
    for claim in r.json().get("claims",[]):
        for review in claim.get("claimReview",[]):
            publisher = review.get("publisher",{}) or {}
            results.append({"claim":claim.get("text",""), "publisher":publisher.get("name","Unknown"),
                "title":review.get("title",""), "rating":review.get("textualRating","Not supplied"),
                "date":review.get("reviewDate",""), "url":review.get("url","")})
    return {"configured":True, "message":"Matching published reviews found." if results else
            "No matching review returned. This does not mean the claim is true.", "claims":results}
