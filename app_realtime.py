from pathlib import Path
import html, joblib, streamlit as st
from dotenv import load_dotenv
from realtime.news_fetcher import search_live_news
from realtime.article_extractor import extract_article
from realtime.fact_checker import search_fact_checks

ROOT=Path(__file__).resolve().parent
MODEL_PATH=ROOT/"model"/"fake_news_pipeline.joblib"
load_dotenv()
st.set_page_config(page_title="Real-Time News Verification",page_icon="📰",layout="wide")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

def predict(text):
    label=int(model.predict([text])[0])
    return ("Likely Real (model label)" if label==1 else "Likely Fake (model label)",
            float(model.decision_function([text])[0]) if hasattr(model,"decision_function") else None)

def fact_checks(query):
    try:
        result=search_fact_checks(query)
        st.info(result["message"])
        for item in result["claims"]:
            with st.container(border=True):
                st.markdown("**Claim:** "+item["claim"])
                st.write("**Publisher:**",item["publisher"])
                st.write("**Fact-check rating:**",item["rating"])
                if item["date"]: st.caption("Review date: "+item["date"])
                if item["title"]: st.write(item["title"])
                if item["url"]: st.markdown("[Open fact-check review]("+item["url"]+")")
    except Exception as e: st.error("Fact-check lookup failed: "+str(e))

model=load_model()
st.title("📰 Real-Time News Detection & Verification")
st.caption("Phase 1: RSS news search + article extraction + existing ML model + Google Fact Check search")
if model is None:
    st.error(f"Existing model not found: {MODEL_PATH}. Keep your trained model at this exact path.")
    st.stop()
st.warning("The ML model identifies patterns learned from historical ISOT data; it does not prove truth. Google Fact Check searches existing published reviews. No match means inconclusive.")
t1,t2,t3=st.tabs(["🔎 Live News Search","🌐 Check News URL","✍️ Check Article Text"])

with t1:
    q=st.text_input("Search current news by topic, headline or claim",placeholder="Example: India space mission latest")
    n=st.slider("Number of results",3,15,8)
    if st.button("Search Live News",type="primary"):
        try: st.session_state["rss_results"]=search_live_news(q,n)
        except Exception as e: st.error("RSS search failed: "+str(e))
    for i,item in enumerate(st.session_state.get("rss_results",[])):
        with st.container(border=True):
            st.markdown(f"**[{item['title']}]({item['url']})**")
            st.caption(item["published"]+" · "+item["source"])
            if item["summary"]: st.write(html.unescape(item["summary"])[:450])
            if st.button("Find fact-checks for headline",key=f"rss_fc_{i}"): fact_checks(item["title"])

with t2:
    url=st.text_input("Paste a public news article URL")
    if st.button("Extract and Analyze URL",type="primary"):
        try: st.session_state["extracted_article"]=extract_article(url)
        except Exception as e: st.error("Article extraction failed: "+str(e))
    article=st.session_state.get("extracted_article")
    if article:
        st.subheader(article["title"] or "Extracted article")
        st.caption(" · ".join(x for x in [article["sitename"],article["date"],article["author"]] if x))
        st.markdown("[Open original article]("+article["url"]+")")
        with st.expander("View extracted text"): st.write(article["text"])
        label,score=predict(article["title"]+" "+article["text"])
        st.metric("Existing ML model output",label)
        if score is not None: st.caption(f"SVM decision score: {score:.3f} (not a probability)")
        st.subheader("Google Fact Check lookup")
        fact_checks(article["title"] or article["text"][:300])

with t3:
    text=st.text_area("Paste a news headline, claim or article",height=220)
    if st.button("Analyze Text",type="primary"):
        if len(text.split())<5: st.warning("Enter at least five words.")
        else:
            label,score=predict(text); st.metric("Existing ML model output",label)
            if score is not None: st.caption(f"SVM decision score: {score:.3f} (not a probability)")
            fact_checks(text[:500])

st.divider()
st.caption("RSS headlines are not verified. Fact-check ratings belong to the cited review publishers. Always inspect sources and dates.")
