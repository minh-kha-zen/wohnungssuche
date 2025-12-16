import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from openai import OpenAI


# ---- Local dev convenience: load .env if present (Streamlit Cloud will use secrets) ----
load_dotenv()


# ---- Constants (copied from main.py) ----
APPLICANT_INFO = """
Wir sind Nils (26) und Minh-Kha (30). Gemeinsam verfügen wir über ein gesichertes 
monatliches Nettoeinkommen von ca. 9.000 € – beide ruhige, zuverlässige, Nichtraucher 
ohne Haustiere. Nils arbeitet als Unternehmensberater bei Bain, Minh-Kha startet im 
Februar bei Celonis und zieht dafür aus Zürich nach München.
""".strip()

WEBSITE_INFO = """
Um Ihnen einen schnellen Überblick zu ermöglichen, haben wir eine kleine Website mit allen Unterlagen zu uns eingerichtet:
https://nils-und-minhkha.lovable.app

Passwort für den Dokumentenbereich: NiMi!2026
""".strip()

IDEAL_MOVE_IN = "Februar 2026"


def _get_secret(name: str) -> str | None:
    """
    Reads a secret from env vars (Streamlit Cloud supports these) and falls back to st.secrets.
    This lets local dev use .env and Cloud deployments use the Secrets UI.
    """
    val = os.getenv(name)
    if val:
        return val
    try:
        # st.secrets acts like a dict; missing key raises.
        v = st.secrets.get(name)  # type: ignore[attr-defined]
        return str(v) if v else None
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _get_clients() -> tuple[FirecrawlApp, OpenAI]:
    firecrawl_api_key = _get_secret("FIRECRAWL_API_KEY")
    openai_api_key = _get_secret("OPENAI_API_KEY")
    if not firecrawl_api_key or not openai_api_key:
        raise ValueError(
            "Missing required secrets: FIRECRAWL_API_KEY and/or OPENAI_API_KEY. "
            "Set them as environment variables or via Streamlit Secrets."
        )
    return FirecrawlApp(api_key=firecrawl_api_key), OpenAI(api_key=openai_api_key)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def extract_listing_info(url: str) -> dict[str, Any]:
    """
    Scrape an immobilienscout24 listing and extract structured information.
    Cached to avoid repeated scraping/LLM calls for the same URL.
    """
    firecrawl, openai_client = _get_clients()

    result = firecrawl.scrape_url(
        url,
        params={
            "formats": ["markdown"],
        },
    )

    markdown_content = result.get("markdown", "")
    if not markdown_content:
        raise ValueError("Could not extract content from the listing (no markdown returned).")

    extraction_prompt = """
Analyze this immobilienscout24 listing and extract the following information in a structured format:

- Title/Address
- Rent (Kaltmiete and Warmmiete if available)
- Size (square meters)
- Rooms
- Floor/Etage
- Available from (Bezugsfrei ab)
- Deposit (Kaution)
- Location/District
- Key features and amenities
- Landlord/Contact info if available
- Any requirements mentioned (income proof, etc.)

Listing content:
""".strip()

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are an expert at analyzing German real estate listings. Extract key information accurately and concisely in German.",
            },
            {
                "role": "user",
                "content": extraction_prompt + "\n\n" + markdown_content[:15000],
            },
        ],
        temperature=0.3,
    )

    extracted_info = (response.choices[0].message.content or "").strip()
    if not extracted_info:
        raise ValueError("Extraction returned empty content.")

    return {
        "url": url,
        "raw_markdown": markdown_content,
        "extracted_info": extracted_info,
    }


def generate_reply(listing_info: dict[str, Any], max_words: int = 150, include_website: bool = True) -> str:
    firecrawl, openai_client = _get_clients()
    _ = firecrawl  # unused here, but kept to ensure secrets validation is consistent

    website_section = WEBSITE_INFO if include_website else ""

    reply_prompt = f"""
Based on the following apartment listing, write a short application message in German.

STRICT STRUCTURE (follow this order):
1. Salutation (e.g., "Guten Tag," or "Guten Morgen,")
2. Why we like this specific listing (1-2 sentences, mention specific features from the listing)
3. About us (use the provided info, keep it brief)
4. Administrative stuff (moving-in date handling - see rules below)
5. {"Our website paragraph (include exactly as provided)" if include_website else "Skip this section"}
6. Closing (express hope to hear back and see the apartment, sign off with "Viele Grüße, Nils & Minh-Kha")

MOVING-IN DATE RULES (very important):
- Our ideal move-in date is {IDEAL_MOVE_IN}
- If the listing's move-in date is EARLIER than {IDEAL_MOVE_IN}: mention that we are flexible regarding the earlier date
- If the listing's move-in date is LATER than {IDEAL_MOVE_IN}: mention that the stated date works perfectly for us
- If there is NO move-in date mentioned: mention that ideally we would move in around {IDEAL_MOVE_IN}

ABOUT US (use this info):
{APPLICANT_INFO}

{f"WEBSITE SECTION (include this exactly if the listing mentions required documents like SCHUFA, income proof, etc.):{os.linesep}{website_section}" if include_website else ""}

STYLE GUIDELINES:
- Maximum {max_words} words total
- Warm but professional tone
- No subject line needed
- Don't be overly formal, keep it natural
- Reference specific features from the listing to show genuine interest

LISTING INFORMATION:
{listing_info["extracted_info"]}

Now write the message:
""".strip()

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You are an expert at writing German apartment application letters.
Write concise, warm, and professional messages. Follow the structure exactly as specified.
Always sign off as "Nils & Minh-Kha". Never use "Sie-Form" excessively - keep it natural.""",
            },
            {"role": "user", "content": reply_prompt},
        ],
        temperature=0.7,
    )

    reply = (response.choices[0].message.content or "").strip()
    if not reply:
        raise ValueError("Reply generation returned empty content.")
    return reply


# ---- Streamlit UI ----
st.set_page_config(page_title="Wohnungssuche Agent", page_icon="🏠", layout="centered")

st.title("🏠 Wohnungssuche Agent")
st.write("Füge einen Immobilienscout24-Link ein, setze die Parameter, und erhalte eine passende Nachricht als Text.")

with st.sidebar:
    st.subheader("Einstellungen")
    max_words = st.slider("Max. Wörter", min_value=60, max_value=250, value=150, step=10)
    include_website = st.checkbox("Website/Unterlagen-Abschnitt zulassen", value=True)
    show_extracted = st.checkbox("Extrahierte Listing-Infos anzeigen", value=False)
    st.divider()
    st.caption("Secrets erwartet: `OPENAI_API_KEY` und `FIRECRAWL_API_KEY` (Environment oder Streamlit Secrets).")

url = st.text_input("Immobilienscout24 URL", placeholder="https://www.immobilienscout24.de/expose/123456789")

col1, col2 = st.columns([1, 1])
with col1:
    run = st.button("Nachricht generieren", type="primary", use_container_width=True)
with col2:
    clear = st.button("Zurücksetzen", use_container_width=True)

if clear:
    st.session_state.pop("last_result", None)
    st.session_state.pop("last_error", None)

missing = [k for k in ("OPENAI_API_KEY", "FIRECRAWL_API_KEY") if not _get_secret(k)]
if missing:
    st.warning(
        "Fehlende Secrets: " + ", ".join(missing) + ". "
        "Lege sie in Streamlit Cloud unter **App settings → Secrets** an (oder lokal per `.env`)."
    )

if run:
    st.session_state.pop("last_error", None)
    if not url or not url.strip():
        st.session_state["last_error"] = "Bitte eine URL eingeben."
    else:
        url = url.strip()
        try:
            with st.spinner("Scrape & Extraktion läuft…"):
                listing_info = extract_listing_info(url)
            with st.spinner("Nachricht wird geschrieben…"):
                reply = generate_reply(listing_info, max_words=max_words, include_website=include_website)
            st.session_state["last_result"] = {
                "url": url,
                "listing_info": listing_info,
                "reply": reply,
            }
        except Exception as e:
            st.session_state["last_error"] = str(e)

if err := st.session_state.get("last_error"):
    st.error(err)

result = st.session_state.get("last_result")
if result:
    if show_extracted:
        with st.expander("Extrahierte Listing-Informationen", expanded=False):
            st.text(result["listing_info"]["extracted_info"])

    st.subheader("Generierte Nachricht")
    st.text_area("Zum Kopieren", value=result["reply"], height=260)
