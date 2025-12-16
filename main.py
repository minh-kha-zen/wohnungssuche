#!/usr/bin/env python3
"""
Wohnungssuche Agent - Extracts immobilienscout24 listing info and generates replies.
"""

import argparse
import os
import sys
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize clients
firecrawl = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Our info
APPLICANT_INFO = """
Wir sind Nils (26) und Minh-Kha (30). Gemeinsam verfügen wir über ein gesichertes 
monatliches Nettoeinkommen von ca. 9.000 € – beide ruhige, zuverlässige, Nichtraucher 
ohne Haustiere. Nils arbeitet als Unternehmensberater bei Bain, Minh-Kha startet im 
Februar bei Celonis und zieht dafür aus Zürich nach München.
"""

WEBSITE_INFO = """
Um Ihnen einen schnellen Überblick zu ermöglichen, haben wir eine kleine Website mit allen Unterlagen zu uns eingerichtet:
https://nils-und-minhkha.lovable.app

Passwort für den Dokumentenbereich: NiMi!2026
"""

IDEAL_MOVE_IN = "Februar 2026"


def extract_listing_info(url: str) -> dict:
    """
    Scrape an immobilienscout24 listing and extract structured information.
    """
    print(f"🔍 Scraping listing: {url}")
    
    # Use Firecrawl to scrape the page
    result = firecrawl.scrape_url(
        url,
        params={
            "formats": ["markdown"],
        }
    )
    
    markdown_content = result.get("markdown", "")
    
    if not markdown_content:
        raise ValueError("Could not extract content from the listing")
    
    print("✅ Successfully scraped listing")
    
    # Use OpenAI to extract structured info from the markdown
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
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are an expert at analyzing German real estate listings. Extract key information accurately and concisely in German."
            },
            {
                "role": "user", 
                "content": extraction_prompt + markdown_content[:15000]  # Limit content length
            }
        ],
        temperature=0.3,
    )
    
    extracted_info = response.choices[0].message.content
    print("✅ Successfully extracted listing information")
    
    return {
        "url": url,
        "raw_markdown": markdown_content,
        "extracted_info": extracted_info,
    }


def generate_reply(listing_info: dict, max_words: int = 150, include_website: bool = True) -> str:
    """
    Generate a personalized reply/application message for the listing.
    
    Args:
        listing_info: Extracted listing information
        max_words: Maximum word count for the message (default: 150)
        include_website: Whether to include our website with documents (default: True)
    """
    print("✍️  Generating reply...")
    
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

{f"WEBSITE SECTION (include this exactly if the listing mentions required documents like SCHUFA, income proof, etc.):{website_section}" if include_website else ""}

STYLE GUIDELINES:
- Maximum {max_words} words total
- Warm but professional tone
- No subject line needed
- Don't be overly formal, keep it natural
- Reference specific features from the listing to show genuine interest

LISTING INFORMATION:
{listing_info['extracted_info']}

EXAMPLE MESSAGE FOR REFERENCE (adapt style but don't copy):
---
Guten Morgen,

die Wohnung in der Schwanthalerhöhe hat uns sofort angesprochen. Besonders der großzügige, offene Grundriss, der Balkon sowie die zentrale Lage passen sehr gut zu dem, was wir suchen.

Kurz zu uns:
Wir sind Nils (26) und Minh-Kha (30). Gemeinsam verfügen wir über ein gesichertes monatliches Nettoeinkommen von ca. 9.000 € – beide ruhige, zuverlässige, Nichtraucher ohne Haustiere. Nils arbeitet als Unternehmensberater bei Bain, Minh-Kha startet im Februar bei Celonis und zieht dafür aus Zürich nach München.

Wir würden uns sehr freuen, von Ihnen zu hören und die Wohnung persönlich besichtigen zu dürfen.

Viele Grüße  
Nils & Minh-Kha
---

Now write the message:
"""
    
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You are an expert at writing German apartment application letters. 
Write concise, warm, and professional messages. Follow the structure exactly as specified.
Always sign off as "Nils & Minh-Kha". Never use "Sie-Form" excessively - keep it natural."""
            },
            {
                "role": "user",
                "content": reply_prompt
            }
        ],
        temperature=0.7,
    )
    
    reply = response.choices[0].message.content
    print("✅ Successfully generated reply")
    
    return reply


def process_listing(url: str, max_words: int = 150, include_website: bool = True) -> None:
    """
    Main function to process a listing URL end-to-end.
    
    Args:
        url: The immobilienscout24 listing URL
        max_words: Maximum word count for the reply (default: 150)
        include_website: Whether to include website with documents (default: True)
    """
    print("\n" + "=" * 60)
    print("🏠 WOHNUNGSSUCHE AGENT")
    print("=" * 60 + "\n")
    
    # Step 1: Extract listing info
    listing_info = extract_listing_info(url)
    
    print("\n" + "-" * 60)
    print("📋 EXTRACTED INFORMATION:")
    print("-" * 60)
    print(listing_info["extracted_info"])
    
    # Step 2: Generate reply
    reply = generate_reply(listing_info, max_words=max_words, include_website=include_website)
    
    print("\n" + "-" * 60)
    print("📝 GENERATED REPLY:")
    print("-" * 60)
    print(reply)
    
    print("\n" + "=" * 60)
    print("✅ Done! Copy the reply above and send it to the landlord.")
    print("=" * 60 + "\n")


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Wohnungssuche Agent - Generate apartment application messages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py https://www.immobilienscout24.de/expose/123456789
  python main.py https://www.immobilienscout24.de/expose/123456789 --words 200
  python main.py https://www.immobilienscout24.de/expose/123456789 --no-website
  python main.py https://www.immobilienscout24.de/expose/123456789 --words 100 --no-website
        """
    )
    
    parser.add_argument(
        "url",
        help="The immobilienscout24 listing URL"
    )
    parser.add_argument(
        "--words", "-w",
        type=int,
        default=150,
        help="Maximum word count for the reply (default: 150)"
    )
    parser.add_argument(
        "--no-website",
        action="store_true",
        help="Exclude the website/documents paragraph from the reply"
    )
    
    args = parser.parse_args()
    
    if "immobilienscout24.de" not in args.url:
        print("⚠️  Warning: This doesn't look like an immobilienscout24 URL")
        print("   The script is optimized for immobilienscout24 listings.")
    
    try:
        process_listing(
            url=args.url,
            max_words=args.words,
            include_website=not args.no_website
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

