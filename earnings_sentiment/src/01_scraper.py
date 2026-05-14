import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time

HEADERS = {"User-Agent": "student research project augustin@email.com"}

def get_transcript(ticker: str, url: str) -> dict:
    r = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(r.text, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return {"ticker": ticker, "url": url, "text": text[:8000]}

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    # Direct URLs of confirmed earnings call transcripts on SEC EDGAR
    transcripts = [
        ("AAPL", "https://www.fool.com/earnings/call-transcripts/2024/10/31/apple-aapl-q4-2024-earnings-call-transcript/"),
        ("MSFT", "https://www.fool.com/earnings/call-transcripts/2024/10/30/microsoft-msft-q1-2025-earnings-call-transcript/"),
        ("NVDA", "https://www.fool.com/earnings/call-transcripts/2024/11/20/nvidia-nvda-q3-2025-earnings-call-transcript/"),
        ("GOOGL", "https://www.fool.com/earnings/call-transcripts/2025/02/05/alphabet-goog-q4-2024-earnings-call-transcript/"),
        ("META", "https://www.fool.com/earnings/call-transcripts/2024/10/30/meta-platforms-meta-q3-2024-earnings-call-transcri/"),
    ]

    records = []
    for ticker, url in transcripts:
        print(f"Fetching {ticker}...")
        try:
            r = get_transcript(ticker, url)
            if len(r["text"]) > 500:
                records.append(r)
                print(f"  Got {len(r['text'])} chars")
            time.sleep(2)
        except Exception as e:
            print(f"  Error: {e}")

    df = pd.DataFrame(records)
    print(f"\nCollected {len(df)} transcripts")
    df.to_csv("data/transcripts_raw.csv", index=False)