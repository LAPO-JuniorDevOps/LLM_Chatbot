import requests
from typing import List
from bs4 import BeautifulSoup  # Make sure to import BeautifulSoup
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, WebDriverException
import time



class ScraperUtils:

    @staticmethod
    def extract_text_from_url(url: str, url_tags: List[str]):
        try:
            response = requests.get(url, timeout=5)
            soup = BeautifulSoup(response.text, "html.parser")
            return "\n".join([tag.get_text() for tag in soup.find_all(lambda tag: tag.name in url_tags)])
        except Exception as e:
            return f"Error fetching {url}: {e}"

    @staticmethod
    def split_text(text: str, chunk_size: int = 500, overlap: int = 50):
        """Split text into manageable chunks."""
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]

    @staticmethod
    def summarize_text(text: str, sentence_count: int = 5):
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, sentence_count)
        return " ".join(str(sentence) for sentence in summary)
    
    @staticmethod
    def scrape_and_filter_relevant_content(embedder, url, query, snippet, similarity_threshold=0.6):
        """
        Scrape using Selenium and filter paragraphs based on similarity to the query.
        """
        query_embedding = embedder.encode([query])
        def create_chunks(paragraphs):
            relevant_chunks = []
            # Split into paragraphs
            paragraph_embeddings = embedder.encode(paragraphs)

            # Compute similarity
            similarities = cosine_similarity(query_embedding, paragraph_embeddings)[0]

            for i, score in enumerate(similarities):
                if score >= similarity_threshold:
                    relevant_chunks.append(paragraphs[i])
            return relevant_chunks
        
        headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Referer": "https://www.google.com/",
    "DNT": "1",
    "Connection": "keep-alive"
}

        options = Options()
        options.add_argument('--headless')  # Run in background
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')  # Needed if running on some Linux environments
        options.add_argument('--ignore-certificate-errors')
        options.add_argument("--window-size=1920,1080")

        try:
            driver = webdriver.Chrome(options=options)
            driver.set_page_load_timeout(10)
            driver.get(url)

            # Wait briefly for content to load
            time.sleep(2)  # You can improve this with WebDriverWait if needed

            # Get full page source (rendered HTML)
            page_source = driver.page_source
            driver.quit()

            # Basic HTML parsing
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(page_source, 'html.parser')
            text = soup.get_text(separator='\n')

            paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 40]
            if not paragraphs:
                print(f"⚠️ Content too short or empty for {url}")
                return
            
            relevant_chunks = create_chunks(paragraphs)

        except (TimeoutException, WebDriverException) as e:
            print(f"❌ Error scraping {url}: {e}")
            return ScraperUtils.summarize_text(snippet)
        except Exception as e:
            print(f"❌ Unexpected error scraping {url}: {e}")
            return ScraperUtils.summarize_text(snippet)

        return "\n\n".join(relevant_chunks[:10])