import os
import json
import logging
import requests
import re
from typing import List, Dict, Set
from abc import ABC, abstractmethod
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import BotUttered
from sentence_transformers import SentenceTransformer, util
import glob
import docx
import fitz
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import torch

# === Config ===
GROQ_API = "ENTER YOUR KEY"
DOCS_FOLDER = r"PATH TO YOU DOCS FOLDER"
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
FAQ_PATH = r"PATH TO YOUR FAQ FOLDER"
SIMILARITY_THRESHOLD = float(os.getenv("FAQ_MATCH_THRESHOLD", 0.85))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "o4-mini"
LLAMA_API_URL = " LLAMA URL"
TEXT_SIMILARITY_THRESHOLD = 0.8  # Threshold for determining text similarity

logger = logging.getLogger(__name__)

# === ScraperUtils ===
class ScraperUtils:
    @staticmethod
    def summarize_text(text: str, sentence_count: int = 5):
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, sentence_count)
        return " ".join(str(sentence) for sentence in summary)

# === Load FAQs ===
def load_faqs(path: str) -> List[Dict[str, str]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load FAQs: {e}")
        return []

# === MMR Diversification ===
def mmr(query_embedding, doc_embeddings, docs, top_k=5, lambda_param=0.7):
    selected = []
    selected_indices = []
    doc_embeddings = [emb for emb in doc_embeddings]
    query_embedding = query_embedding.unsqueeze(0)

    candidates = list(enumerate(doc_embeddings))

    for _ in range(top_k):
        mmr_scores = []
        for idx, candidate_embedding in candidates:
            if idx in selected_indices:
                continue
            relevance = util.pytorch_cos_sim(query_embedding, candidate_embedding.unsqueeze(0)).item()
            diversity = max(
                [util.pytorch_cos_sim(candidate_embedding.unsqueeze(0), doc_embeddings[i].unsqueeze(0)).item()
                 for i in selected_indices] or [0]
            )
            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
            mmr_scores.append((idx, mmr_score))

        if not mmr_scores:
            break
        selected_idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(docs[selected_idx])
        selected_indices.append(selected_idx)

    return selected

# === Document Retriever ===
class CompanyDocRetriever:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.docs = []
        self.embeddings = []
        self.load_documents()

    def load_documents(self):
        files = glob.glob(os.path.join(self.folder_path, "*"))
        for path in files:
            try:
                ext = os.path.splitext(path)[1].lower()
                if ext == ".txt":
                    content = self.read_txt(path)
                    self.process_paragraphs(content.split("\n"))
                elif ext == ".docx":
                    self.process_docx(path)
                elif ext == ".pdf":
                    self.process_pdf(path)
                else:
                    logger.warning(f"Unsupported file format: {path}")
            except Exception as e:
                logger.warning(f"Error loading {path}: {e}")

    def process_paragraphs(self, paragraphs: List[str]):
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        self.docs.extend(paragraphs)
        self.embeddings.extend([
            embedding_model.encode(p, convert_to_tensor=True)
            for p in paragraphs
        ])

    def process_docx(self, path: str):
        try:
            doc = docx.Document(path)
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            self.process_paragraphs(paragraphs)
        except Exception as e:
            logger.warning(f"Error reading DOCX file {path}: {e}")

    def process_pdf(self, path: str):
        try:
            with fitz.open(path) as doc:
                paragraphs = []
                for page in doc:
                    text = page.get_text("blocks")
                    for b in text:
                        block_text = b[4].strip()
                        if block_text:
                            paragraphs.append(block_text)
                self.process_paragraphs(paragraphs)
        except Exception as e:
            logger.warning(f"Error reading PDF file {path}: {e}")

    def read_txt(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Error reading TXT file {path}: {e}")
            return ""

    def filter_by_keywords(self, query: str, chunks: List[str]) -> List[str]:
        stop = set(stopwords.words("english")) - {"mrs", "mr", "dr", "ms"}
        keywords = set(word_tokenize(query.lower())) - stop
        filtered = [chunk for chunk in chunks if keywords & set(word_tokenize(chunk.lower()))]
        return filtered

    def search(self, query: str, top_k=7) -> str:
        if not self.embeddings:
            logger.warning("No company documents are loaded for search.")
            return "Company documents are not available at the moment."

        query_embedding = embedding_model.encode(query, convert_to_tensor=True)
        selected_chunks = mmr(query_embedding, self.embeddings, self.docs, top_k=top_k)
        selected_chunks = self.filter_by_keywords(query, selected_chunks)
        if not selected_chunks:
            return "Sorry, no relevant information was found in the company database."
        summary = ScraperUtils.summarize_text(" ".join(selected_chunks))
        return summary

company_doc_retriever = CompanyDocRetriever(DOCS_FOLDER)

# === Text Utility Functions ===
def similar_text(text1: str, text2: str, threshold=TEXT_SIMILARITY_THRESHOLD) -> bool:
    """Check if two text snippets are semantically similar."""
    if not text1 or not text2:
        return False
    if len(text1) < 10 or len(text2) < 10:  # Skip very short texts
        return text1 == text2
        
    # Use existing embedding model
    emb1 = embedding_model.encode(text1, convert_to_tensor=True)
    emb2 = embedding_model.encode(text2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(emb1, emb2).item()
    return similarity > threshold

def process_response(raw_response: str) -> str:
    """Clean and format the response for better readability."""
    # Remove numbered lists that span multiple paragraphs
    lines = raw_response.split('\n')
    processed_lines = []
    
    # Join consecutive short paragraphs that are part of the same thought
    i = 0
    while i < len(lines):
        if i < len(lines) - 1 and len(lines[i]) < 100 and not lines[i].startswith('*') and not lines[i+1].startswith('*'):
            processed_lines.append(f"{lines[i]} {lines[i+1]}")
            i += 2
        else:
            processed_lines.append(lines[i])
            i += 1
    
    return "\n\n".join([line for line in processed_lines if line.strip()])

def consolidate_response(text: str) -> str:
    """Consolidate fragmented responses into a cohesive whole."""
    # Remove redundant list numbering that spans paragraphs
    lines = text.split('\n')
    consolidated = []
    
    # Remove redundant information and number patterns
    seen_content = set()
    for line in lines:
        # Remove the numbering pattern from the start of lines
        clean_line = re.sub(r'^\d+\.\s*', '', line).strip()
        # Skip if we've seen similar content before
        if clean_line and not any(similar_text(clean_line, seen) for seen in seen_content):
            consolidated.append(line)
            seen_content.add(clean_line)
    
    return '\n'.join(consolidated)

def post_process_response(response: str) -> str:
    """Final cleanup of responses before sending to user."""
    # Fix numbered lists that got split
    list_pattern = re.compile(r'^\d+\.\s', re.MULTILINE)
    if list_pattern.search(response):
        lines = response.split('\n')
        processed_lines = []
        for line in lines:
            if list_pattern.match(line) and processed_lines and not processed_lines[-1].endswith('.'):
                processed_lines[-1] += ' ' + line
            else:
                processed_lines.append(line)
        response = '\n'.join(processed_lines)
    
    # Join short paragraphs that are probably part of the same thought
    paragraphs = response.split('\n\n')
    i = 0
    joined_paragraphs = []
    while i < len(paragraphs):
        if i < len(paragraphs) - 1 and len(paragraphs[i]) < 100 and len(paragraphs[i+1]) < 100:
            joined_paragraphs.append(f"{paragraphs[i]} {paragraphs[i+1]}")
            i += 2
        else:
            joined_paragraphs.append(paragraphs[i])
            i += 1
    
    # Remove duplicate content
    final_paragraphs = []
    seen_paragraphs = set()
    for para in joined_paragraphs:
        cleaned_para = para.strip()
        if cleaned_para and not any(similar_text(cleaned_para, seen) for seen in seen_paragraphs):
            final_paragraphs.append(para)
            seen_paragraphs.add(cleaned_para)
    
    return '\n\n'.join(final_paragraphs)

# === Abstract AI Model Class ===
class AIModel(ABC):
    @abstractmethod
    def query(self, prompt: str) -> str:
        pass

# === OpenAI Model ===
class OpenAIModel(AIModel):
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def query(self, prompt: str) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful and knowledgeable assistant for LAPO Microfinance Bank. Always provide your complete answer in a single response. Be concise and direct."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 500
        }
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            raw_response = result["choices"][0]["message"]["content"].strip()
            return post_process_response(consolidate_response(raw_response))
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return "Sorry, I couldn't get the information at the moment."

# === LLaMA Model ===
class LLaMAModel(AIModel):
    def __init__(self, api_url: str):
        self.api_url = api_url

    def query(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        data = {
            "prompt": prompt,
            "temperature": 0.4,
            "max_tokens": 500
        }
        try:
            response = requests.post(self.api_url, json=data, headers=headers)
            response.raise_for_status()
            result = response.json()
            print(result)
            raw_response = result.get("response", "Sorry, I couldn't get the information at the moment.")
            processed_response = post_process_response(consolidate_response(raw_response))
            return processed_response
        except Exception as e:
            logger.error(f"LLaMA API call failed: {e}")
            return "Sorry, I couldn't get the information at the moment."

# === Groq Model ===
class GroqModel(AIModel):
    def __init__(self, api_key: str, model: str = "meta-llama/llama-4-maverick-17b-128e-instruct"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

    def query(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        system_message = """You are Lala, a smart AI assistant for LAPO Microfinance Bank.
IMPORTANT: Always provide your COMPLETE answer in a SINGLE response.
Never split your answer into multiple messages.
Be concise, clear, and well-organized.
If listing items, use a single numbered or bulleted list.
Avoid repeating information."""
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 800
        }
        try:
            response = requests.post(self.api_url, headers=headers, json=data, timeout=15)
            print(json.dumps(data, indent=2))
            response.raise_for_status()
            raw_response = response.json()["choices"][0]["message"]["content"].strip()
            processed_response = post_process_response(consolidate_response(raw_response))
            return processed_response
        except Exception as e:
            print('Response body:', response.text if 'response' in locals() else "No response")
            logger.error(f"Groq API failed: {e}")
            raise


# === Fallback Action ===
class ActionFallbackOpenAI(Action):

    def __init__(self, ai_model: AIModel = GroqModel(api_key=GROQ_API), fallback_model: AIModel = LLaMAModel(api_url=LLAMA_API_URL)):
        self.ai_model = ai_model
        self.fallback = fallback_model

    def name(self):
        return "action_fallback_llama"

    async def run(self, dispatcher, tracker: Tracker, domain):
        intent_ranking = tracker.latest_message.get("intent_ranking", [])
        if intent_ranking and intent_ranking[0]["name"] == "nmr":
            dispatcher.utter_message(text="I'm not sure how to help with that. Can you rephrase?")
            return []

        conversation = await self.get_full_conversation(dispatcher, tracker)
        user_message = tracker.latest_message.get("text")

        summary = ScraperUtils.summarize_text(" ".join(conversation))

        faq_answer = self.check_faq_override(user_message)
        doc_summary = ""
        
        if faq_answer:
            logger.info("FAQ match used.")
            doc_summary = faq_answer
        else:
            doc_summary = company_doc_retriever.search(user_message)

        combined_prompt = f"""You are Lala, a smart, internet-aware AI assistant for LAPO Microfinance Bank, headquartered at Maryland, Lagos, Nigeria.

IMPORTANT INSTRUCTIONS:
1. Provide a single, cohesive response with logical organization
2. Avoid repeating the same information multiple times
3. Be concise and direct - prioritize clarity over verbosity
4. If listing items, use a single numbered or bulleted list
5. Format your response as a single coherent paragraph when appropriate
6. Only ask one follow-up question at the end if relevant

### Conversation so far:
{summary}

### User Query:
{user_message}

### Company Document Summary:
{doc_summary}

### Response:"""

        # Try Groq first
        try:
            response = self.ai_model.query(combined_prompt)
            # Send as a single message
            dispatcher.utter_message(text=response)
        except Exception as e:
            logger.error(f"Primary model failed: {e}")
            # Fallback to LLaMA if Groq fails
            try:
                fallback_response = self.fallback.query(combined_prompt)
                dispatcher.utter_message(text=fallback_response)
            except Exception as e2:
                logger.error(f"Fallback model also failed: {e2}")
                dispatcher.utter_message(text="I'm having trouble connecting to my knowledge base right now. Can you please try again in a moment?")

        return []

    def check_faq_override(self, user_message: str) -> str:
        faqs = load_faqs(FAQ_PATH)
        if not faqs:
            return ""

        query_embedding = embedding_model.encode(user_message, convert_to_tensor=True)
        best_match = (None, 0.0)
        for faq in faqs:
            question = faq.get("question", "")
            answer = faq.get("answer", "")
            question_embedding = embedding_model.encode(question, convert_to_tensor=True)
            score = util.pytorch_cos_sim(query_embedding, question_embedding).item()
            if score > best_match[1]:
                best_match = (answer, score)

        if best_match[1] >= SIMILARITY_THRESHOLD:
            return best_match[0]
        return ""

    async def get_full_conversation(self, dispatcher: CollectingDispatcher, tracker: Tracker):
        conversation_turns = []
        for event in tracker.events:
            if event['event'] == 'user':
                conversation_turns.append(f"User: {event.get('text')}")
            elif event['event'] == 'bot':
                conversation_turns.append(f"Bot: {event.get('text')}")
        return conversation_turns[:-1]

company_doc_retriever = CompanyDocRetriever(DOCS_FOLDER)
