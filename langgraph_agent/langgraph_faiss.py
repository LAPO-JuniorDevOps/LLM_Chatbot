# from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import (
    WebBaseLoader, DirectoryLoader, TextLoader, PyPDFLoader,
    JSONLoader, UnstructuredWordDocumentLoader,
    UnstructuredHTMLLoader
)
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from dotenv import load_dotenv

import threading
import time
from datetime import datetime, timedelta
import os, json
import concurrent.futures

load_dotenv()  # Load variables from .env

# 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
EXA_API_KEY = os.getenv("EXA_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY") 
os.environ["GOOGLE_CSE_ID"] = os.getenv("GOOGLE_CSE_ID") 
os.environ["USER_AGENT"] = "AIAssistant/1.0"  # Fix User-Agent warning

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["USER_AGENT"] = "LapoAssistant/1.0"

import requests
import os


class ExaSearchAPI:
    def __init__(self, api_key=None):
        self.api_key = api_key or EXA_API_KEY
        self.api_url = "https://api.exa.ai/search"
        if not self.api_key:
            raise ValueError("EXA API key missing. Set it as ENV variable 'EXA_API_KEY' or pass explicitly.")

    def search(self, query, num_results=5):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "query": query,
            "numResults": num_results,
            "includeDomains": [],
            "useAutoprompt": True,
            "getRawContent": True
        }
        response = requests.post(self.api_url, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("results", []):
            results.append({
                "title": item.get("title"),
                "url": item.get("url"),
                "content": item.get("content", "")[:3000]
            })
        return results
exa_search_client = ExaSearchAPI()

# 
# app = FastAPI()

# 
class QueryInput(BaseModel):
    query: str
    context: str

class SummarizeInput(BaseModel):
    previous_summary: str
    recent_turns: str

# 
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore_path = "company_docs_index"
company_docs = r"C:\inetpub\wwwroot\LLM_Chatbot\langgraph_agent\company_docs"

def load_json_files(folder: str):
    docs = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".json"):
                try:
                    loader = JSONLoader(
                        file_path=os.path.join(root, file),
                        jq_schema=".text",
                        text_content=True
                    )
                    docs.extend(loader.load())
                except: pass
    return docs

def build_index():
    docs = []
    docs.extend(DirectoryLoader(company_docs, glob="/*.txt", loader_cls=TextLoader).load())
    docs.extend(DirectoryLoader(company_docs, glob="/*.pdf", loader_cls=PyPDFLoader).load())
    docs.extend(DirectoryLoader(company_docs, glob="/*.docx", loader_cls=UnstructuredWordDocumentLoader).load())
    docs.extend(DirectoryLoader(company_docs, glob="/*.html", loader_cls=UnstructuredHTMLLoader).load()) # Add this line for HTML
    docs.extend(load_json_files(company_docs))
    docs.extend(WebBaseLoader(["https://www.lapo-nigeria.org/"]).load())

    global vectorstore, retriever
    vectorstore = FAISS.from_documents(docs, embedding=embedding_model)
    vectorstore.save_local(vectorstore_path)
    retriever = vectorstore.as_retriever()

if os.path.exists(vectorstore_path):
    vectorstore = FAISS.load_local(vectorstore_path, embeddings=embedding_model, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()
else:
    build_index()


def schedule_midnight_update():
    def updater():
        while True:
            now = datetime.now()
            next_midnight = datetime.combine(now.date() + timedelta(days=1), datetime.min.time())
            sleep_seconds = (next_midnight - now).total_seconds()
            print(f"🕛 Sleeping until midnight ({sleep_seconds:.0f} seconds)...")
            time.sleep(sleep_seconds)

            try:
                print("🔄 Midnight index refresh started...")
                build_index()
                print("✅ Document index refreshed at midnight.")
            except Exception as e:
                print(f"❌ Failed to refresh index at midnight: {e}")
    threading.Thread(target=updater, daemon=True).start()

schedule_midnight_update()

# 
@tool
def scrape_website(url: str) -> str:
    """Scrape visible content from a webpage."""
    try:
        loader = WebBaseLoader([url])
        return loader.load()[0].page_content[:3000]
    except Exception as e:
        return f"Scrape failed: {e}"

search = DuckDuckGoSearchAPIWrapper()
google_search = GoogleSearchAPIWrapper()


def scrape_websites_concurrently(urls):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(scrape_website, url): url for url in urls}
        results = []
        for future in concurrent.futures.as_completed(futures):
            url = futures[future]
            try:
                result = future.result()
                results.append({"url": url, "content": result})
            except Exception as e:
                print(f"Error scraping {url}: {e}")
        return results







@tool
def search_and_scrape(query: str, num_urls_to_scrape: int = 5) -> str:
    """
    Real-time prioritized web search tool:
    1. EXA AI Search API (Primary)
    2. DuckDuckGo Search (Fallback)
    3. Google Programmable Search (Final fallback)

    Returns:
        str: Compiled readable blocks of the latest web content.
    """

    context_blocks = []

     # --- PRIORITY 1: EXA ---
    try:
        exa_results = exa_search_client.search(query=query, num_results=num_urls_to_scrape)
        if exa_results:
            print("✅ EXA API returned results.")
            urls_to_scrape = [res["url"] for res in exa_results]
            scraped_contents = scrape_websites_concurrently(urls_to_scrape)

            for res in exa_results:
                scraped_data = next((item for item in scraped_contents if item["url"] == res["url"]), None)
                # If scraped_data is None or its content is None, fallback to Exa's original content (snippet)
                content = scraped_data["content"] if scraped_data and scraped_data["content"] is not None else res.get("content", "No content.")
                block = f"📌 {res['title']}\n🔗 {res['url']}\n{content}"
                context_blocks.append(block)
            return "\n\n".join(context_blocks)
        else:
            print("⚠ EXA API returned no results.")
    except Exception as exa_err:
        print(f"⚠ EXA API failed: {exa_err}. Falling back to DuckDuckGo...")
   
    # --- PRIORITY 2: DuckDuckGo ---
    try:
        ddg_results = search.results(query, max_results=num_urls_to_scrape)
        if ddg_results:
            print("✅ DuckDuckGo returned results.")
            urls_to_scrape = [r["link"] for r in ddg_results]
            scraped_contents = scrape_websites_concurrently(urls_to_scrape)

            for r in ddg_results:
                scraped_data = next((item for item in scraped_contents if item["url"] == r["link"]), None)
                content = scraped_data["content"] if scraped_data else r.get('snippet', 'No snippet.')
                block = f"📌 {r['title']}\n🔗 {r['link']}\n{content}"
                context_blocks.append(block)
            return "\n\n".join(context_blocks)
        else:
            print("⚠ DuckDuckGo returned no results.")
    except Exception as ddg_err:
        print(f"⚠ DuckDuckGo failed: {ddg_err}. Falling back to Google...")

    # --- PRIORITY 3: Google Programmable Search ---
    try:
        google_results = google_search.results(query=query, num_results=num_urls_to_scrape)
        if google_results:
            print("✅ Google Programmable Search returned results.")
            urls_to_scrape = [r["link"] for r in google_results]
            scraped_contents = scrape_websites_concurrently(urls_to_scrape)

            for r in google_results:
                scraped_data = next((item for item in scraped_contents if item["url"] == r["link"]), None)
                content = scraped_data["content"] if scraped_data else r.get('snippet', 'No snippet.')
                block = f"📌 {r['title']}\n🔗 {r['link']}\n{content}"
                context_blocks.append(block)
            return "\n\n".join(context_blocks)
        else:
            print("⚠ Google returned no results.")
    except Exception as google_err:
        print(f"❌ Google Programmable Search failed: {google_err}.")

    return "❌ No usable search results could be retrieved from any source."


    # for res in results:
    #     if scraped_urls_count >= num_urls_to_scrape:
    #         break
    #     url = res.get("link", "")
    #     snippet = res.get("snippet", "No snippet.")
    #     if not url:
    #         continue

    #     # Exclude Instagram and Facebook URLs from full scraping
    #     if "instagram.com" in url or "facebook.com" in url:
    #         texts.append(f"From {url} (snippet only, full scraping skipped due to platform restrictions):\n{snippet}")
    #         scraped_urls_count += 1
    #         continue

    #     for attempt in range(max_retries_per_url):
    #         try:
    #             scraped = scrape_website(url)
    #             if scraped and len(scraped.strip()) >= 100:  # Accept if decent content found
    #                 texts.append(f"From {url}:\n{scraped}")
    #                 scraped_urls_count += 1
    #                 break  # Break from retry loop if successful
    #             else:
    #                 print(f"Attempt {attempt + 1} to scrape {url} yielded insufficient content. Retrying...")
    #         except Exception as e:
    #             print(f"❌ Attempt {attempt + 1} to scrape {url} failed: {e}. Retrying...")
    #     else:  # This else block executes if the for loop completes without a 'break'
    #         texts.append(f"From {url} (snippet only, failed to scrape after {max_retries_per_url} attempts):\n{snippet}")
    #         scraped_urls_count += 1

    # return "\n\n".join(texts) if texts else "No usable search results could be retrieved from any source."


# 
rag_prompt = PromptTemplate.from_template("""
Use the following context to answer the user's question.

<context>
{context}
</context>

Question: {question}
""")

rag_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=OPENAI_API_KEY),
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": rag_prompt}
)

@tool
def rag_lookup(query: str) -> str:
    """Use this tool to find information specifically within LAPO Microfinance Bank's internal documents. Always use this when the user asks about LAPO's services, policies, or internal information."""
    return rag_chain.invoke({"query": query})["result"]


@tool
def smart_lapo_search(query: str) -> str:
    """
    Intelligently searches for information about LAPO Microfinance Bank.
    Prioritizes internal RAG lookup, falling back to web search if needed.
    """
    print(f"Smart LAPO Search initiated for: {query}")

    rag_result = rag_lookup(query)
    web_result = search_and_scrape(query)

    return f"Results from Internal Search: {rag_result}\n In addition, here's what I found on the web: {web_result}"

retriever_tool = create_retriever_tool(
    retriever,
    name="company_docs",
    description="Answer questions about LAPO Microfinance Bank using internal documents and website."
)

@tool
def chatgpt_lookup(query: str) -> str:
    """Use this tool for general knowledge questions that are specifically about LAPO Microfinance Bank. It will not answer anything outside that scope."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a helpful assistant that ONLY answers questions related to LAPO Microfinance Bank. "
         "If the question is not about LAPO Microfinance Bank, respond with: "
         "'I'm sorry, I can only answer questions related to LAPO Microfinance Bank.'"),
        ("human", "{query}")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=OPENAI_API_KEY)

   # Generate the full message list from the template
    messages = prompt.format_messages(query=query)

    # Invoke the model directly with the formatted messages
    response = llm.invoke(messages)

    return response.content
# 
openai_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=OPENAI_API_KEY)

agent_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are Lala, a helpful, professional, and accurate AI assistant for LAPO Microfinance Bank. "
     "Your primary goal is to assist users with inquiries related to LAPO Microfinance Bank's services, policies, and general financial information.\n\n"
     "*Tool Usage Guidelines:*\n"
     "1.  *Prioritize Recent Knowledge:* For questions specifically about LAPO Microfinance Bank products, always attempt to use search_and_scrape first. These contain the most accurate and up-to-date information.\n"
     "2.  *Prioritize Internal Knowledge:* For questions specifically about LAPO Microfinance Bank (e.g., account types, loan criteria, contact details), always attempt to use company_docs or rag_lookup first. These contain the most accurate and up-to-date internal information.\n"
     "3.  **smart_lapo_search**: Use this for almost all questions related to LAPO Microfinance Bank, its services, policies, or any general information. This tool intelligently uses internal documents first and then falls back to web search if necessary.\n"
     "4.  **chatgpt_lookup**: Use this tool for general knowledge related to LAPO, to provide broader context and comprehensive answers when smart_lapo_search might not be sufficient. This tool provides direct, comprehensive answers similar to ChatGPT.\n"
     "5.  **scrape_website**: Use this only when the user provides a specific URL and explicitly asks for its content to be retrieved or summarized.\n"
     "*Response Guidelines:*\n"
     "1.  *Accuracy:* Always strive for the most accurate information available through your tools.\n"
     "2.  *Conciseness:* Provide clear, direct, and concise answers.\n"
     "3.  *Completeness:* Ensure your answer fully addresses the user's query based on the information you can access, combining insights from all relevant tools.\n"
     "4.  *Handling Unanswered Questions:* If you cannot find a definitive answer using your tools, clearly state that you do not have the information and suggest alternative actions (e.g., contacting LAPO directly via provided contact details).\n"
     "5.  *Safety & Ethics:* Do not provide personalized financial advice, sensitive personal information, or engage in any activity outside the scope of a banking assistant.\n"
     "6.  *Maintain Context:* Always consider the ongoing conversation history when formulating your response.\n\n"
     "7.  *Scope:* Only answer questions relating to LAPO Microfinance bank, redirect all questions that are out of scope back to LAPO"
     "*LAPO Microfinance Bank Contact Information (for your reference and to provide to users if needed):*\n"
     "*   Website: www.lapo-nigeria.org\n"
     "*   Phone: +2348139840230\n"
     "*   Email: customersupport@lapo-nigeria.org\n"
     "*   WhatsApp: 08150553264\n"
    ),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

tools = [retriever_tool, smart_lapo_search, search_and_scrape, scrape_website, chatgpt_lookup]

react_agent = create_tool_calling_agent(
    llm=openai_llm,
    tools=tools,
    prompt=agent_prompt
)

agent_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    verbose=True
)

# 
final_prompt = PromptTemplate.from_template("""
You are Lala, an AI assistant for LAPO Microfinance Bank. Your goal is to provide comprehensive, accurate, and helpful answers.

*LAPO Microfinance Bank Contact Information:*
*   Website: www.lapo-nigeria.org
*   Phone: +2348139840230
*   Email: customersupport@lapo-nigeria.org
*   WhatsApp: 08150553264
                                            
**LAPO Mobile App Google Playstore Link: https://play.google.com/store/apps/details?id=com.digicore.lapo&pcampaignid=web_share

*Current Date and Time:* {now}

---

*INSTRUCTIONS FOR GENERATING RESPONSE:*

1.  *PRIORITY:* The *USER QUERY* is the most important piece of information. Address it directly and comprehensively if it is related to LAPO microfinance bank.
2.  *Synthesize Information:* Combine relevant details from all provided contexts (LAPO internal/web, ChatGPT general knowledge) into a single, coherent, and detailed answer.
3.  *Elaborate and Explain:* Provide explanations and elaborations as needed to make the answer comprehensive, similar to how ChatGPT provides detailed responses.
4.  *Maintain LAPO Persona:* Even when using general knowledge, ensure the tone and focus remain consistent with a LAPO Microfinance Bank assistant. Integrate LAPO-specific information seamlessly.
5.  *Avoid Redundancy:* Do not re-answer questions or repeat information already covered. Focus on the user's current intent.
6.  *Conciseness (within comprehensiveness):* While comprehensive, avoid unnecessary filler. Be direct and clear.
7.  *Handling Unanswered Questions:* If information is still missing, clearly state what could not be found and suggest contacting LAPO directly.
8.  *Scope:* Only answer questions relating to LAPO Microfinance bank, redirect all questions that are out of scope back to LAPO
9.  *Image:* Always answer questions in a way that presents LAPO Microfinance bank in a positive light.
                                            

---

*PROVIDED INFORMATION:*

*USER QUERY:*
{query}

*LAPO INTERNAL/WEB CONTEXT (from smart_lapo_search):*
{lapo_context}

*CHATGPT GENERAL KNOWLEDGE CONTEXT (from chatgpt_lookup):*
{chatgpt_context}

*SUMMARY OF CONVERSATION:*
{summary}

---

*YOUR FINAL ANSWER:*
"""
)


def final_answer(query: str, lapo_context: str, chatgpt_context: str, summary: str) -> str:
    return openai_llm.invoke(final_prompt.format(now = datetime.now(), query=query, lapo_context=lapo_context, chatgpt_context=chatgpt_context, summary=summary))

def chat_with_agent(query: str, summary: str) -> str:
    print("🔍 Running agent...")

    try:
        intermediate = agent_executor.invoke({"input": query})
    except Exception as e:
        print("❌ Agent error:", e)
        output_text = "Sorry, something went wrong while processing your request."
        return final_answer(query, output_text, summary)

    lapo_context = ""
    chatgpt_context = ""

    # 
    print(f"📦 Agent Intermediate Output: {intermediate}")

    # 
    if isinstance(intermediate, dict) and "output" in intermediate:
        lapo_context = intermediate["output"]

    # 
    if isinstance(intermediate, dict) and "tool_calls" in intermediate and "tool_outputs" in intermediate:
        for i, tool_call in enumerate(intermediate["tool_calls"]):
            tool_output = intermediate["tool_outputs"][i]
            tool_name = tool_call.tool  # Name as recorded by LangChain

            print(f"🔧 Tool Called: {tool_name}")  # Debugging Tool Name

            if tool_name == "smart_lapo_search":
                lapo_context = tool_output
            elif tool_name == "chatgpt_lookup":
                chatgpt_context = tool_output

    # 
    if not lapo_context and not chatgpt_context and isinstance(intermediate, dict) and "output" in intermediate:
        lapo_context = intermediate["output"]

    # 
    print("✨ Finalizing with OpenAI...")

    return final_answer(query, lapo_context, chatgpt_context, summary)


# 
summarize_prompt = PromptTemplate.from_template("""
Based on the following conversation history, create a concise summary that captures the current state of the discussion and the user's most recent intent. Prioritize information from the 'Recent Turns' section, and ensure the summary reflects what still needs to be addressed or clarified. 
**Crucially, identify and exclude topics or questions that have been fully and satisfactorily answered in the 'Previous Summary'.**

Previous Summary:
{context}

Recent Turns:
{new_conversation}

Updated Summary:
"""
)

def summarize_context(context: str, recent_turns: str) -> str:
    return openai_llm.invoke(summarize_prompt.format(context=context, new_conversation=recent_turns))