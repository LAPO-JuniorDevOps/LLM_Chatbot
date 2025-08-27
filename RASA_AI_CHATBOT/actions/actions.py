# actions.py
import requests
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from typing import Any, Dict, List, Text
import os

# Base URL from environment variable
BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

class ActionFallbackOpenAI(Action):
    def name(self) -> Text:
        return "action_fallback_llama"

    def extract_recent_convo(self, tracker: Tracker, max_turns: int = 4) -> str:
        events = tracker.events
        turns = []
        for e in events:
            if e.get("event") == "user":
                turns.append(f"User: {e.get('text')}")
            elif e.get("event") == "bot":
                turns.append(f"Bot: {e.get('text')}")
        return "\n".join(turns[-(2 * max_turns):])

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:

        latest_user_input = tracker.latest_message.get("text")
        previous_summary = tracker.get_slot("conversation_summary") or ""
        print('Previous Summary: ',previous_summary)
        recent_turns = self.extract_recent_convo(tracker)

        # ✨ Summarization logic here
        try:
            summary_response = requests.post(f"{BASE_URL}/summarize/", json={
                "previous_summary": previous_summary,
                "recent_turns": recent_turns
            })
            print({
                "previous_summary": previous_summary,
                "recent_turns": recent_turns
            })
            new_summary = summary_response.json().get("summary", previous_summary)
            print('Summary: ', new_summary)
        except Exception as e:
            print('This error occured:', e)
            new_summary = previous_summary  # fallback if summarizer fails

        # 🔍 Now query the RAG agent with full memory
        try:
            if new_summary == '':
                new_summary = recent_turns
            rag_response = requests.post(f"{BASE_URL}/ask/", json={
                "query": latest_user_input,
                "context": new_summary,
            })
            print({
                "query": latest_user_input,
                "context": new_summary
            })
            response = rag_response.json().get("response", "error")
            # print(response)
            result = response
        except Exception as e:
            result = f"❌ Agent failed: {str(e)}"
            print(result)
            result = "Sorry, I couldn’t process that."

        
        dispatcher.utter_message(text=result)
        return [SlotSet("conversation_summary", new_summary)]


