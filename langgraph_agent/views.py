from django.shortcuts import render

from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
from .langgraph_faiss import chat_with_agent, summarize_context

@csrf_exempt
def ask_view(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            query = data.get("query", "")
            context = data.get("context", "")
            response = chat_with_agent(query, context).dict()['content']
            print(response)
            return JsonResponse({"response": response})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
def summarize_view(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            previous = data.get("previous_summary", "")
            recent = data.get("recent_turns", "")
            summary = summarize_context(previous, recent).dict()['content']
            print(summary)
            return JsonResponse({"summary": summary})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)


# Create your views here.
# @app.post("/ask")
# def ask(data: QueryInput):
#     raw_response = chat_with_agent(data.query, data.context)
#     if not isinstance(raw_response, str):
#         raw_response = raw_response.content
    
#     print('AI chatbot answers:', raw_response, data.context)
#     return {"response": raw_response}

# @app.post("/summarize")
# def summarize(data: SummarizeInput):
#     summary = summarize_context(data.previous_summary, data.recent_turns).content
#     print("summary: ", summary)
#     return {"summary": summary}
