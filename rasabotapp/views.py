import json
import requests  # Make sure this is installed: `pip install requests`
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render



def chatbot(request):
    return render(request, 'chatpage.html')

def chatbot_view(request):
    return render(request, 'chatbot.html')


def home(request):
    return render(request, 'chatbot.html')
    # return render(request, 'front.html')

@csrf_exempt  # ✅ Disable CSRF protection for debugging (remove later for security)
def rasa_proxy(request):
    print(f"Received {request.method} request")  # Debugging output

    if request.method == "POST":
        try:
            data = json.loads(request.body.decode('utf-8'))  # ✅ Decode request body correctly
            print(f"Received data: {data}")  # Debugging output

            rasa_response = requests.post(
                "http://localhost:5005/webhooks/rest/webhook",
                json=data,  # ✅ Use `json=` instead of `data=`
                headers={"Content-Type": "application/json"}
            )

            print(f"Rasa Response: {rasa_response.text}")  # Debugging output

            return JsonResponse(rasa_response.json(), safe=False)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)
    return JsonResponse({"error": "Only POST requests are allowed"}, status=405)

@csrf_exempt
def reset_rasa_session(request, sender_id):
    if request.method in ["GET", "POST"]:
        rasa_url = f"http://localhost:5005/conversations/{sender_id}/tracker/events"
        events = [{"event": "restart"}, {"event": "reset_slots"}]
        
        try:
            response = requests.post(rasa_url, json=events)
            if response.ok:
                return JsonResponse({"status": "done"})
            else:
                return JsonResponse({"status": "error", "details": response.text}, status=500)
        except Exception as e:
            return JsonResponse({"status": "error", "exception": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Method not allowed"}, status=405)