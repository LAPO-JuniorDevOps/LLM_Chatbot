import json
import requests  # Make sure this is installed: `pip install requests`
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render

def chatbot(request):
    return render(request, 'chatpage.html')

def home(request):
    return render(request, 'front.html')

@csrf_exempt  # ✅ Disable CSRF protection for debugging (remove later for security)
def rasa_proxy(request):
    print(f"Received {request.method} request")  # Debugging output

    if request.method == "POST":
        try:
            data = json.loads(request.body.decode('utf-8'))  # ✅ Decode request body correctly
            print(f"Received data: {data}")  # Debugging output

            rasa_response = requests.post(
                "http://localhost:8085/webhooks/rest/webhook",
                json=data,  # ✅ Use `json=` instead of `data=`
                headers={"Content-Type": "application/json"}
            )

            print(f"Rasa Response: {rasa_response.text}")  # Debugging output

            return JsonResponse(rasa_response.json(), safe=False)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)
    return JsonResponse({"error": "Only POST requests are allowed"}, status=405)
