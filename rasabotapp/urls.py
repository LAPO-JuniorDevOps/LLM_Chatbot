from django.urls import path
from .views import chatbot, rasa_proxy, home

urlpatterns = [
    path('chatbot/', chatbot, name='chatbot'),
    path('chatbot/send/', rasa_proxy, name='rasa_proxy'),  # ✅ Ensure correct endpoint
    path('', home, name='home')
]
