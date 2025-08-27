from django.urls import path
from .views import chatbot_view, chatbot, rasa_proxy, home, reset_rasa_session

urlpatterns = [
    path('chatbot/', chatbot_view, name='chatbot'),
    path('chatbot-old/', chatbot, name='chatbot-old'),
    path('chatbot/send/', rasa_proxy, name='rasa_proxy'),  # ✅ Ensure correct endpoint
    path('', home, name='home'),
    path('reset-session/<str:sender_id>/', reset_rasa_session, name='reset_rasa_session')
]
