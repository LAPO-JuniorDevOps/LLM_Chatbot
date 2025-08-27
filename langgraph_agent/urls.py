from django.urls import path
from .views import ask_view, summarize_view

urlpatterns = [
    path("ask/", ask_view, name="ask"),
    path("summarize/", summarize_view, name="summarize"),
]
