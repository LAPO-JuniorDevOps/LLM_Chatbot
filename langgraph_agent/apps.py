from django.apps import AppConfig
from .langgraph_faiss import schedule_midnight_update

class LanggraphAgentConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'langgraph_agent'

    def ready(self):
        schedule_midnight_update()
