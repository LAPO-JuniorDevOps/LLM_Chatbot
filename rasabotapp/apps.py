from django.apps import AppConfig
import threading
import time
import requests


class RasabotappConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'rasabotapp'

    def ready(self):
        def reset_on_startup():
            time.sleep(3)  # Wait a bit to let server fully start
            try:
                sender_id = "default"  # or pull from env/settings
                url = f"http://localhost:8000/reset-session/{sender_id}/"
                requests.get(url)
                print("✅ Rasa session reset on Django startup.")
            except Exception as e:
                print("❌ Failed to reset Rasa session:", e)

        threading.Thread(target=reset_on_startup).start()