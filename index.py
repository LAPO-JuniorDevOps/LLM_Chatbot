import sys
import os

# Point to your project directory
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "rasabot")))

from django.core.wsgi import get_wsgi_application
os.environ['DJANGO_SETTINGS_MODULE'] = 'rasabot.settings'
application = get_wsgi_application()
