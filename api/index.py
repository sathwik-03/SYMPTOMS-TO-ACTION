import sys
import os

# Add the project root to the python path so imports resolve correctly for src/ and api_server.py
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Vercel serverless function expects an ASGI application named 'app'
from api_server import app
