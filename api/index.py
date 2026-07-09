import sys
import os

# Add Major Project/src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../Major Project/src'))

# Import the Flask application instance
from app import app
