from app import app
from pyngrok import ngrok

# Open a ngrok tunnel to the HTTP server
public_url = ngrok.connect(5000)
print(f"Public URL: {public_url}")

# Start the Flask server
app.run(host='0.0.0.0', port=5000)