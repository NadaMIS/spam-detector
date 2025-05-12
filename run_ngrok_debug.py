import sys
import time
from app import app

print("Starting ngrok tunnel...")

try:
    # Import pyngrok and configure
    from pyngrok import ngrok, conf
    from pyngrok.exception import PyngrokNgrokError
    
    # Set higher timeout
    conf.get_default().request_timeout = 30.0
    
    # Try to connect with retries
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            print(f"Attempt {retry_count + 1}/{max_retries} to establish ngrok tunnel...")
            # Open ngrok tunnel to HTTP server
            public_url = ngrok.connect(5000)
            print(f"Success! Public URL: {public_url}")
            break
        except PyngrokNgrokError as e:
            retry_count += 1
            if retry_count >= max_retries:
                print(f"Failed to establish ngrok tunnel after {max_retries} attempts")
                print(f"Error: {e}")
                print("Running in local mode only")
                break
            print(f"Connection failed: {e}")
            print(f"Retrying in 5 seconds...")
            time.sleep(5)
except ImportError:
    print("Error: pyngrok not installed")
    print("Install with: pip3 install pyngrok")
    print("Running in local mode only")
except Exception as e:
    print(f"Unexpected error setting up ngrok: {e}")
    print("Running in local mode only")

# Start the Flask app
print("Starting Flask application...")
try:
    app.run(host='0.0.0.0', port=5000)
except Exception as e:
    print(f"Error starting Flask: {e}")
    sys.exit(1)