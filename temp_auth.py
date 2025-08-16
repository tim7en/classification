import ee

def authenticate_gee():
    """
    A simple script to handle Google Earth Engine authentication.
    """
    try:
        ee.Authenticate()
        ee.Initialize()
        print("✅ Google Earth Engine has been authenticated and initialized successfully.")
    except Exception as e:
        print(f"❌ An error occurred during authentication: {e}")
        print("Please try running 'earthengine authenticate' in your terminal if the issue persists.")

if __name__ == "__main__":
    authenticate_gee()
