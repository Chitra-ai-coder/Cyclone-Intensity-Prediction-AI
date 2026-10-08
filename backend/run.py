import subprocess
import webbrowser
import time
import os
import sys

def main():
    print("Initializing Cyclone Intensity AI...")
    
    # 1. Locate the exact uvicorn executable inside your active .venv
    venv_bin = os.path.dirname(sys.executable)
    uvicorn_path = os.path.join(venv_bin, "uvicorn")
    
    # Fallback in case uvicorn is installed globally instead of in .venv
    if not os.path.exists(uvicorn_path):
        uvicorn_path = "uvicorn"

    # 2. Boot the FastAPI backend using the direct executable path
    server_process = subprocess.Popen(
        [uvicorn_path, "backend.main:app", "--reload"],
        cwd=os.getcwd()
    )

    # 3. Give the server 2 seconds to fully load the .pth weights into memory
    time.sleep(2)
    
    # 4. Locate the frontend HTML file
    possible_paths = ["index.html", "frontend/index.html"]
    html_path = None
    
    for path in possible_paths:
        if os.path.exists(path):
            html_path = os.path.abspath(path)
            break
            
    # 5. Automatically launch the default web browser
    if html_path:
        print(f"Launching Frontend UI: {html_path}")
        webbrowser.open(f"file://{html_path}")
    else:
        print("Warning: index.html not found! Please check your file structure.")

    print("\nSystem is LIVE! Press the 'Stop' button in VS Code to shut everything down.")
    
    # 6. Keep the terminal alive so the server doesn't crash
    try:
        server_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down AI Backend...")
        server_process.terminate()
        server_process.wait()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()