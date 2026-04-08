import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI
from inference import run
app = FastAPI()


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/run")
def run_endpoint():
    """
    OpenEnv-required endpoint.
    Calls your inference.run() function.
    """
    try:
        result = run()
        return result
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/")
def home():
    return {
        "name": "Email Triage OpenEnv",
        "status": "running",
        "endpoint": "/run",
        "usage": "POST with {task: easy|medium|hard}"
    }

def main():
    """
    Entry point for: uv run server
    """
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()