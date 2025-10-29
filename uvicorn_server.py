from api import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=5000,
        workers=1, 
        loop="asyncio",
        timeout_keep_alive=60
    )