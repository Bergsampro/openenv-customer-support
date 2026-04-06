from fastapi import FastAPI
import uvicorn
import sys
import os

# Allow it to find environment.py in the root folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment import CustomerSupportEnv, Action

app = FastAPI()
env = CustomerSupportEnv()

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/reset")
def reset(task_id: str = "task_1_easy"):
    return env.reset(task_id=task_id).model_dump()

@app.post("/step")
def step(action: Action):
    return env.step(action)

@app.get("/state")
def state():
    return env.state().model_dump()

def start():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    start()