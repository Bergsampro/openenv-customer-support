from fastapi import FastAPI
from environment import CustomerSupportEnv, Action
import uvicorn

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