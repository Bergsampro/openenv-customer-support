from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class Ticket(BaseModel):
    id: str
    message: str
    sentiment: float
    category: str

class Observation(BaseModel):
    queue_size: int
    current_ticket: Optional[Ticket]

class Action(BaseModel):
    action_type: str
    department: Optional[str] = None

class CustomerSupportEnv:
    def __init__(self):
        self.queue: List[Ticket] = []
        self.step_count = 0
        self.max_steps = 5
        self.task = "task_1_easy"

    def reset(self, task_id="task_1_easy") -> Observation:
        self.task = task_id
        self.step_count = 0
        
        if task_id == "task_1_easy":
            self.queue = [
                Ticket(id="T1", message="How do I reset my password?", sentiment=0.8, category="tech"),
                Ticket(id="T2", message="Question about my invoice.", sentiment=0.6, category="billing")
            ]
        elif task_id == "task_2_medium":
            self.queue = [
                Ticket(id="T3", message="THIS IS GARBAGE I WANT A REFUND NOW!!!", sentiment=0.1, category="billing"),
                Ticket(id="T4", message="Screen is flickering", sentiment=0.4, category="tech")
            ]
        else: # task_3_hard
            self.queue = [
                Ticket(id="T5", message="System crashed mid-transaction, lost data.", sentiment=0.2, category="tech")
            ]
        return self.state()

    def state(self) -> Observation:
        return Observation(
            queue_size=len(self.queue), 
            current_ticket=self.queue[0] if self.queue else None
        )

    def step(self, action: Action) -> Dict[str, Any]:
        self.step_count += 1
        
        if not self.queue:
            return {"observation": self.state().model_dump(), "reward": 0.99, "done": True, "info": {"msg": "Queue empty"}}

        ticket = self.queue.pop(0)
        reward = 0.01 # Strict baseline greater than 0

        if self.task == "task_1_easy":
            if action.action_type == "route" and action.department == ticket.category:
                reward = 0.99 # Strict ceiling less than 1
        
        elif self.task == "task_2_medium":
            if ticket.sentiment <= 0.2 and action.action_type == "escalate":
                reward = 0.99 
            elif action.action_type == "route" and action.department == ticket.category:
                reward = 0.50
                
        else: # task_3_hard
            if action.action_type == "escalate" and action.department == "tech":
                reward = 0.99
            elif action.action_type == "escalate":
                reward = 0.70
            elif action.action_type == "route":
                reward = 0.20
            else:
                reward = 0.01

        done = self.step_count >= self.max_steps or len(self.queue) == 0

        return {
            "observation": self.state().model_dump(),
            "reward": float(reward),
            "done": done,
            "info": {"processed_ticket": ticket.id}
        }