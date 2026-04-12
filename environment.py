from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Ticket(BaseModel):
    id: str = Field(description="Unique ticket identifier")
    message: str = Field(description="The latest message from the customer")
    patience: int = Field(description="Customer patience drops each turn. If 0, ticket fails.")
    hidden_data: str = Field(description="Internal ground truth for grading")

class Observation(BaseModel):
    ticket_id: str
    message: str
    customer_patience: int
    db_result: Optional[str] = Field(description="Result from the database if queried", default=None)

class Action(BaseModel):
    action_type: str = Field(description="Must be 'ask_user', 'query_db', or 'route'")
    payload: str = Field(description="The question to ask, the DB query string, or the department to route to")

class CustomerSupportEnv:
    def __init__(self):
        self.step_count = 0
        self.max_steps = 5
        self.task = "task_1_easy"
        self.current_ticket: Optional[Ticket] = None
        self.last_db_result = None

    def reset(self, task_id="task_1_easy") -> Observation:
        self.task = task_id
        self.step_count = 0
        self.last_db_result = None
        
        if task_id == "task_1_easy":
            self.current_ticket = Ticket(id="T1", message="I forgot my password.", patience=3, hidden_data="tech")
        elif task_id == "task_2_medium":
            self.current_ticket = Ticket(id="T2", message="My screen cracked. Is it under warranty? Order #991", patience=4, hidden_data="Order #991: Warranty Expired")
        else: # task_3_hard
            self.current_ticket = Ticket(id="T3", message="You charged me twice for Order #555! Refund me!", patience=2, hidden_data="Order #555: Single charge successful. No duplicate.")
            
        return self.state()

    def state(self) -> Observation:
        if not self.current_ticket:
             return Observation(ticket_id="NONE", message="Queue empty", customer_patience=0)
        return Observation(
            ticket_id=self.current_ticket.id,
            message=self.current_ticket.message,
            customer_patience=self.current_ticket.patience,
            db_result=self.last_db_result
        )

    def step(self, action: Action) -> Dict[str, Any]:
        self.step_count += 1
        
        if not self.current_ticket:
            return {"observation": self.state().model_dump(), "reward": 0.99, "done": True, "info": {}}

        reward = 0.01
        done = False
        
        self.current_ticket.patience -= 1
        if self.current_ticket.patience <= 0:
            return {"observation": self.state().model_dump(), "reward": 0.01, "done": True, "info": {"msg": "Customer lost patience."}}

        if action.action_type == "query_db":
            if "991" in action.payload and self.task == "task_2_medium":
                self.last_db_result = self.current_ticket.hidden_data
                reward = 0.20 
            elif "555" in action.payload and self.task == "task_3_hard":
                self.last_db_result = self.current_ticket.hidden_data
                reward = 0.20
            else:
                self.last_db_result = "No records found."
                reward = 0.05 
                
        elif action.action_type == "ask_user":
            self.current_ticket.message = "I already gave you the info! Please help!"
            reward = 0.05 
            
        elif action.action_type == "route":
            done = True
            patience_bonus = (self.current_ticket.patience / 5.0) 
            
            if self.task == "task_1_easy" and action.payload == "tech":
                reward = min(0.80 + patience_bonus, 0.99)
            elif self.task == "task_2_medium" and action.payload == "billing" and self.last_db_result:
                reward = min(0.80 + patience_bonus, 0.99) 
            elif self.task == "task_3_hard" and action.payload == "escalate" and self.last_db_result:
                reward = min(0.90 + patience_bonus, 0.99) 
            else:
                reward = 0.10

        # Math Failsafe: STRICTLY between 0 and 1
        reward = max(0.01, min(reward, 0.99))
        done = done or (self.step_count >= self.max_steps)

        return {
            "observation": self.state().model_dump(),
            "reward": float(reward),
            "done": done,
            "info": {}
        }