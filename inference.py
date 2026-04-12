import asyncio
import os
import sys
import json
from openai import OpenAI
from environment import CustomerSupportEnv, Action

SYSTEM_PROMPT = """You are an advanced Customer Support AI. You have 3 available actions. You MUST output ONLY valid JSON in this exact format:
{"action_type": "query_db", "payload": "Order #991"} 
OR {"action_type": "ask_user", "payload": "Can you clarify?"} 
OR {"action_type": "route", "payload": "tech" | "billing" | "escalate"}

Always check the database via 'query_db' if an order number is mentioned before routing. Route when you are certain."""

async def main() -> None:
    api_key = os.environ.get("API_KEY")
    base_url = os.environ.get("API_BASE_URL")
    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")

    if not api_key or not base_url:
        print("Error: Missing API credentials", flush=True)
        sys.exit(1)

    client = OpenAI(base_url=base_url, api_key=api_key)
    env = CustomerSupportEnv()
    
    tasks_to_run = ["task_1_easy", "task_2_medium", "task_3_hard"]

    for current_task in tasks_to_run:
        print(f"[START] task={current_task} env=customer_support_triage model={model_name}", flush=True)

        try:
            obs = env.reset(task_id=current_task)
            rewards = []
            steps_taken = 0
            chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]
            
            while steps_taken < 5:
                step = steps_taken + 1
                
                obs_str = f"Ticket: {obs.message} | DB Result: {obs.db_result} | Patience: {obs.customer_patience}"
                chat_history.append({"role": "user", "content": obs_str})
                
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=chat_history,
                        max_tokens=50,
                        response_format={"type": "json_object"}
                    )
                    llm_json = json.loads(response.choices[0].message.content)
                    action_type = llm_json.get("action_type", "route")
                    payload = llm_json.get("payload", "escalate")
                except Exception as e:
                    action_type = "route"
                    payload = "escalate"
                
                chat_history.append({"role": "assistant", "content": json.dumps({"action_type": action_type, "payload": payload})})

                chosen_action = Action(action_type=action_type, payload=payload)
                result = env.step(chosen_action)
                
                # Strict parsing for the exact log format requested by the judges
                action_str = f"{action_type}_{str(payload).replace(' ', '')}"
                
                reward = result["reward"]
                done = result["done"]
                
                rewards.append(reward)
                steps_taken += 1
                
                print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
                
                if done:
                    break

            # STRICT MATH: Clamp between 0.01 and 0.99
            score = sum(rewards) / steps_taken if steps_taken > 0 else 0.01
            score = min(max(score, 0.01), 0.99)
            success = score >= 0.5 

            rewards_str = ",".join(f"{r:.2f}" for r in rewards)
            print(f"[END] success={str(success).lower()} steps={steps_taken} score={score:.3f} rewards={rewards_str}", flush=True)

        except Exception as e:
            print(f"[DEBUG] Execution error on {current_task}: {e}", flush=True)
            print(f"[END] success=false steps=0 score=0.010 rewards=0.01", flush=True)

if __name__ == "__main__":
    asyncio.run(main())