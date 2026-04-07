import asyncio
import os
import sys
from openai import OpenAI
from environment import CustomerSupportEnv, Action

async def main() -> None:
    # 1. STRICT CREDENTIALS (Matches Sample & Proxy Requirements)
    api_key = os.environ.get("API_KEY")
    base_url = os.environ.get("API_BASE_URL")
    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")

    if not api_key or not base_url:
        print("Error: Missing API_KEY or API_BASE_URL from the hackathon environment.")
        sys.exit(1)

    # 2. INITIALIZE CLIENT (Must use the injected proxy URL)
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )

    env = CustomerSupportEnv()
    
    # 3. EXACT [START] LOG FORMAT
    print(f"[START] task=task_1_easy env=customer_support_triage model={model_name}", flush=True)

    try:
        obs = env.reset(task_id="task_1_easy")
        
        rewards = []
        steps_taken = 0
        
        # Max steps allowed loop
        for step in range(1, 6):
            # Baseline Action
            chosen_action = Action(action_type="route", department="tech")
            
            result = env.step(chosen_action)
            
            reward = result["reward"] or 0.0
            done = result["done"]
            
            rewards.append(reward)
            steps_taken = step
            
            # 4. EXACT [STEP] LOG FORMAT (Lowercase booleans, 2 decimal floats)
            print(f"[STEP] step={step} action=route_to_tech reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
            
            if done:
                break

        # Calculate final score based on sample logic
        score = sum(rewards) / steps_taken if steps_taken > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= 0.1 # Example threshold

        # 5. EXACT [END] LOG FORMAT
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={str(success).lower()} steps={steps_taken} score={score:.3f} rewards={rewards_str}", flush=True)

    except Exception as e:
        print(f"[DEBUG] Execution error: {e}", flush=True)
        print(f"[END] success=false steps=0 score=0.000 rewards=0.00", flush=True)

if __name__ == "__main__":
    asyncio.run(main())