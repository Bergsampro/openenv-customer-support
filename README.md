# OpenEnv Customer Support Triage

A real-world OpenEnv simulation for testing AI agents on ticket routing, sentiment analysis, and escalation.

## Spaces
* **Observation Space**: `queue_size` (int) and `current_ticket` (object with id, message, sentiment).
* **Action Space**: `action_type` (route, escalate, resolve) and `department` (tech, billing).

## Running Locally
Ensure you have `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` set as environment variables, then run:
`python inference.py`
