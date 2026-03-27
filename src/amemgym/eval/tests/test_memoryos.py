import os
from dotenv import load_dotenv
from argparse import ArgumentParser


import json
import tempfile
import shutil
import random
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Setup paths
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ..memoryos_overall import evaluate_item, OVERALL_PROMPT,setup_logger, logger
from amemgym.assistants import create_agent

from amemgym.utils import save_json, load_json


def create_mock_item():
    """Create mock evaluation item with minimal data."""
    base_time = datetime(2024, 1, 1, 10, 0, 0)

    return {
        "id": "test_item_001",
        "start_time": base_time.isoformat(),
        "user_profile": "Test user who likes coffee and works in tech.",
        "state_schema": {
            "preference": "string",
            "location": "string"
        },
        "periods": [
            {
                "period_end": (base_time + timedelta(hours=1)).isoformat(),
                "state": {
                    "preference": "coffee",
                    "location": "office"
                },
                "sessions": [
                    {
                        "session_time": (base_time + timedelta(minutes=10)).isoformat(),
                        "query": "What should I drink to stay focused?"
                    },
                    {
                        "session_time": (base_time + timedelta(minutes=30)).isoformat(),
                        "query": "Where should I work today?"
                    }
                ]
            },
            {
                "period_end": (base_time + timedelta(hours=2)).isoformat(),
                "state": {
                    "preference": "tea",
                    "location": "home"
                },
                "sessions": [
                    {
                        "session_time": (base_time + timedelta(minutes=70)).isoformat(),
                        "query": "I want to try something different today."
                    }
                ]
            }
        ],
        "qas": [
            {
                "query": "What is my preferred drink?",
                "required_info": ["preference"],
                "answer_choices": [
                    {"answer": "Coffee", "state": ["coffee", "office"]},
                    {"answer": "Tea", "state": ["tea", "home"]},
                    {"answer": "Water", "state": ["water", "gym"]}
                ]
            },
            {
                "query": "Where do I prefer to work?",
                "required_info": ["location"],
                "answer_choices": [
                    {"answer": "Office", "state": ["coffee", "office"]},
                    {"answer": "Home", "state": ["tea", "home"]},
                    {"answer": "Cafe", "state": ["coffee", "cafe"]}
                ]
            }
        ]
    }


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate MemoryOS agents")
    parser.add_argument("--env_data", type=str, default="data/v1.base/data.json")
    parser.add_argument("--env_config", type=str, default="configs/env/v1.base.json")
    parser.add_argument("--agent_config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="eval-output/v1.base/memoryos")
    parser.add_argument("--off_policy_dir", type=str, default="")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    load_dotenv()

    agent_config = load_json(args.agent_config)
    env_config = load_json(args.env_config)
    for key in ["llm_config_low_temp", "llm_config_high_temp"]:
        env_config[key] |= {
            "base_url": env_config[key].get("base_url") or os.environ.get("OPENAI_BASE_URL"),
            "api_key": env_config[key].get("api_key") or os.environ.get("OPENAI_API_KEY"),
            "source": "env:interaction"
        }

    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Current directory: {current_dir}")
    output_dir = os.path.join(current_dir, agent_config["name"])
    if args.reset and os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(args.agent_config, output_dir)

    data = load_json(args.env_data)
    # data = create_mock_item()
    for item in data:
        item_dir = os.path.join(output_dir, item["id"])
        os.makedirs(item_dir, exist_ok=True)

        agent = create_agent(agent_config, output_dir=item_dir)

        save_json(os.path.join(item_dir, "agent_config.json"), agent_config)

        if args.off_policy_dir:
            off_policy = True
            shutil.copytree(
                os.path.join(args.off_policy_dir, item["id"], "interactions"),
                os.path.join(item_dir, "interactions"),
                dirs_exist_ok=True
            )
        else:
            off_policy = False
            os.makedirs(os.path.join(item_dir, "interactions"), exist_ok=True)

        log_dir = os.path.join(item_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        setup_logger(os.path.join(log_dir, "evaluate.log"))
        logger.info(item["id"])
        evaluate_item(item, agent, item_dir, env_config, off_policy)