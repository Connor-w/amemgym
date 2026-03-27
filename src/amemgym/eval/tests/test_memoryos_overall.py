## 2. test_memoryos_overall.py（单元测试）

"""Unit test for MemoryOS overall evaluation with mock data."""

import os
import json
import tempfile
import shutil
import random
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Setup paths
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ..memoryos_overall import evaluate_item, OVERALL_PROMPT
from amemgym.utils import save_json, load_json


def create_mock_env_config():
    """Create mock environment configuration."""
    return {
        "llm_config_low_temp": {
            "base_url": None,
            "api_key": None,
            "llm_model": "gpt-4.1-mini",
            "temperature": 0.0,
            "max_tokens": 8192,
            "source": "env:interaction"
        },
        "llm_config_high_temp": {
            "base_url": None,
            "api_key": None,
            "llm_model": "gpt-4.1-mini",
            "temperature": 0.7,
            "max_tokens": 8192,
            "source": "env:interaction"
        },
        "num_rounds_init": 2,
        "num_rounds_update": 2
    }


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


class MockMemoryOSAgent:
    """Mock MemoryOS Agent for testing without real LLM calls."""
    
    def __init__(self, config):
        self.config = config
        self.local_msgs = []
        self.memory_state = []
        self.call_count = 0
        
    def reset(self):
        self.local_msgs = []
        self.memory_state = []
        
    def act(self, obs: str) -> str:
        """Mock act method."""
        self.call_count += 1
        response = f"Mock response to: {obs[:30]}..."
        self.local_msgs.append({"role": "user", "content": obs})
        self.local_msgs.append({"role": "assistant", "content": response})
        self.memory_state.append({"query": obs, "response": response})
        return response
        
    def add_msgs(self, messages: list):
        """Mock add_msgs method."""
        self.local_msgs.extend(messages)
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                self.memory_state.append({
                    "query": messages[i].get("content", ""),
                    "response": messages[i+1].get("content", "")
                })
        
    def save_state(self, local_dir: str):
        """Mock save_state."""
        os.makedirs(local_dir, exist_ok=True)
        state = {
            "local_msgs": self.local_msgs,
            "memory_state": self.memory_state,
            "call_count": self.call_count
        }
        with open(os.path.join(local_dir, "mock_state.json"), "w") as f:
            json.dump(state, f, indent=2)
            
    def load_state(self, local_dir: str):
        """Mock load_state."""
        state_path = os.path.join(local_dir, "mock_state.json")
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                state = json.load(f)
            self.local_msgs = state.get("local_msgs", [])
            self.memory_state = state.get("memory_state", [])
            self.call_count = state.get("call_count", 0)
            
    def answer_question(self, question: str) -> str:
        """Mock answer_question with deterministic JSON response."""
        self.call_count += 1
        
        # Simple logic: check if question contains keywords
        question_lower = question.lower()
        
        if "drink" in question_lower or "coffee" in question_lower or "tea" in question_lower:
            # Prefer coffee in period 0, tea in period 1
            if len(self.memory_state) < 3:
                answer_num = 1  # Coffee
            else:
                answer_num = 2  # Tea
        elif "work" in question_lower or "where" in question_lower or "location" in question_lower:
            # Prefer office in period 0, home in period 1
            if len(self.memory_state) < 3:
                answer_num = 1  # Office
            else:
                answer_num = 2  # Home
        else:
            answer_num = random.randint(1, 3)
            
        return json.dumps({"answer": answer_num})


def mock_sample_session_given_query(llm_config, query, agent, start_time, user_profile, 
                                    period_end, state_schema, hist, max_rounds):
    """Mock session generation without real LLM calls."""
    session_msgs = []
    current_query = query
    
    for round_idx in range(max_rounds):
        # User message
        session_msgs.append({
            "role": "user", 
            "content": current_query,
            "timestamp": datetime.now().isoformat()
        })
        
        # Agent response
        response = agent.act(current_query)
        session_msgs.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Generate follow-up query
        current_query = f"Follow-up {round_idx + 1}: Tell me more"
        
    return session_msgs


def test_memoryos_overall():
    """Test MemoryOS overall evaluation with mock data."""
    print("=" * 60)
    print("MemoryOS Overall Evaluation Unit Test")
    print("=" * 60)
    
    # Create temporary directory
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Current directory: {current_dir}")
    # 在当前文件目录下创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="memoryos_test_", dir=current_dir)
    print(f"\nTest output directory: {temp_dir}")
    
    try:
        # Create mock data
        env_config = create_mock_env_config()
        item = create_mock_item()
        
        # Create mock agent
        agent_config = {
            "type": "memoryos",
            "name": "test-memoryos"
        }
        agent = MockMemoryOSAgent(agent_config)
        
        item_dir = os.path.join(temp_dir, item["id"])
        os.makedirs(item_dir, exist_ok=True)
        
        # Patch sample_session_given_query to avoid real LLM calls
        with patch('amemgym.eval.memoryos_overall.sample_session_given_query', mock_sample_session_given_query):
            print("\nRunning evaluate_item...")
            evaluate_item(item, agent, item_dir, env_config, off_policy=False)
        
        # Verify outputs
        print("\nVerifying outputs...")
        
        # Check results file
        results_path = os.path.join(item_dir, "overall_results.json")
        assert os.path.exists(results_path), f"Results file not found: {results_path}"
        results = load_json(results_path)
        print(f"✓ Results file created: {results_path}")
        
        # Check metrics file
        metrics_path = os.path.join(item_dir, "overall_metrics.json")
        assert os.path.exists(metrics_path), f"Metrics file not found: {metrics_path}"
        metrics = load_json(metrics_path)
        print(f"✓ Metrics file created: {metrics_path}")
        
        # Verify results structure
        num_periods = len(item["periods"])
        num_questions = len(item["qas"])
        
        assert len(results) == num_periods, f"Expected {num_periods} periods, got {len(results)}"
        assert all(len(r) == num_questions for r in results), "Mismatch in questions per period"
        print(f"✓ Results structure correct: {num_periods} periods x {num_questions} questions")
        
        # Verify each result has required fields
        for pi, period_results in enumerate(results):
            for qi, result in enumerate(period_results):
                assert result is not None, f"Missing result at period {pi}, question {qi}"
                required_fields = ["query", "answer", "response", "scores", "json_error"]
                for field in required_fields:
                    assert field in result, f"Missing field '{field}' in result [{pi}][{qi}]"
        print("✓ All results have required fields")
        
        # Verify scores
        for metric_name in metrics:
            metric_data = metrics[metric_name]
            assert len(metric_data) == num_periods, f"Metric {metric_name} has wrong period count"
            for period_scores in metric_data:
                assert len(period_scores) == num_questions, f"Metric {metric_name} has wrong question count"
        print("✓ All metrics computed correctly")
        
        # Verify agent states were saved
        for pi in range(num_periods):
            state_dir = os.path.join(item_dir, f"agent_states/period_{pi:02d}")
            assert os.path.exists(state_dir), f"Agent state dir not found: {state_dir}"
            state_file = os.path.join(state_dir, "mock_state.json")
            assert os.path.exists(state_file), f"State file not found: {state_file}"
        print(f"✓ Agent states saved for all {num_periods} periods")
        
        # Verify interactions were saved
        for pi in range(num_periods):
            interactions_path = os.path.join(item_dir, f"interactions/period_{pi:02d}.json")
            assert os.path.exists(interactions_path), f"Interactions not found: {interactions_path}"
            interactions = load_json(interactions_path)
            assert len(interactions) == len(item["periods"][pi]["sessions"]), \
                f"Wrong number of interactions in period {pi}"
        print(f"✓ Interactions saved for all {num_periods} periods")
        
        # Print summary
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        print(f"Total agent calls: {agent.call_count}")
        print(f"Memory state entries: {len(agent.memory_state)}")
        print(f"Results shape: {len(results)} periods x {len(results[0])} questions")
        
        # Sample result
        sample = results[0][0]
        print(f"\nSample result:")
        print(f"  Query: {sample['query'][:50]}...")
        print(f"  Golden answer: {sample['answer']}")
        print(f"  Response: {sample['response']}")
        print(f"  JSON error: {sample['json_error']}")
        print(f"  Scores: {sample['scores']}")
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up: {temp_dir}")


def test_overall_prompt():
    """Test the OVERALL_PROMPT template."""
    print("\n" + "=" * 60)
    print("Testing OVERALL_PROMPT template")
    print("=" * 60)
    
    query = "What is my preferred drink?"
    choices = "1: Coffee\n2: Tea\n3: Water"
    
    prompt = OVERALL_PROMPT.format(query=query, choices=choices)
    print(f"\nGenerated prompt:\n{prompt}")
    
    assert query in prompt
    assert choices in prompt
    assert "JSON format" in prompt
    assert '"answer": int' in prompt
    
    print("✓ OVERALL_PROMPT template correct")
    return True


if __name__ == "__main__":
    # Run tests
    success = True
    
    success = test_overall_prompt() and success
    success = test_memoryos_overall() and success
    
    sys.exit(0 if success else 1)