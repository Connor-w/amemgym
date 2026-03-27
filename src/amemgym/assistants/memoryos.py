# amemgym/assistants/memoryos.py

import os
import json
import shutil
from typing import List, Dict
from copy import deepcopy
from loguru import logger

from .base import BaseAgent

try:
    from memoryos import Memoryos
except ImportError:
    raise ImportError("MemoryOS未安装，请运行: uv pip install memoryos-pro -i https://pypi.org/simple")



class MemoryOSAgent(BaseAgent):
    """
    MemoryOS接入AMemGym的Assistant实现

    关键发现：get_response内部已包含add_memory，会自动存储到记忆库
    """

    def __init__(self, config: Dict):
        """
        初始化MemoryOSAgent

        Args:
            config: 配置字典，需包含：
                - openai_api_key: OpenAI API密钥
                - data_storage_path: 数据存储路径（每个agent实例应独立）
                - user_id: 用户ID（默认amemgym_user）
                - assistant_id: 助手ID
                - llm_model: 模型名称（默认gpt-4o-mini）
                - openai_base_url: 可选，自定义API地址
                - embedding_model_name: 可选，嵌入模型（默认BAAI/bge-m3）
                - short_term_capacity: STM容量（默认7）
                - mid_term_capacity: MTM容量（默认2000）
                - long_term_knowledge_capacity: LTM知识容量（默认100）
                - retrieval_queue_capacity: 检索队列容量（默认7）
                - mid_term_heat_threshold: MTM热度阈值（默认5.0）
                - llm_config: AMemGym LLM配置
        """
        self.config = deepcopy(config)
        self.llm_config = self.config.get("llm_config", {})
        self.agent_config = self.config.get("agent_config", {})

        self.user_id = self.agent_config.get("user_id", f"amemgym_user_{id(self)}")
        self.assistant_id = self.agent_config.get("assistant_id", "amemgym_assistant")
        self.data_storage_path = self.agent_config.get("data_storage_path", f"./memoryos_data_{self.user_id}")

        self._ensure_unique_storage()
        self.memory = self._create_memory_instance()
        self.local_msgs: List[Dict] = []

    def _ensure_unique_storage(self):
        """确保存储路径唯一"""
        if os.path.exists(self.data_storage_path):
            import uuid
            self.data_storage_path = f"{self.data_storage_path}_{uuid.uuid4().hex[:8]}"
            self.agent_config["data_storage_path"] = self.data_storage_path

    def _create_memory_instance(self) -> Memoryos:
        """创建MemoryOS实例"""
        api_key = self.llm_config.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("未提供OpenAI API密钥")

        base_url = self.llm_config.get("base_url") or os.environ.get("OPENAI_BASE_URL")
        llm_model = self.agent_config.get("llm_model", self.llm_config.get("llm_model", "gpt-4.1-mini"))

        return Memoryos(
            user_id=self.user_id,
            openai_api_key=api_key,
            openai_base_url=base_url,
            data_storage_path=self.data_storage_path,
            assistant_id=self.assistant_id,
            llm_model=llm_model,
            short_term_capacity=self.agent_config.get("short_term_capacity", 7),
            mid_term_capacity=self.agent_config.get("mid_term_capacity", 2000),
            long_term_knowledge_capacity=self.agent_config.get("long_term_knowledge_capacity", 100),
            retrieval_queue_capacity=self.agent_config.get("retrieval_queue_capacity", 7),
            mid_term_heat_threshold=self.agent_config.get("mid_term_heat_threshold", 5.0),
            embedding_model_name=self.agent_config.get("embedding_model_name", "BAAI/bge-m3")
        )

    def reset(self):
        """重置Assistant状态"""
        self.local_msgs = []
        if os.path.exists(self.data_storage_path):
            shutil.rmtree(self.data_storage_path, ignore_errors=True)

        import uuid
        self.user_id = f"amemgym_user_{uuid.uuid4().hex[:8]}"
        self.data_storage_path = f"./memoryos_data_{self.user_id}"
        self.agent_config["data_storage_path"] = self.data_storage_path
        self.memory = self._create_memory_instance()


    def act(self, obs: str) -> str:
        """
        处理用户输入并生成回复

        使用get_response：内部自动检索记忆 + 生成回复 + 存储到记忆库
        """
        # get_response自动完成：检索 → 生成 → 存储
        response = self.memory.get_response(
            query=obs,
            relationship_with_user="friend",
            style_hint=""
        )

        # 缓存到local_msgs（用于构建近期上下文）
        self.local_msgs.append({"role": "user", "content": obs})
        self.local_msgs.append({"role": "assistant", "content": response})

        # 注意：无需调用add_memory，get_response内部已存储

        return response

    def add_msgs(self, messages: List[Dict]):
        """
        批量添加消息到MemoryOS记忆库

        用于AMemGym批量注入历史对话
        """
        if not messages:
            return

        # 缓存到本地
        self.local_msgs.extend(messages)

        # 显式调用add_memory（因为get_response只存储当前轮次）
        i = 0
        while i < len(messages):
            msg = messages[i]

            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                assistant_content = ""

                if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                    assistant_content = messages[i + 1].get("content", "")
                    i += 2
                else:
                    assistant_content = "[No response]"
                    i += 1

                try:
                    self.memory.add_memory(
                        user_input=user_content,
                        agent_response=assistant_content
                    )
                    logger.trace(f"Added to memoryOS: {user_content[:50]}...")
                except Exception as e:
                    logger.warning(f"Failed to add memory: {e}")
            else:
                i += 1

    def save_state(self, local_dir: str):
        """保存Assistant状态"""
        os.makedirs(local_dir, exist_ok=True)

        state = {
            "config": self.config,
            "llm_config": self.llm_config,
            "agent_config": self.agent_config,
            "user_id": self.user_id,
            "assistant_id": self.assistant_id,
            "data_storage_path": self.data_storage_path,
            "local_msgs": self.local_msgs
        }

        with open(os.path.join(local_dir, "memoryos_state.json"), "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

        backup_path = os.path.join(local_dir, "memoryos_data_backup")
        if os.path.exists(self.data_storage_path):
            if os.path.exists(backup_path):
                shutil.rmtree(backup_path)
            shutil.copytree(self.data_storage_path, backup_path, dirs_exist_ok=True)

    def load_state(self, local_dir: str):
        """加载Assistant状态"""
        state_path = os.path.join(local_dir, "memoryos_state.json")
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)

        self.config = state["config"]
        self.llm_config = state["llm_config"]
        self.agent_config = state["agent_config"]
        self.user_id = state["user_id"]
        self.assistant_id = state["assistant_id"]
        self.data_storage_path = state["data_storage_path"]
        self.local_msgs = state["local_msgs"]

        backup_path = os.path.join(local_dir, "memoryos_data_backup")
        if os.path.exists(backup_path):
            if os.path.exists(self.data_storage_path):
                shutil.rmtree(self.data_storage_path, ignore_errors=True)
            shutil.copytree(backup_path, self.data_storage_path, dirs_exist_ok=True)

        self.memory = self._create_memory_instance()

    def answer_question(self, question: str):
        """
        基于记忆回答问题

        注意：get_response会自动存储到记忆库，无法避免
        这是MemoryOS的设计限制
        """
        # 使用get_response，接受其自动存储的行为
        response = self.memory.get_response(
            query=question,
            relationship_with_user="friend",
            style_hint=""
        )

        # 不缓存到local_msgs（与act区分）

        return response
