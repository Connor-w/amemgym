# amemgym/agents/memoryos.py

import os
import json
import shutil
from typing import List, Dict, Optional
from .base import BaseAgent
from amemgym.utils import call_llm

# 尝试导入MemoryOS
try:
    from memoryos import Memoryos
except ImportError:
    raise ImportError("MemoryOS未安装，请运行: pip install MemoryOS-BaiJia")


class MemoryOSAgent(BaseAgent):
    """
    MemoryOS接入AMemGym的Agent实现

    MemoryOS核心方法：
    - add_memory(user_input, agent_response): 添加对话记忆
    - get_response(query, **kwargs): 检索记忆并生成回复
    - retrieve_memory(query): 仅检索记忆（不生成回复）- 如有此方法
    - get_user_profile_summary(): 获取用户画像
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

        self.config = config
        self.user_id = config.get("user_id", f"amemgym_user_{id(self)}")
        self.assistant_id = config.get("assistant_id", "amemgym_assistant")
        self.data_storage_path = config.get("data_storage_path", f"./memoryos_data_{self.user_id}")

        # 确保每个agent实例有独立的存储路径（用于隔离）
        self._ensure_unique_storage()

        # 初始化MemoryOS
        self.memory = self._create_memory_instance()

        # 本地消息缓存（用于批量更新）
        self.local_msgs: List[Dict] = []

    def _ensure_unique_storage(self):
        """确保存储路径唯一，避免不同agent实例间数据混淆"""
        # 如果路径已存在，添加随机后缀
        if os.path.exists(self.data_storage_path):
            import uuid
            self.data_storage_path = f"{self.data_storage_path}_{uuid.uuid4().hex[:8]}"
            # 更新config以便save_state使用
            self.config["data_storage_path"] = self.data_storage_path

    def _create_memory_instance(self) -> Memoryos:
        """创建MemoryOS实例"""
        return Memoryos(
            user_id=self.user_id,
            openai_api_key=self.config["openai_api_key"],
            openai_base_url=self.config.get("openai_base_url"),
            data_storage_path=self.data_storage_path,
            assistant_id=self.assistant_id,
            llm_model=self.config.get("llm_model", "gpt-4o-mini"),
            short_term_capacity=self.config.get("short_term_capacity", 7),
            mid_term_capacity=self.config.get("mid_term_capacity", 2000),
            long_term_knowledge_capacity=self.config.get("long_term_knowledge_capacity", 100),
            retrieval_queue_capacity=self.config.get("retrieval_queue_capacity", 7),
            mid_term_heat_threshold=self.config.get("mid_term_heat_threshold", 5.0),
            embedding_model_name=self.config.get("embedding_model_name", "BAAI/bge-m3")
        )

    def reset(self):
        """
        重置Agent状态

        清理：
        1. 本地消息缓存
        2. 存储目录（彻底删除记忆文件）
        3. 重新初始化MemoryOS实例
        """
        # 清理本地缓存
        self.local_msgs = []

        # 删除存储目录（彻底重置）
        if os.path.exists(self.data_storage_path):
            shutil.rmtree(self.data_storage_path, ignore_errors=True)

        # 创建新的唯一路径
        import uuid
        self.user_id = f"amemgym_user_{uuid.uuid4().hex[:8]}"
        self.data_storage_path = f"./memoryos_data_{self.user_id}"
        self.config["data_storage_path"] = self.data_storage_path
        self.config["user_id"] = self.user_id

        # 重新初始化
        self.memory = self._create_memory_instance()

    def act(self, obs: str) -> str:
        """
        处理用户输入并生成回复（核心交互接口）

        流程：
        1. 使用MemoryOS.get_response获取回复（内部自动检索记忆并生成）
        2. 将对话添加到本地缓存

        Args:
            obs: 用户输入

        Returns:
            助手回复
        """
        # 使用MemoryOS的get_response（内部包含记忆检索和回复生成）
        response = self.memory.get_response(
            query=obs,
            relationship_with_user="friend",
            style_hint=""
        )

        # 添加到本地缓存（用于后续批量处理或状态保存）
        self.local_msgs.append({"role": "user", "content": obs})
        self.local_msgs.append({"role": "assistant", "content": response})

        return response

    def add_msgs(self, messages: List[Dict]):
        """
        批量添加消息到MemoryOS

        注意：AMemGym可能传入任意数量消息，不一定是成对的。
        我们按顺序处理，每当遇到user消息且下一条是assistant时，组成一对添加。

        Args:
            messages: 消息列表，每个消息为{"role": "user"/"assistant", "content": "..."}
        """
        if not messages:
            return

        # 缓存到本地
        self.local_msgs.extend(messages)

        # 处理添加到MemoryOS
        i = 0
        while i < len(messages):
            msg = messages[i]

            if msg.get("role") == "user":
                # 查找对应的assistant回复（可能在后一条）
                assistant_content = ""
                if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                    assistant_content = messages[i + 1].get("content", "")
                    i += 2  # 跳过已处理的assistant消息
                else:
                    # 没有对应的assistant消息，使用占位符
                    assistant_content = "[No response]"
                    i += 1

                # 添加到MemoryOS
                try:
                    self.memory.add_memory(
                        user_input=msg.get("content", ""),
                        agent_response=assistant_content
                    )
                except Exception as e:
                    print(f"Warning: Failed to add memory: {e}")
            else:
                # assistant消息（没有前置user消息），跳过
                i += 1

    def save_state(self, local_dir: str):
        """
        保存Agent状态

        MemoryOS的数据已自动持久化到data_storage_path，
        这里只需保存配置和本地缓存，便于恢复。

        Args:
            local_dir: 保存目录
        """
        os.makedirs(local_dir, exist_ok=True)

        # 保存配置
        state = {
            "config": self.config,
            "user_id": self.user_id,
            "assistant_id": self.assistant_id,
            "data_storage_path": self.data_storage_path,
            "local_msgs": self.local_msgs
        }

        with open(os.path.join(local_dir, "memoryos_state.json"), "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

        # 可选：将MemoryOS数据目录打包备份
        import shutil
        backup_path = os.path.join(local_dir, "memoryos_data_backup")
        if os.path.exists(self.data_storage_path):
            shutil.copytree(self.data_storage_path, backup_path, dirs_exist_ok=True)

    def load_state(self, local_dir: str):
        """
        加载Agent状态

        Args:
            local_dir: 状态目录
        """
        # 加载状态文件
        state_path = os.path.join(local_dir, "memoryos_state.json")
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)

        # 恢复配置
        self.config = state["config"]
        self.user_id = state["user_id"]
        self.assistant_id = state["assistant_id"]
        self.data_storage_path = state["data_storage_path"]
        self.local_msgs = state["local_msgs"]

        # 恢复MemoryOS数据（如果有备份）
        backup_path = os.path.join(local_dir, "memoryos_data_backup")
        if os.path.exists(backup_path):
            if os.path.exists(self.data_storage_path):
                shutil.rmtree(self.data_storage_path, ignore_errors=True)
            shutil.copytree(backup_path, self.data_storage_path, dirs_exist_ok=True)

        # 重新初始化MemoryOS实例
        self.memory = self._create_memory_instance()

    def answer_question(self, question: str) -> str:
        """
        基于记忆回答问题（评测接口）

        重要：此方法应只读不写入，避免污染测试数据。

        流程：
        1. 使用MemoryOS.get_response获取回复
        2. 但需要注意：get_response会添加新记忆，我们需要在之后清理或使用替代方案

        由于MemoryOS没有纯粹的检索接口，我们使用get_response但记录状态，
        或者使用retrieve_memory（如果可用）。

        Args:
            question: 问题

        Returns:
            回答
        """
        # 方案1：如果MemoryOS有retrieve_memory方法，优先使用
        if hasattr(self.memory, 'retrieve_memory'):
            try:
                retrieved = self.memory.retrieve_memory(question)
                context = self._format_retrieved_memory(retrieved)
                messages = [
                    {"role": "system", "content": f"Based on memories:\n{context}"},
                    {"role": "user", "content": question}
                ]
                return call_llm(messages, self.config.get("llm_config", {}))
            except Exception:
                pass  # 失败则回退到方案2

        # 方案2：使用get_response（会添加记忆，但无法避免）
        # 注意：这会添加新记忆到MemoryOS，可能影响后续测试
        response = self.memory.get_response(
            query=question,
            relationship_with_user="friend",
            style_hint=""
        )
        return response

    def _format_retrieved_memory(self, retrieved) -> str:
        """格式化检索到的记忆"""
        if isinstance(retrieved, str):
            return retrieved
        elif isinstance(retrieved, list):
            return "\n".join(str(item) for item in retrieved)
        elif isinstance(retrieved, dict):
            return json.dumps(retrieved, ensure_ascii=False, indent=2)
        else:
            return str(retrieved)

    def get_user_profile(self) -> Optional[str]:
        """
        获取用户画像（MemoryOS特有功能）

        Returns:
            用户画像摘要，或None
        """
        try:
            return self.memory.get_user_profile_summary()
        except Exception as e:
            print(f"Warning: Failed to get user profile: {e}")
            return None