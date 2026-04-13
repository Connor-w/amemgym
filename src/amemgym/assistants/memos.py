from amemgym.utils import call_llm
from loguru import logger
import os
import json
import shutil
import requests
from copy import deepcopy
from backoff import on_exception, expo
from typing import List, Dict, Any, Optional
from .base import BaseAgent


@on_exception(expo, Exception, max_tries=10)
def insert_memos_memory(api_base: str, messages: List[Dict], user_id: str,
                        mem_cube_id: str, async_mode: str = "sync"):
    """
    Insert messages into MemOS memory via REST API.
    MemOS自动处理记忆提取，无需手动控制infer参数。
    """
    url = f"{api_base}/product/add"

    data = {
        "user_id": user_id,
        "mem_cube_id": mem_cube_id,
        "messages": messages,
        "async_mode": async_mode
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=data, timeout=30)
    response.raise_for_status()

    result = response.json()
    logger.trace(f"MemOS add response: {result}")
    return result


def format_memos_memories(memories_data: Dict) -> str:
    """
    格式化MemOS返回的记忆结构。
    MemOS返回结构包含 text_mem, act_mem, para_mem 等多类型记忆。
    """
    if not memories_data or "data" not in memories_data:
        return ""

    data = memories_data["data"]
    all_memories = []

    # 处理文本记忆 (text_mem)
    if "text_mem" in data and data["text_mem"]:
        for cube_mem in data["text_mem"]:
            if "memories" in cube_mem:
                for mem in cube_mem["memories"]:
                    memory_text = mem.get("memory", "")
                    if memory_text:
                        all_memories.append(memory_text)

    # 处理行为记忆 (act_mem)
    if "act_mem" in data and data["act_mem"]:
        for cube_mem in data["act_mem"]:
            if "memories" in cube_mem:
                for mem in cube_mem["memories"]:
                    memory_text = mem.get("memory", "")
                    if memory_text:
                        all_memories.append(f"[Action] {memory_text}")

    # 按时间排序（如果有时间戳）
    # MemOS返回的记忆通常已按相关性排序

    if not all_memories:
        return ""

    return "\n".join([f"- {m}" for m in all_memories])


class MemOSAgent(BaseAgent):
    """
    MemOS记忆层接入AMemGym评测框架。

    特性：
    1. 支持Self-Hosted和Cloud两种模式
    2. MemOS自动控制记忆提取策略，无需外部infer参数
    3. 混合搜索：向量+图+全文检索
    4. act()方法会更新记忆，answer_question()只检索不更新
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = deepcopy(config)
        self.mode = self.config["memory_config"].get("mode", "self_hosted")
        self.api_base = self.config["memory_config"]["api_base"]
        self.user_id = self.config["memory_config"]["user_id"]
        self.mem_cube_id = self.config["memory_config"]["mem_cube_id"]
        self.async_mode = self.config["memory_config"].get("async_mode", "sync")

        # Cloud模式需要API Key
        if self.mode == "cloud":
            self.api_key = self.config["memory_config"].get("api_key") or os.getenv("MEMOS_API_KEY")
        else:
            self.api_key = None

        # 本地消息缓存（短期记忆）
        self.local_msgs: List[Dict] = []
        self.reset()

    def reset(self):
        """重置Agent状态，清空本地缓存但不删除MemOS中的长期记忆"""
        self.local_msgs = []
        # 注意：MemOS的长期记忆是持久的，reset()不清除历史记忆

    def _search_memories(self, query: str, limit: int = 10) -> Dict:
        """
        调用MemOS搜索API检索相关记忆。
        """
        url = f"{self.api_base}/product/search"

        data = {
            "query": query,
            "user_id": self.user_id,
            "mem_cube_id": self.mem_cube_id,
            "limit": limit
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"MemOS search failed: {e}")
            return {"data": {"text_mem": [], "act_mem": [], "para_mem": []}}

    def _build_system_prompt(self, memories_str: str) -> str:
        """构建包含检索到的记忆的system prompt"""
        return (
            "You are a helpful AI assistant with long-term memory. "
            "Use the following retrieved memories to personalize your response. "
            "If the memories are irrelevant to the current query, you may ignore them.\n\n"
            f"Retrieved user memories:\n{memories_str if memories_str else 'No relevant memories found.'}"
        )

    def act(self, obs: str) -> str:
        """
        执行动作并更新记忆。
        这是AMemGym交互循环的主要入口：检索记忆 -> 生成回复 -> 保存到记忆库。

        关于自主控制：MemOS的add接口会自动处理记忆提取，无需外部控制infer选项。
        """
        new_msg = {"role": "user", "content": obs}

        # 1. 检索相关记忆（长期记忆）
        relevant_memories = self._search_memories(
            query=obs,
            limit=self.config["agent_config"]["top_k"]
        )
        memories_str = format_memos_memories(relevant_memories)
        logger.trace(f"Retrieved memories: {memories_str[:200]}...")

        # 2. 构建消息列表（system + 短期记忆 + 当前输入）
        system_prompt = self._build_system_prompt(memories_str)
        messages = [{"role": "system", "content": system_prompt}] + self.local_msgs + [new_msg]

        # 3. 调用LLM生成回复
        response = call_llm(messages, self.config["llm_config"])
        new_response = {"role": "assistant", "content": response}

        # 4. 记录当前交互到本地缓存并触发记忆更新
        self.add_msgs(messages=[new_msg, new_response])

        return response

    def answer_question(self, question: str) -> tuple:
        """
        回答问题（不更新记忆）。
        用于评测阶段：只检索现有记忆，不写入新记忆。

        Returns:
            tuple: (response, token_usage)
        """
        new_msg = {"role": "user", "content": question}

        # 1. 检索记忆（与act相同）
        relevant_memories = self._search_memories(
            query=question,
            limit=self.config["agent_config"]["top_k"]
        )
        memories_str = format_memos_memories(relevant_memories)
        logger.trace(f"Retrieved memories for Q&A: {memories_str[:200]}...")

        # 2. 构建消息
        system_prompt = self._build_system_prompt(memories_str)
        messages = [{"role": "system", "content": system_prompt}] + self.local_msgs + [new_msg]

        # 3. 调用LLM（返回token使用情况用于评测分析）
        return call_llm(messages, self.config["llm_config"], return_token_usage=True)

    def add_msgs(self, messages: List[Dict]):
        """
        添加消息到记忆系统。
        当本地缓存达到阈值时，批量写入MemOS长期记忆。

        MemOS会自动：
        - 提取关键事实和偏好
        - 去重和更新现有记忆
        - 建立记忆间的图关系
        """
        assert len(messages) == 2, "Only support two-turn interactions in one batch"

        limit = (self.config["agent_config"]["update_bsz"] +
                 self.config["agent_config"]["local_length"])

        self.local_msgs += messages

        # 当本地消息超过限制时，批量更新到MemOS
        if len(self.local_msgs) >= limit:
            update_bsz = self.config["agent_config"]["update_bsz"]
            msgs_to_insert = self.local_msgs[:update_bsz]
            self.local_msgs = self.local_msgs[update_bsz:]

            logger.trace(f"Inserting {len(msgs_to_insert)} messages into MemOS")

            # 格式化消息以便MemOS更好地提取记忆
            formatted_msgs = []
            for msg in msgs_to_insert:
                if msg["role"] == "user":
                    formatted_msgs.append({
                        "role": "user",
                        "content": f"USER INPUT: {msg['content']}"
                    })
                elif msg["role"] == "assistant":
                    formatted_msgs.append({
                        "role": "assistant",
                        "content": f"ASSISTANT RESPONSE: {msg['content']}"
                    })
                else:
                    raise ValueError(f"Unknown message role: {msg['role']}")

            # 调用MemOS API
            insert_memos_memory(
                api_base=self.api_base,
                messages=formatted_msgs,
                user_id=self.user_id,
                mem_cube_id=self.mem_cube_id,
                async_mode=self.async_mode
            )

    def load_state(self, local_dir: str):
        """
        从本地目录加载Agent状态。
        MemOS的长期记忆是持久的，这里主要恢复本地短期缓存。
        """
        # 恢复本地消息历史
        msg_path = os.path.join(local_dir, "msg_history.json")
        if os.path.exists(msg_path):
            with open(msg_path, "r", encoding="utf-8") as f:
                self.local_msgs = json.load(f)

        # 注意：MemOS的长期记忆存储在服务端，无需本地恢复
        logger.info(f"Loaded state from {local_dir}, local_msgs: {len(self.local_msgs)}")

    def save_state(self, local_dir: str):
        """
        保存Agent状态到本地目录。
        """
        os.makedirs(local_dir, exist_ok=True)

        # 保存本地消息历史
        msg_path = os.path.join(local_dir, "msg_history.json")
        with open(msg_path, "w", encoding="utf-8") as f:
            json.dump(self.local_msgs, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved state to {local_dir}")

    def get_all_memories(self) -> List[str]:
        """
        获取用户的所有记忆（用于调试和分析）。
        """
        # MemOS没有直接的"get all" API，可以通过空查询或特定查询获取
        # 这里使用一个通用查询来获取尽可能多的记忆
        result = self._search_memories(query="*", limit=100)

        memories = []
        data = result.get("data", {})
        for mem_type in ["text_mem", "act_mem", "para_mem"]:
            if mem_type in data:
                for cube_mem in data[mem_type]:
                    if "memories" in cube_mem:
                        for mem in cube_mem["memories"]:
                            memories.append(mem.get("memory", ""))

        return [m for m in memories if m]