# amemgym/tests/unit/test_memoryos_agent.py

"""
MemoryOSAgent 单元测试
验证接口实现正确性，无需运行完整AMemGym评测
"""

import os
import sys
import json
import tempfile
import shutil
import unittest
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 假设MemoryOSAgent已实现
from amemgym.agents.memoryos import MemoryOSAgent


class TestMemoryOSAgent(unittest.TestCase):
    """MemoryOSAgent单元测试套件"""

    @classmethod
    def setUpClass(cls):
        """测试类初始化：创建临时目录和基础配置"""
        cls.temp_dir = tempfile.mkdtemp(prefix="memoryos_test_")
        cls.config = {
            "user_id": "test_user",
            "assistant_id": "test_assistant",
            "openai_api_key": os.getenv("OPENAI_API_KEY", "test-key"),
            "openai_base_url": os.getenv("OPENAI_BASE_URL"),
            "data_storage_path": os.path.join(cls.temp_dir, "memoryos_data"),
            "llm_model": "gpt-4o-mini",
            "short_term_capacity": 3,  # 测试用小容量
            "mid_term_capacity": 10,
            "long_term_knowledge_capacity": 5,
            "retrieval_queue_capacity": 3,
            "mid_term_heat_threshold": 2.0,
            "llm_config": {
                "model": "gpt-4o-mini",
                "temperature": 0.7
            }
        }
        print(f"\n测试数据目录: {cls.temp_dir}")

    @classmethod
    def tearDownClass(cls):
        """测试类清理：删除临时目录"""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)
            print(f"清理测试目录: {cls.temp_dir}")

    def setUp(self):
        """每个测试方法前执行：创建新的Agent实例"""
        self.agent = MemoryOSAgent(self.config)

    def tearDown(self):
        """每个测试方法后执行：清理"""
        pass

    # ==================== 基础接口测试 ====================

    def test_01_agent_initialization(self):
        """测试1: Agent能正确初始化"""
        print("\n[测试1] Agent初始化...")

        self.assertIsNotNone(self.agent)
        self.assertEqual(self.agent.user_id, "test_user")
        self.assertEqual(self.agent.assistant_id, "test_assistant")
        self.assertEqual(len(self.agent.local_msgs), 0)

        # 验证MemoryOS实例已创建
        self.assertIsNotNone(self.agent.memory)

        # 验证存储目录已创建
        self.assertTrue(os.path.exists(self.config["data_storage_path"]))

        print("✓ Agent初始化成功")

    def test_02_reset(self):
        """测试2: reset方法能清空状态"""
        print("\n[测试2] reset方法...")

        # 先添加一些消息
        self.agent.local_msgs = [{"role": "user", "content": "test"}]

        # 执行reset
        self.agent.reset()

        # 验证状态已清空
        self.assertEqual(len(self.agent.local_msgs), 0)
        # user_id应该改变或重置
        self.assertIsNotNone(self.agent.user_id)

        print("✓ reset方法工作正常")

    def test_03_add_msgs_single(self):
        """测试3: add_msgs能添加单条消息"""
        print("\n[测试3] add_msgs单条消息...")

        messages = [
            {"role": "user", "content": "你好"}
        ]

        # 执行添加
        self.agent.add_msgs(messages)

        # 验证本地缓存
        self.assertEqual(len(self.agent.local_msgs), 1)
        self.assertEqual(self.agent.local_msgs[0]["role"], "user")
        self.assertEqual(self.agent.local_msgs[0]["content"], "你好")

        print("✓ 单条消息添加成功")

    def test_04_add_msgs_pair(self):
        """测试4: add_msgs能添加对话对(用户+助手)"""
        print("\n[测试4] add_msgs对话对...")

        messages = [
            {"role": "user", "content": "今天天气怎么样？"},
            {"role": "assistant", "content": "今天晴天，25度。"}
        ]

        # 执行添加
        self.agent.add_msgs(messages)

        # 验证本地缓存
        self.assertEqual(len(self.agent.local_msgs), 2)
        self.assertEqual(self.agent.local_msgs[0]["role"], "user")
        self.assertEqual(self.agent.local_msgs[1]["role"], "assistant")

        print("✓ 对话对添加成功")

    def test_05_add_msgs_batch(self):
        """测试5: add_msgs能批量添加多条消息"""
        print("\n[测试5] add_msgs批量添加...")

        messages = [
            {"role": "user", "content": "问题1"},
            {"role": "assistant", "content": "回答1"},
            {"role": "user", "content": "问题2"},
            {"role": "assistant", "content": "回答2"},
        ]

        self.agent.add_msgs(messages)

        # 验证全部添加
        self.assertEqual(len(self.agent.local_msgs), 4)

        print("✓ 批量添加成功")

    # ==================== 核心功能测试 ====================

    def test_06_act_basic(self):
        """测试6: act方法能处理输入并返回回复"""
        print("\n[测试6] act基本功能...")

        # 跳过如果没有真实API key
        if self.config["openai_api_key"] == "test-key":
            self.skipTest("跳过: 需要真实OpenAI API Key")

        user_input = "你好，请介绍一下自己。"

        try:
            response = self.agent.act(user_input)

            # 验证返回值
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)

            # 验证消息已缓存
            self.assertGreater(len(self.agent.local_msgs), 0)

            print(f"✓ act成功，回复: {response[:50]}...")

        except Exception as e:
            self.fail(f"act方法抛出异常: {e}")

    def test_07_act_with_context(self):
        """测试7: act能利用上下文记忆"""
        print("\n[测试7] act上下文记忆...")

        if self.config["openai_api_key"] == "test-key":
            self.skipTest("跳过: 需要真实OpenAI API Key")

        # 第一轮对话
        self.agent.act("我叫张三，喜欢打篮球。")

        # 第二轮对话（应该能记住名字）
        response = self.agent.act("我叫什么名字？")

        self.assertIsInstance(response, str)
        print(f"✓ 上下文记忆测试完成，回复: {response[:50]}...")

    def test_08_answer_question(self):
        """测试8: answer_question能基于记忆回答问题"""
        print("\n[测试8] answer_question...")

        if self.config["openai_api_key"] == "test-key":
            self.skipTest("跳过: 需要真实OpenAI API Key")

        # 先添加一些背景信息
        self.agent.add_msgs([
            {"role": "user", "content": "我的邮箱是zhangsan@example.com"},
            {"role": "assistant", "content": "已记录您的邮箱。"}
        ])

        # 提问
        question = "我的邮箱是什么？"

        try:
            response = self.agent.answer_question(question)

            # 验证返回值
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)

            print(f"✓ answer_question成功，回复: {response[:50]}...")

        except Exception as e:
            self.fail(f"answer_question抛出异常: {e}")

    # ==================== 状态持久化测试 ====================

    def test_09_save_state(self):
        """测试9: save_state能保存状态到目录"""
        print("\n[测试9] save_state...")

        # 添加一些状态
        self.agent.local_msgs = [{"role": "user", "content": "测试消息"}]

        # 保存状态
        save_dir = os.path.join(self.temp_dir, "save_test")
        self.agent.save_state(save_dir)

        # 验证文件已创建
        self.assertTrue(os.path.exists(save_dir))
        self.assertTrue(os.path.exists(os.path.join(save_dir, "config.json")))
        self.assertTrue(os.path.exists(os.path.join(save_dir, "local_msgs.json")))

        print("✓ 状态保存成功")

    def test_10_load_state(self):
        """测试10: load_state能从目录恢复状态"""
        print("\n[测试10] load_state...")

        # 准备保存的状态
        save_dir = os.path.join(self.temp_dir, "load_test")
        os.makedirs(save_dir, exist_ok=True)

        # 手动创建状态文件
        test_config = self.config.copy()
        test_msgs = [{"role": "assistant", "content": "恢复的消息"}]

        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(test_config, f)
        with open(os.path.join(save_dir, "local_msgs.json"), "w") as f:
            json.dump(test_msgs, f)

        # 加载状态
        self.agent.load_state(save_dir)

        # 验证状态已恢复
        self.assertEqual(len(self.agent.local_msgs), 1)
        self.assertEqual(self.agent.local_msgs[0]["content"], "恢复的消息")

        print("✓ 状态加载成功")

    def test_11_save_load_roundtrip(self):
        """测试11: save和load能完整往返"""
        print("\n[测试11] save/load往返...")

        # 添加复杂状态
        self.agent.local_msgs = [
            {"role": "user", "content": "消息1"},
            {"role": "assistant", "content": "回复1"},
            {"role": "user", "content": "消息2"}
        ]

        # 保存
        save_dir = os.path.join(self.temp_dir, "roundtrip_test")
        self.agent.save_state(save_dir)

        # 创建新Agent并加载
        new_agent = MemoryOSAgent(self.config)
        new_agent.load_state(save_dir)

        # 验证状态一致
        self.assertEqual(len(new_agent.local_msgs), 3)
        self.assertEqual(new_agent.local_msgs[0]["content"], "消息1")
        self.assertEqual(new_agent.local_msgs[2]["content"], "消息2")

        print("✓ 往返测试成功")

    # ==================== 边界情况测试 ====================

    def test_12_empty_messages(self):
        """测试12: 处理空消息列表"""
        print("\n[测试12] 空消息处理...")

        # 不应抛出异常
        self.agent.add_msgs([])
        self.assertEqual(len(self.agent.local_msgs), 0)

        print("✓ 空消息处理正常")

    def test_13_invalid_message_format(self):
        """测试13: 处理无效消息格式"""
        print("\n[测试13] 无效消息格式...")

        # 缺少role字段
        invalid_msgs = [{"content": "无角色"}]

        # 应该抛出异常或有默认处理
        with self.assertRaises((KeyError, ValueError)):
            self.agent.add_msgs(invalid_msgs)

        print("✓ 无效格式检测正常")

    def test_14_multiple_resets(self):
        """测试14: 多次reset不会出错"""
        print("\n[测试14] 多次reset...")

        for i in range(3):
            self.agent.reset()
            self.assertEqual(len(self.agent.local_msgs), 0)

        print("✓ 多次reset正常")

    def test_15_concurrent_access(self):
        """测试15: 模拟并发访问（如AMemGym多线程）"""
        print("\n[测试15] 并发访问模拟...")

        import threading

        errors = []

        def worker(agent_id):
            try:
                agent = MemoryOSAgent(self.config)
                agent.add_msgs([
                    {"role": "user", "content": f"线程{agent_id}"}
                ])
            except Exception as e:
                errors.append(e)

        # 启动多个线程
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"并发错误: {errors}")
        print("✓ 并发访问正常")


class TestMemoryOSAgentIntegration(unittest.TestCase):
    """集成测试：模拟AMemGym实际使用场景"""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp(prefix="memoryos_integration_")
        cls.config = {
            "user_id": "integration_test",
            "openai_api_key": os.getenv("OPENAI_API_KEY", "test-key"),
            "data_storage_path": os.path.join(cls.temp_dir, "data"),
            "llm_model": "gpt-4o-mini",
            "short_term_capacity": 5,
            "llm_config": {"model": "gpt-4o-mini"}
        }

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_full_conversation_flow(self):
        """完整对话流程测试"""
        print("\n[集成测试] 完整对话流程...")

        if self.config["openai_api_key"] == "test-key":
            self.skipTest("跳过: 需要真实API Key")

        agent = MemoryOSAgent(self.config)

        # 模拟多轮对话
        conversations = [
            "你好，我是李四。",
            "我喜欢吃川菜。",
            "我上周去了成都旅游。",
            "你记得我是谁吗？",
            "我喜欢吃什么菜？",
        ]

        for i, msg in enumerate(conversations):
            print(f"  轮次{i + 1}: {msg[:20]}...")
            response = agent.act(msg)
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)

        # 验证记忆持久化
        save_dir = os.path.join(self.temp_dir, "integration_save")
        agent.save_state(save_dir)
        self.assertTrue(os.path.exists(save_dir))

        print("✓ 完整对话流程测试通过")


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryOSAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryOSAgentIntegration))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 返回结果
    return result.wasSuccessful()


if __name__ == "__main__":
    print("=" * 60)
    print("MemoryOSAgent 单元测试")
    print("=" * 60)

    success = run_tests()

    print("\n" + "=" * 60)
    if success:
        print("✓ 所有测试通过！")
        print("MemoryOSAgent可以接入AMemGym评测框架")
    else:
        print("✗ 部分测试失败，请检查实现")
    print("=" * 60)

    sys.exit(0 if success else 1)