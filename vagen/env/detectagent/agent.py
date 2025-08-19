import numpy as np

class SimpleDetectAgent:
    """
    一个简单的示例 Agent，用于与 DetectAgentEnv 交互。

    这个 Agent 的逻辑非常基础，旨在演示如何将 Agent 与环境连接。
    在 M4Agent 的实际应用中，这部分将被一个复杂的、基于模型的 Agent 所取代，
    该 Agent 能够理解多模态输入（图像 + 文本提示）并生成动作。
    """
    def __init__(self, action_space):
        """
        初始化 Agent。

        Args:
            action_space: 环境的动作空间，Agent 需要从中采样或选择动作。
        """
        self.action_space = action_space

    def get_action(self, observation, prompt):
        """
        根据当前的观测（图像）和文本提示生成一个动作。

        在 M4Agent 中，这里将是模型的核心推理步骤。模型会接收
        `observation` 和 `prompt` 作为输入，并输出一个动作。

        Args:
            observation: 当前的环境观测（例如，一个图像）。
            prompt (str): 驱动 Agent 行为的文本提示。

        Returns:
            action: 一个在动作空间内的有效动作。
        """
        # 打印收到的信息，以模拟 Agent 的“思考”过程
        print(f"Agent 收到提示: '{prompt}'")
        print(f"Agent 正在观察一个形状为 {observation.shape} 的图像。")

        # 【待实现】这里应该是您复杂模型的推理逻辑。
        # 例如:
        # 1. 使用一个多模态模型处理图像和文本。
        # 2. 模型输出一个代表边界框和置信度的向量。
        # action = model.predict(observation, prompt)

        # 作为演示，我们返回一个随机动作。
        # 这模拟了一个完全未经训练的 Agent 的行为。
        random_action = self.action_space.sample()
        print(f"Agent 决定执行随机动作: {np.round(random_action, 2)}")
        
        return random_action

def create_agent(env):
    """
    一个工厂函数，用于创建和返回一个 Agent 实例。
    """
    return SimpleDetectAgent(env.action_space)
