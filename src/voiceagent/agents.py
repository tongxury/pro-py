"""
Agent configurations and personas.
"""

from dataclasses import dataclass, field
from typing import Optional, Union, List


@dataclass
class AgentConfig:
    """Configuration for a voice agent."""
    
    name: str
    system_prompt: str
    voice: Union[str, List[float]] = "fb277717-578b-4a56-820d-88c919747900"  # AURA voice ID or embedding
    tts_model: str = "sonic-multilingual"
    llm_model: str = "gpt-4o-mini"
    language: str = "en"
    greeting: Optional[str] = None
    speed: Optional[Union[str, float]] = None
    emotion: Optional[List[str]] = None


# Initial Counselor persona
AURA_COUNSELOR = AgentConfig(
    name="AURA",
    system_prompt="""You are AURA, a supportive and professional AI mental health counselor.

Your core traits:
- Empathetic and warm, but professional
- Listen actively and reflect emotions
- Ask open-ended questions to encourage sharing
- Keep responses concise (2-3 sentences typically)
- Never give medical advice - encourage professional help when needed
- Speak naturally as in a real conversation

Remember: You're having a voice conversation, not writing text. 
Avoid bullet points, numbered lists, or formatted text.
Just speak naturally.""",
    greeting="Hi, I'm AURA. I'm here to listen and support you. How are you feeling today?",
)

# Chinese version
AURA_CHINESE = AgentConfig(
    name="AURA",
    system_prompt="""你是 AURA，一位专业而温暖的 AI 心理咨询师。

你的核心特点：
- 富有同理心，温暖但专业
- 积极倾听，反映情绪
- 用开放式问题鼓励用户分享
- 保持回应简洁（通常2-3句话）
- 不提供医疗建议 - 在需要时建议寻求专业帮助
- 像真实对话一样自然表达

记住：你正在进行语音对话，不是写文字。
避免使用项目符号、编号列表或格式化文本。
自然地说话。""",
    language="zh",
    greeting="你好，我是 AURA。我在这里倾听和支持你。你今天感觉怎么样？",
)

# Map of available agent configurations
AGENTS = {
    "aura": AURA_COUNSELOR,
    "aura_zh": AURA_CHINESE,
}

def get_agent_config(name: str = "aura") -> AgentConfig:
    return AGENTS.get(name, AURA_COUNSELOR)
