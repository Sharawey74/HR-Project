"""
Agent 4 Factory
Selects between Direct HTTP (fast) or LangChain (advanced) implementation
"""
from typing import Union, Optional
import logging

from ..core.config import get_config

logger = logging.getLogger(__name__)


def get_explainer_agent(
    use_langchain: Optional[bool] = None,
    config=None
):
    """
    Factory function to get appropriate Agent 4 implementation
    
    Args:
        use_langchain: Force LangChain mode (None = use config default)
        config: Application config
    
    Returns:
        Agent instance (Direct HTTP or LangChain)
    
    Mode Selection Logic:
    - If use_langchain=True: Try LangChain, fallback to Direct HTTP
    - If use_langchain=False: Use Direct HTTP only
    - If use_langchain=None: Use config.llm.use_langchain setting
    """
    from .agent4_llm_explainer import LLMExplainerAgent
    
    config = config or get_config()
    
    # Determine mode from parameter or config
    if use_langchain is None:
        use_langchain = getattr(config.llm, 'use_langchain', False)
    
    # Try LangChain mode if requested
    if use_langchain:
        try:
            from .agent4_langchain_explainer import LangChainExplainerAgent
            logger.info("🔗 Initializing LangChain Explainer (advanced mode)")
            agent = LangChainExplainerAgent(config)
            
            if agent.llm_available:
                logger.info("✅ LangChain Explainer ready")
                return agent
            else:
                logger.warning("⚠️ LangChain unavailable, falling back to Direct HTTP")
                
        except ImportError as e:
            logger.warning(f"⚠️ LangChain not installed: {e}")
            logger.info("💡 Install with: pip install langchain-ollama")
        except Exception as e:
            logger.warning(f"⚠️ LangChain init failed: {e}")
    
    # Default to Direct HTTP mode
    logger.info("⚡ Using Direct HTTP Explainer (fast mode)")
    return LLMExplainerAgent(config)
