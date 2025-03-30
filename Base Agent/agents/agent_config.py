#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Agent Configuration Module

Defines the configuration schema for agents and provides mechanisms
for loading and validating agent configurations.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Assuming utils contains the load_config function and error classes
# If load_config is moved here, adjust imports accordingly.
from utils.utils import ConfigurationError # Keep error class import

logger = logging.getLogger(__name__)

@dataclass
class AgentConfigSchema:
    """
    Defines the structure for agent configuration.

    Attributes:
        agent_id: Unique identifier for the agent.
        description: Optional description of the agent's purpose.
        model_type: Type or identifier of the language model to use (e.g., 'gpt-4', 'claude-3').
        model_parameters: Dictionary of parameters for the LLM (e.g., max_tokens, temperature).
        memory_type: Type of memory backend ('in_memory', 'file_based', 'database').
        memory_settings: Configuration specific to the chosen memory type
                         (e.g., file path for 'file_based', connection string for 'database').
        log_level: Logging level for the agent (e.g., 'INFO', 'DEBUG').
        tools: List of tools or capabilities the agent is permitted to use.
        permissions: Specific permissions granted to the agent.
        max_retries: Maximum number of retries for transient errors.
        retry_delay: Base delay (in seconds) for exponential backoff.
    """
    agent_id: str
    description: Optional[str] = None
    model_type: str = "default_model"
    model_parameters: Dict[str, Any] = field(default_factory=lambda: {"max_tokens": 1000, "temperature": 0.7})
    memory_type: str = "in_memory"
    memory_settings: Dict[str, Any] = field(default_factory=dict) # e.g., {"file_path": "/path/to/memory.json"}
    log_level: str = "INFO"
    tools: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    max_retries: int = 3
    retry_delay: float = 1.0 # seconds

    def __post_init__(self):
        """Perform basic validation after initialization."""
        if not self.agent_id:
            raise ConfigurationError("agent_id cannot be empty.")
        if self.memory_type == "file_based" and not self.memory_settings.get("file_path"):
            raise ConfigurationError("memory_settings must include 'file_path' for 'file_based' memory type.")
        # Add more validation rules as needed (e.g., for database settings)
        logger.debug(f"AgentConfigSchema validated for agent_id: {self.agent_id}")

# The load_config function remains in utils.py for now,
# but it should return an instance of *this* AgentConfigSchema.
# If more complex validation or loading logic specific to AgentConfig
# is needed, it could be implemented here.

# Example of how it might be used (assuming load_config is in utils):
# from utils.utils import load_config
# try:
#     config = load_config('config/agent_config.yaml')
#     print(f"Loaded config for agent: {config.agent_id}")
# except (FileNotFoundError, ConfigurationError) as e:
#     print(f"Error loading configuration: {e}")
