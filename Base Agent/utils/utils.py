#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility/Helper Module

Provides shared functionality such as common error classes and helper routines
for the entire multi-agent system.
"""

import logging
import yaml
from dataclasses import dataclass

# Configure basic logging for utilities module if needed
logger = logging.getLogger(__name__)

# --- Custom Error Classes ---

class BaseAgentError(Exception):
    """Base exception class for agent-related errors."""
    pass

class TransientError(BaseAgentError):
    """
    Represents temporary errors that might be resolved upon retry.
    Examples: Network timeouts, temporary API unavailability.
    """
    pass

class PermanentError(BaseAgentError):
    """
    Represents errors that are unlikely to be resolved by retrying.
    Examples: Invalid credentials, malformed requests, non-existent resources.
    """
    pass

class ConfigurationError(PermanentError):
    """Error related to loading or validating agent configuration."""
    pass

class MemoryError(BaseAgentError):
    """Error related to memory storage or retrieval."""
    pass

class LLMError(BaseAgentError):
    """Error related to interactions with the Language Model."""
    pass


# --- Helper Functions (Example: Config Loading) ---

@dataclass
class AgentConfigSchema:
    """
    A basic placeholder for the AgentConfig schema.
    This will likely be defined more concretely in agent_config.py
    and potentially loaded/validated here or there.
    """
    agent_id: str
    model_type: str = "default_model"
    max_tokens: int = 1000
    temperature: float = 0.7
    memory_type: str = "in_memory"
    memory_path: str | None = None # Path for file-based, connection string for DB, etc.
    log_level: str = "INFO"
    # Add other fields like tool/permission settings as needed


def load_config(config_path: str) -> AgentConfigSchema:
    """
    Loads agent configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        An AgentConfigSchema object populated with configuration values.

    Raises:
        ConfigurationError: If the file cannot be read or parsed, or if validation fails.
        FileNotFoundError: If the config file does not exist.
    """
    logger.info(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        if not config_data:
            raise ConfigurationError(f"Configuration file is empty or invalid: {config_path}")

        # Basic validation (more robust validation can be added)
        required_fields = ['agent_id'] # Add other mandatory fields
        for field in required_fields:
            if field not in config_data:
                raise ConfigurationError(f"Missing required configuration field '{field}' in {config_path}")

        # Create AgentConfigSchema object (adjust as per actual schema definition)
        # This assumes direct mapping; add more complex parsing/validation if needed
        config = AgentConfigSchema(**config_data)
        logger.info(f"Configuration loaded successfully for agent: {config.agent_id}")
        return config

    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file {config_path}: {e}")
        raise ConfigurationError(f"Error parsing YAML configuration file {config_path}: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error loading configuration from {config_path}: {e}")
        raise ConfigurationError(f"Unexpected error loading configuration from {config_path}: {e}") from e

# Add other utility functions as needed (e.g., logging setup helpers, data formatters)
