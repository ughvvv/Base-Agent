#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base Agent Module

Defines the common interface (the "skeleton") for all specialized agents.
Includes core lifecycle methods and integrates with configuration, memory,
LLM management, and central logging.
"""

import abc
import logging
import json # For structured logging if using standard logger directly
from typing import Any, Dict, Optional

# Import necessary components from other modules
from agents.agent_config import AgentConfigSchema
from memory_manager.memory_manager import MemoryManager, get_memory_manager
from llm_manager.llm_manager import LLMManager, get_llm_manager
from utils.utils import BaseAgentError, ConfigurationError, MemoryError, LLMError # Import relevant errors

# Configure logger for this module
logger = logging.getLogger(__name__)

class BaseAgent(abc.ABC):
    """
    Abstract Base Class for all specialized agents.

    Provides a common structure including lifecycle methods, configuration handling,
    memory management, LLM interaction, and logging integration.

    Subclasses must implement the `process` method at a minimum, and potentially
    override other lifecycle methods for specialized behavior.
    """

    def __init__(self, config_path: str):
        """
        Initializes the BaseAgent.

        Args:
            config_path: Path to the agent's configuration file (e.g., YAML).

        Raises:
            ConfigurationError: If the configuration cannot be loaded or is invalid.
            MemoryError: If the memory manager cannot be initialized.
            LLMError: If the LLM manager cannot be initialized.
        """
        self.config: AgentConfigSchema
        self.memory_manager: MemoryManager
        self.llm_manager: LLMManager
        self.agent_id: str

        try:
            # 1. Load Configuration
            # Assuming load_config is in utils, adjust if moved
            from utils.utils import load_config
            self.config = load_config(config_path)
            self.agent_id = self.config.agent_id
            self._configure_logging() # Configure logging based on loaded config

            self.log_event("agent_initialization_start", {"config_path": config_path})

            # 2. Initialize Memory Manager
            self.memory_manager = get_memory_manager(self.config)

            # 3. Initialize LLM Manager
            self.llm_manager = get_llm_manager(self.config)

            self.log_event("agent_initialization_complete", {"agent_id": self.agent_id, "memory_type": self.config.memory_type, "llm_type": self.config.model_type})

        except (ConfigurationError, MemoryError, LLMError) as e:
            logger.exception(f"Failed to initialize agent {getattr(self.config, 'agent_id', 'UNKNOWN')} due to: {e}")
            # Log the specific error type and message before re-raising
            self.log_event("agent_initialization_failed", {"error_type": type(e).__name__, "error_message": str(e)})
            raise # Re-raise the specific error
        except Exception as e:
            # Catch any other unexpected initialization errors
            logger.exception(f"Unexpected error during agent initialization: {e}")
            self.log_event("agent_initialization_failed", {"error_type": "UnexpectedError", "error_message": str(e)})
            raise ConfigurationError(f"Unexpected error during agent initialization: {e}") from e


    def _configure_logging(self):
        """
        Configures the agent's specific logger based on the loaded configuration.
        This is a basic setup; integrate with the central JSON logging utility here.
        """
        agent_logger = logging.getLogger(f"agent.{self.agent_id}")
        try:
            level = getattr(logging, self.config.log_level.upper(), logging.INFO)
            agent_logger.setLevel(level)
            # In a real scenario, you'd configure handlers here, potentially
            # using the central logging framework's setup function.
            # For now, ensure the root logger has a basic handler if needed for output.
            if not logging.getLogger().hasHandlers():
                 logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            logger.info(f"Configured logger for agent '{self.agent_id}' with level {self.config.log_level.upper()}")
        except Exception as e:
            logger.error(f"Failed to configure logging for agent '{self.agent_id}': {e}. Using default.")
            # Fallback or default configuration


    def log_event(self, event_type: str, details: Dict[str, Any]):
        """
        Logs a structured event using the central logging framework.

        This method should be adapted to use the specific JSON logging utility
        provided in the Agent Onboarding Guide. For now, it uses the standard
        logger with a structured format attempt.

        Args:
            event_type: A string identifying the type of event (e.g., 'task_start', 'llm_call').
            details: A dictionary containing relevant details about the event.
        """
        agent_logger = logging.getLogger(f"agent.{self.agent_id}")
        log_payload = {
            "event_type": event_type,
            "agent_id": self.agent_id,
            "details": details,
            # Add other standard fields required by the central logger (timestamp, level etc.)
            # The central logging utility might handle these automatically.
        }
        # Use the appropriate log level based on event type or severity
        # Defaulting to INFO for now. Error events should use .error() or .exception()
        agent_logger.info(json.dumps(log_payload)) # Basic JSON logging


    # --- Agent Lifecycle Methods ---

    def fetch_context(self, key: str) -> Optional[Any]:
        """
        Fetches necessary context or data from the Memory Manager.
        Wrapper around memory_manager.read_memory with logging.

        Args:
            key: The key for the memory item to retrieve.

        Returns:
            The retrieved data or None if not found.

        Raises:
            MemoryError: If reading from memory fails.
        """
        self.log_event("fetch_context_start", {"key": key})
        try:
            context = self.memory_manager.read_memory(self.agent_id, key)
            self.log_event("fetch_context_success", {"key": key, "found": context is not None})
            return context
        except MemoryError as e:
            self.log_event("fetch_context_failed", {"key": key, "error": str(e)})
            logger.error(f"Failed to fetch context for key '{key}': {e}")
            raise # Re-raise the original error

    def run_llm(self, prompt: str, model_id: Optional[str] = None, override_params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Runs a prompt through the LLM Manager, handling retries and logging.
        Wrapper around llm_manager.call_llm.

        Args:
            prompt: The prompt to send to the LLM.
            model_id: Optional specific model ID to use.
            override_params: Optional parameters to override LLM defaults for this call.

        Returns:
            The response from the LLM.

        Raises:
            LLMError: If the LLM call fails after retries or encounters a permanent error.
        """
        self.log_event("run_llm_start", {"prompt_length": len(prompt), "model_id": model_id or self.config.model_type})
        try:
            response = self.llm_manager.call_llm(prompt, model_id=model_id, override_params=override_params)
            # Log success, potentially with response metadata if available/safe
            self.log_event("run_llm_success", {"model_id": model_id or self.config.model_type})
            return response
        except (LLMError, PermanentError) as e: # Catch specific errors from LLM manager
            self.log_event("run_llm_failed", {"model_id": model_id or self.config.model_type, "error_type": type(e).__name__, "error_message": str(e)})
            logger.error(f"LLM call failed: {e}")
            raise LLMError(f"LLM call failed: {e}") from e # Wrap in LLMError if not already


    @abc.abstractmethod
    def process(self, input_data: Any) -> Any:
        """
        The main logic method for the agent. Must be implemented by subclasses.

        This method orchestrates the agent's task, potentially calling
        fetch_context, run_llm, and store_result.

        Args:
            input_data: The primary input required for the agent's task.

        Returns:
            The result of the agent's processing.
        """
        self.log_event("process_start", {"input_type": type(input_data).__name__})
        # Subclass implementation goes here
        pass

    def store_result(self, key: str, value: Any) -> None:
        """
        Stores results or state back into the Memory Manager.
        Wrapper around memory_manager.write_memory with logging.

        Args:
            key: The key under which to store the value.
            value: The data to store.

        Raises:
            MemoryError: If writing to memory fails.
        """
        self.log_event("store_result_start", {"key": key, "value_type": type(value).__name__})
        try:
            self.memory_manager.write_memory(self.agent_id, key, value)
            self.log_event("store_result_success", {"key": key})
        except MemoryError as e:
            self.log_event("store_result_failed", {"key": key, "error": str(e)})
            logger.error(f"Failed to store result for key '{key}': {e}")
            raise # Re-raise the original error


    def run(self, input_data: Any) -> Any:
        """
        Executes the main processing logic of the agent.
        This is a convenience method that calls the abstract `process` method.
        """
        self.log_event("agent_run_start", {"input_type": type(input_data).__name__})
        try:
            result = self.process(input_data)
            self.log_event("agent_run_complete", {"result_type": type(result).__name__})
            return result
        except BaseAgentError as e: # Catch known agent errors
             self.log_event("agent_run_failed", {"error_type": type(e).__name__, "error_message": str(e)})
             logger.exception(f"Agent {self.agent_id} failed during run: {e}")
             raise # Re-raise agent-specific errors
        except Exception as e: # Catch unexpected errors during processing
             self.log_event("agent_run_failed", {"error_type": "UnexpectedError", "error_message": str(e)})
             logger.exception(f"Agent {self.agent_id} encountered an unexpected error during run: {e}")
             raise BaseAgentError(f"Unexpected error during agent run: {e}") from e


# Example of how a subclass might look (in a separate file ideally):
# class MySpecialAgent(BaseAgent):
#     def process(self, input_data: str) -> str:
#         """Processes a string input."""
#         self.log_event("my_special_process_start", {"input_string": input_data})
#
#         # 1. Fetch some context
#         previous_result = self.fetch_context("last_run_output")
#         prompt = f"Input: {input_data}\nPrevious Result: {previous_result}\nGenerate next step:"
#
#         # 2. Call LLM
#         llm_response = self.run_llm(prompt)
#         processed_output = llm_response.get("response", "Error: No response text") # Adapt based on actual LLM response structure
#
#         # 3. Store result
#         self.store_result("last_run_output", processed_output)
#         self.store_result("latest_input", input_data)
#
#         self.log_event("my_special_process_complete", {"output_length": len(processed_output)})
#         return processed_output
