#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM Manager Module

Centralizes interactions with Language Model APIs. Handles prompt submission,
response retrieval, and basic error handling with retry logic.
"""

import abc
import time
import logging
from typing import Any, Dict, Optional

# Assuming utils contains error classes and AgentConfigSchema definition is accessible
# Adjust import if AgentConfigSchema is elsewhere or needed directly
from utils.utils import LLMError, TransientError, PermanentError, ConfigurationError
# from agents.agent_config import AgentConfigSchema # Example if needed

logger = logging.getLogger(__name__)

class LLMManager(abc.ABC):
    """
    Abstract Base Class for Language Model Managers.

    Defines the interface for interacting with different LLM providers.
    Subclasses must implement the call_llm method.
    """

    def __init__(self, config: 'AgentConfigSchema'):
        """
        Initializes the LLM Manager.

        Args:
            config: The AgentConfigSchema object containing model parameters,
                    retry settings, and potentially API keys/endpoints (though
                    keys are often better handled via environment variables).
        """
        self.config = config
        self.model_type = config.model_type
        self.model_parameters = config.model_parameters
        self.max_retries = config.max_retries
        self.retry_delay = config.retry_delay
        logger.info(f"Initializing {self.__class__.__name__} for model type: {self.model_type}")
        # API keys should ideally be loaded securely, e.g., from environment variables
        # self.api_key = os.environ.get("LLM_API_KEY")
        # if not self.api_key:
        #     logger.warning("LLM_API_KEY environment variable not set.")
            # Depending on the LLM, this might be a fatal error or handled later

    @abc.abstractmethod
    def _call_api(self, prompt: str, specific_params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Internal method to make the actual API call to the specific LLM provider.
        Subclasses must implement this.

        Args:
            prompt: The input prompt string.
            specific_params: Optional dictionary of parameters to override or add
                             to the default model parameters for this specific call.

        Returns:
            The raw response from the LLM API.

        Raises:
            TransientError: For temporary issues like network errors, rate limits.
            PermanentError: For issues like invalid API keys, bad requests.
            LLMError: For other LLM-specific errors.
        """
        pass

    def call_llm(self, prompt: str, model_id: Optional[str] = None, override_params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Calls the configured Language Model with the given prompt, handling retries.

        Args:
            prompt: The input prompt string.
            model_id: Optional specific model ID to use, overriding the default from config.
                      (Note: The manager is typically initialized for one model type,
                       this allows flexibility if the provider supports multiple variants).
            override_params: Optional dictionary of parameters to override default
                             model parameters for this specific call (e.g., temperature).

        Returns:
            The processed response from the LLM.

        Raises:
            PermanentError: If a non-retryable error occurs after exhausting retries.
            LLMError: If an LLM-specific error occurs that isn't classified as Transient/Permanent.
        """
        current_model = model_id or self.model_type
        combined_params = self.model_parameters.copy()
        if override_params:
            combined_params.update(override_params)

        logger.info(f"Calling LLM (Model: {current_model}) with prompt (length: {len(prompt)} chars).")
        logger.debug(f"LLM Params: {combined_params}")

        retries = 0
        while retries <= self.max_retries:
            try:
                logger.debug(f"LLM call attempt {retries + 1}/{self.max_retries + 1}")
                # Prepare parameters for the specific API call
                api_params = combined_params.copy()
                # Add model_id if the underlying API needs it explicitly
                # api_params['model'] = current_model # Example

                response = self._call_api(prompt, specific_params=api_params)
                logger.info(f"LLM call successful on attempt {retries + 1}.")
                # Process/validate response if necessary
                return response

            except TransientError as e:
                logger.warning(f"Transient error during LLM call (attempt {retries + 1}): {e}")
                retries += 1
                if retries > self.max_retries:
                    logger.error(f"Max retries ({self.max_retries}) exceeded for transient error.")
                    raise PermanentError(f"LLM call failed after {self.max_retries} retries due to transient errors.") from e
                
                # Exponential backoff
                delay = self.retry_delay * (2 ** (retries - 1))
                logger.info(f"Retrying LLM call in {delay:.2f} seconds...")
                time.sleep(delay)

            except (PermanentError, LLMError) as e:
                logger.error(f"Permanent or unclassified LLM error during call: {e}")
                raise # Re-raise immediately, no retries for these

            except Exception as e:
                # Catch any other unexpected errors
                logger.exception(f"Unexpected error during LLM call: {e}")
                # Treat unexpected errors as potentially permanent to avoid infinite loops
                raise PermanentError(f"Unexpected error during LLM call: {e}") from e
        
        # This part should theoretically not be reached due to raises in the loop
        raise PermanentError("LLM call failed after exhausting all retries.")


# --- Example Stub Implementation (Does not call a real API) ---

class StubLLMManager(LLMManager):
    """
    A stub implementation of the LLM Manager for testing purposes.
    Does not make real API calls. Returns predefined responses or simulates errors.
    """
    def _call_api(self, prompt: str, specific_params: Optional[Dict[str, Any]] = None) -> Any:
        """Simulates an LLM API call."""
        logger.info(f"[STUB] Simulating LLM call for model '{self.model_type}' with prompt: '{prompt[:50]}...'")
        
        # --- Simulation Logic ---
        # You can add logic here to simulate different scenarios:
        # 1. Simulate success:
        response_text = f"This is a simulated response to the prompt: '{prompt[:30]}...'"
        logger.info("[STUB] Simulation successful.")
        return {"response": response_text, "model_used": self.model_type, "params": specific_params}

        # 2. Simulate a transient error sometimes:
        # import random
        # if random.random() < 0.5: # 50% chance of transient error
        #     logger.warning("[STUB] Simulating transient error (e.g., rate limit).")
        #     raise TransientError("Simulated rate limit exceeded")
        # else:
        #     # Success case as above
        #     response_text = f"This is a simulated response to the prompt: '{prompt[:30]}...'"
        #     logger.info("[STUB] Simulation successful.")
        #     return {"response": response_text, "model_used": self.model_type, "params": specific_params}

        # 3. Simulate a permanent error:
        # logger.error("[STUB] Simulating permanent error (e.g., invalid config).")
        # raise PermanentError("Simulated invalid API key or configuration")
        # ------------------------


# --- Factory Function (Optional but Recommended) ---

def get_llm_manager(config: 'AgentConfigSchema') -> LLMManager:
    """
    Factory function to create an LLMManager instance based on configuration.

    Args:
        config: The AgentConfigSchema object containing LLM settings.

    Returns:
        An instance of an LLMManager subclass.

    Raises:
        ConfigurationError: If the model_type is unsupported or configuration is invalid.
    """
    model_type = config.model_type
    logger.info(f"Creating LLM manager for model type: {model_type}")

    # Add real implementations here based on model_type
    if model_type == "stub" or "default_model" in model_type: # Use stub for default/testing
         logger.warning("Using StubLLMManager. No real API calls will be made.")
         return StubLLMManager(config)
    # elif "gpt-" in model_type:
    #     from .openai_manager import OpenAILLMManager # Example
    #     return OpenAILLMManager(config)
    # elif "claude-" in model_type:
    #     from .anthropic_manager import AnthropicLLMManager # Example
    #     return AnthropicLLMManager(config)
    else:
        raise ConfigurationError(f"Unsupported model_type for LLM Manager: {model_type}")
