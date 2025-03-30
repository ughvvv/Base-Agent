#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit Tests for the BaseAgent Module.

Focuses on testing the BaseAgent initialization, lifecycle method wrappers,
and integration points with mocked dependencies (MemoryManager, LLMManager).
"""

import unittest
import os
import yaml
import logging
from unittest.mock import patch, MagicMock, mock_open

# Modules to test and mock
from agents.base_agent import BaseAgent
from agents.agent_config import AgentConfigSchema
from memory_manager.memory_manager import MemoryManager, InMemoryMemoryManager
from llm_manager.llm_manager import LLMManager, StubLLMManager
from utils.utils import ConfigurationError, MemoryError, LLMError, BaseAgentError

# Disable logging noise during tests
logging.disable(logging.CRITICAL)

# --- Test Fixtures ---

# Example valid config data
VALID_CONFIG_DATA = {
    'agent_id': 'test_agent_001',
    'description': 'Test agent',
    'model_type': 'stub', # Use stub for testing
    'model_parameters': {'temperature': 0.5},
    'memory_type': 'in_memory',
    'memory_settings': {},
    'log_level': 'DEBUG',
    'max_retries': 2,
    'retry_delay': 0.1
}

# Example invalid config data (missing agent_id)
INVALID_CONFIG_DATA = {
    'description': 'Invalid agent',
    'model_type': 'stub',
}

# --- Helper: A Concrete Agent for Testing ---

class ConcreteTestAgent(BaseAgent):
    """A minimal concrete implementation of BaseAgent for testing purposes."""
    def process(self, input_data: str) -> str:
        """Simple process implementation for testing."""
        self.log_event("test_process_start", {"input": input_data})
        
        # Simulate interaction with memory and LLM
        context = self.fetch_context("some_key")
        prompt = f"Input: {input_data}, Context: {context}"
        llm_response = self.run_llm(prompt)
        
        result = f"Processed: {input_data} with response: {llm_response.get('response', 'N/A')}"
        self.store_result("output_key", result)
        
        self.log_event("test_process_end", {"output": result})
        return result

# --- Test Cases ---

class TestBaseAgentInitialization(unittest.TestCase):
    """Tests the initialization (__init__) of the BaseAgent."""

    def setUp(self):
        # Create a dummy config file path
        self.test_config_path = "temp_test_config.yaml"

    def tearDown(self):
        # Clean up dummy config file
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)

    def _write_config(self, data):
        with open(self.test_config_path, 'w') as f:
            yaml.dump(data, f)

    @patch('utils.utils.load_config')
    @patch('memory_manager.memory_manager.get_memory_manager')
    @patch('llm_manager.llm_manager.get_llm_manager')
    def test_successful_initialization(self, mock_get_llm, mock_get_mem, mock_load_config):
        """Test successful agent initialization with valid config."""
        self._write_config(VALID_CONFIG_DATA)
        
        # Configure mocks
        mock_config_obj = AgentConfigSchema(**VALID_CONFIG_DATA)
        mock_load_config.return_value = mock_config_obj
        mock_mem_manager = MagicMock(spec=MemoryManager)
        mock_get_mem.return_value = mock_mem_manager
        mock_llm_manager = MagicMock(spec=LLMManager)
        mock_get_llm.return_value = mock_llm_manager

        # Instantiate the agent
        agent = ConcreteTestAgent(config_path=self.test_config_path)

        # Assertions
        self.assertIsInstance(agent.config, AgentConfigSchema)
        self.assertEqual(agent.agent_id, VALID_CONFIG_DATA['agent_id'])
        self.assertEqual(agent.memory_manager, mock_mem_manager)
        self.assertEqual(agent.llm_manager, mock_llm_manager)
        mock_load_config.assert_called_once_with(self.test_config_path)
        mock_get_mem.assert_called_once_with(mock_config_obj)
        mock_get_llm.assert_called_once_with(mock_config_obj)
        # Could also mock and check log_event calls if needed

    @patch('utils.utils.load_config', side_effect=ConfigurationError("Bad config"))
    def test_initialization_fails_on_config_load_error(self, mock_load_config):
        """Test init fails if config loading raises ConfigurationError."""
        self._write_config(INVALID_CONFIG_DATA) # Content doesn't matter due to mock
        with self.assertRaises(ConfigurationError):
            ConcreteTestAgent(config_path=self.test_config_path)
        mock_load_config.assert_called_once_with(self.test_config_path)

    @patch('utils.utils.load_config')
    @patch('memory_manager.memory_manager.get_memory_manager', side_effect=MemoryError("Mem init failed"))
    def test_initialization_fails_on_memory_error(self, mock_get_mem, mock_load_config):
        """Test init fails if get_memory_manager raises MemoryError."""
        self._write_config(VALID_CONFIG_DATA)
        mock_config_obj = AgentConfigSchema(**VALID_CONFIG_DATA)
        mock_load_config.return_value = mock_config_obj

        with self.assertRaises(MemoryError):
            ConcreteTestAgent(config_path=self.test_config_path)
        mock_get_mem.assert_called_once_with(mock_config_obj)

    @patch('utils.utils.load_config')
    @patch('memory_manager.memory_manager.get_memory_manager')
    @patch('llm_manager.llm_manager.get_llm_manager', side_effect=LLMError("LLM init failed"))
    def test_initialization_fails_on_llm_error(self, mock_get_llm, mock_get_mem, mock_load_config):
        """Test init fails if get_llm_manager raises LLMError."""
        self._write_config(VALID_CONFIG_DATA)
        mock_config_obj = AgentConfigSchema(**VALID_CONFIG_DATA)
        mock_load_config.return_value = mock_config_obj
        mock_get_mem.return_value = MagicMock(spec=MemoryManager) # Memory init succeeds

        with self.assertRaises(LLMError):
            ConcreteTestAgent(config_path=self.test_config_path)
        mock_get_llm.assert_called_once_with(mock_config_obj)


class TestBaseAgentLifecycleMethods(unittest.TestCase):
    """Tests the wrapper methods (fetch_context, run_llm, store_result, run)."""

    @patch('utils.utils.load_config')
    @patch('memory_manager.memory_manager.get_memory_manager')
    @patch('llm_manager.llm_manager.get_llm_manager')
    def setUp(self, mock_get_llm, mock_get_mem, mock_load_config):
        """Set up a test agent instance with mocked dependencies."""
        # Use real config object but mocked managers
        self.mock_config = AgentConfigSchema(**VALID_CONFIG_DATA)
        mock_load_config.return_value = self.mock_config

        self.mock_memory_manager = MagicMock(spec=MemoryManager)
        mock_get_mem.return_value = self.mock_memory_manager

        self.mock_llm_manager = MagicMock(spec=LLMManager)
        mock_get_llm.return_value = self.mock_llm_manager

        # Patch log_event to avoid actual logging/JSON formatting issues in tests
        with patch.object(ConcreteTestAgent, 'log_event', return_value=None) as self.mock_log_event:
             # Write dummy config file needed for init path
             self.test_config_path = "temp_test_config_lifecycle.yaml"
             with open(self.test_config_path, 'w') as f:
                 yaml.dump(VALID_CONFIG_DATA, f)
             self.agent = ConcreteTestAgent(config_path=self.test_config_path)
        
        # Clean up dummy file after setup patch context exits
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)


    def test_fetch_context_success(self):
        """Test successful fetch_context call."""
        expected_data = {"some": "data"}
        self.mock_memory_manager.read_memory.return_value = expected_data
        
        result = self.agent.fetch_context("my_key")
        
        self.assertEqual(result, expected_data)
        self.mock_memory_manager.read_memory.assert_called_once_with(self.agent.agent_id, "my_key")
        # Check log_event calls (example)
        self.mock_log_event.assert_any_call("fetch_context_start", {"key": "my_key"})
        self.mock_log_event.assert_any_call("fetch_context_success", {"key": "my_key", "found": True})

    def test_fetch_context_not_found(self):
        """Test fetch_context when key is not found."""
        self.mock_memory_manager.read_memory.return_value = None
        
        result = self.agent.fetch_context("missing_key")
        
        self.assertIsNone(result)
        self.mock_memory_manager.read_memory.assert_called_once_with(self.agent.agent_id, "missing_key")
        self.mock_log_event.assert_any_call("fetch_context_success", {"key": "missing_key", "found": False})

    def test_fetch_context_raises_error(self):
        """Test fetch_context when memory manager raises MemoryError."""
        self.mock_memory_manager.read_memory.side_effect = MemoryError("Read failed")
        
        with self.assertRaises(MemoryError):
            self.agent.fetch_context("error_key")
            
        self.mock_memory_manager.read_memory.assert_called_once_with(self.agent.agent_id, "error_key")
        self.mock_log_event.assert_any_call("fetch_context_failed", {"key": "error_key", "error": "Read failed"})

    def test_run_llm_success(self):
        """Test successful run_llm call."""
        expected_response = {"response": "LLM says hi"}
        self.mock_llm_manager.call_llm.return_value = expected_response
        
        result = self.agent.run_llm("Hello LLM")
        
        self.assertEqual(result, expected_response)
        self.mock_llm_manager.call_llm.assert_called_once_with("Hello LLM", model_id=None, override_params=None)
        self.mock_log_event.assert_any_call("run_llm_start", {"prompt_length": 9, "model_id": "stub"})
        self.mock_log_event.assert_any_call("run_llm_success", {"model_id": "stub"})

    def test_run_llm_raises_error(self):
        """Test run_llm when LLM manager raises LLMError."""
        self.mock_llm_manager.call_llm.side_effect = LLMError("API call failed")
        
        with self.assertRaises(LLMError):
            self.agent.run_llm("Error prompt")
            
        self.mock_llm_manager.call_llm.assert_called_once_with("Error prompt", model_id=None, override_params=None)
        self.mock_log_event.assert_any_call("run_llm_failed", {"model_id": "stub", "error_type": "LLMError", "error_message": "LLM call failed: API call failed"})


    def test_store_result_success(self):
        """Test successful store_result call."""
        data_to_store = {"result": "done"}
        self.agent.store_result("output_data", data_to_store)
        
        self.mock_memory_manager.write_memory.assert_called_once_with(self.agent.agent_id, "output_data", data_to_store)
        self.mock_log_event.assert_any_call("store_result_start", {"key": "output_data", "value_type": "dict"})
        self.mock_log_event.assert_any_call("store_result_success", {"key": "output_data"})

    def test_store_result_raises_error(self):
        """Test store_result when memory manager raises MemoryError."""
        self.mock_memory_manager.write_memory.side_effect = MemoryError("Write failed")
        
        with self.assertRaises(MemoryError):
            self.agent.store_result("error_key", "some value")
            
        self.mock_memory_manager.write_memory.assert_called_once_with(self.agent.agent_id, "error_key", "some value")
        self.mock_log_event.assert_any_call("store_result_failed", {"key": "error_key", "error": "Write failed"})

    def test_run_method_success(self):
        """Test the main run method calling the process method successfully."""
        input_data = "Test Input"
        # Mock underlying calls made by ConcreteTestAgent.process
        self.mock_memory_manager.read_memory.return_value = "Mock Context"
        self.mock_llm_manager.call_llm.return_value = {"response": "Mock LLM Response"}
        
        expected_output = "Processed: Test Input with response: Mock LLM Response"
        result = self.agent.run(input_data)

        self.assertEqual(result, expected_output)
        # Verify mocks were called as expected by the process method
        self.mock_memory_manager.read_memory.assert_called_once_with(self.agent.agent_id, "some_key")
        self.mock_llm_manager.call_llm.assert_called_once()
        self.mock_memory_manager.write_memory.assert_called_once_with(self.agent.agent_id, "output_key", expected_output)
        # Check run-level logs
        self.mock_log_event.assert_any_call("agent_run_start", {"input_type": "str"})
        self.mock_log_event.assert_any_call("agent_run_complete", {"result_type": "str"})

    @patch.object(ConcreteTestAgent, 'process', side_effect=ValueError("Process failed"))
    def test_run_method_catches_process_error(self, mock_process):
        """Test that the run method catches and logs errors from process."""
        input_data = "Trigger Error"
        
        with self.assertRaises(BaseAgentError) as cm:
            self.agent.run(input_data)
        
        self.assertIn("Unexpected error during agent run: Process failed", str(cm.exception))
        mock_process.assert_called_once_with(input_data)
        self.mock_log_event.assert_any_call("agent_run_failed", {"error_type": "UnexpectedError", "error_message": "Process failed"})


if __name__ == '__main__':
    unittest.main()
