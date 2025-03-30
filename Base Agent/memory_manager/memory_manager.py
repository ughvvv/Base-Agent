#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Memory Manager Module

Abstracts the storage and retrieval of context data for agents.
Provides a common interface and basic implementations (in-memory, file-based).
Designed for future extension (e.g., database-backed storage).
"""

import abc
import json
import logging
import os
from typing import Any, Dict, Optional

# Assuming utils contains error classes
from utils.utils import MemoryError, PermanentError, ConfigurationError

logger = logging.getLogger(__name__)

class MemoryManager(abc.ABC):
    """
    Abstract Base Class for Memory Managers.

    Defines the interface for storing and retrieving agent context/memory.
    Subclasses must implement the read_memory and write_memory methods.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the Memory Manager.

        Args:
            config: Configuration settings specific to the memory manager implementation
                    (e.g., file path, connection string).
        """
        self.config = config
        logger.info(f"Initializing {self.__class__.__name__} with config: {config}")

    @abc.abstractmethod
    def read_memory(self, agent_id: str, key: str) -> Optional[Any]:
        """
        Reads a specific piece of memory/context for a given agent.

        Args:
            agent_id: The unique identifier of the agent.
            key: The key identifying the piece of memory to retrieve.

        Returns:
            The retrieved memory data, or None if the key is not found.

        Raises:
            MemoryError: If there's an issue reading from the memory store.
        """
        pass

    @abc.abstractmethod
    def write_memory(self, agent_id: str, key: str, value: Any) -> None:
        """
        Writes a specific piece of memory/context for a given agent.

        Args:
            agent_id: The unique identifier of the agent.
            key: The key identifying the piece of memory to store.
            value: The data to store.

        Raises:
            MemoryError: If there's an issue writing to the memory store.
        """
        pass

    @abc.abstractmethod
    def load_agent_memory(self, agent_id: str) -> Dict[str, Any]:
        """
        Loads the entire memory state for a given agent.

        Args:
            agent_id: The unique identifier of the agent.

        Returns:
            A dictionary representing the agent's complete memory state.
            Returns an empty dictionary if no memory exists for the agent.

        Raises:
            MemoryError: If there's an issue loading the memory state.
        """
        pass

    @abc.abstractmethod
    def save_agent_memory(self, agent_id: str, memory_state: Dict[str, Any]) -> None:
        """
        Saves the entire memory state for a given agent.

        Args:
            agent_id: The unique identifier of the agent.
            memory_state: A dictionary representing the agent's complete memory state.

        Raises:
            MemoryError: If there's an issue saving the memory state.
        """
        pass


# --- Implementations ---

class InMemoryMemoryManager(MemoryManager):
    """
    An in-memory implementation of the Memory Manager.
    Stores data in a Python dictionary. Data is lost when the process ends.
    Suitable for testing or ephemeral agents.
    """
    _memory_store: Dict[str, Dict[str, Any]] = {} # agent_id -> {key: value}

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # No specific config needed for in-memory, but accept it for consistency

    def read_memory(self, agent_id: str, key: str) -> Optional[Any]:
        logger.debug(f"Reading memory for agent '{agent_id}', key '{key}' (in-memory)")
        agent_memory = self._memory_store.get(agent_id, {})
        return agent_memory.get(key)

    def write_memory(self, agent_id: str, key: str, value: Any) -> None:
        logger.debug(f"Writing memory for agent '{agent_id}', key '{key}' (in-memory)")
        if agent_id not in self._memory_store:
            self._memory_store[agent_id] = {}
        self._memory_store[agent_id][key] = value

    def load_agent_memory(self, agent_id: str) -> Dict[str, Any]:
        logger.debug(f"Loading entire memory for agent '{agent_id}' (in-memory)")
        return self._memory_store.get(agent_id, {}).copy() # Return a copy

    def save_agent_memory(self, agent_id: str, memory_state: Dict[str, Any]) -> None:
        logger.debug(f"Saving entire memory for agent '{agent_id}' (in-memory)")
        self._memory_store[agent_id] = memory_state.copy() # Store a copy


class FileBasedMemoryManager(MemoryManager):
    """
    A file-based implementation of the Memory Manager.
    Stores each agent's memory as a JSON file.
    Requires 'file_path' in the configuration, which acts as a template.
    The agent_id will be substituted into the path if '{agent_id}' is present.
    If not, it assumes a single file path for all agents (less common).
    """
    _file_path_template: str

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if "file_path" not in config:
            raise ConfigurationError("FileBasedMemoryManager requires 'file_path' in configuration.")
        self._file_path_template = config["file_path"]
        # Ensure the directory exists if the template implies directories
        try:
            # Attempt to create the directory structure implied by the template
            # This handles cases like "memlog/{agent_id}/memory.json"
            potential_dir = os.path.dirname(self._file_path_template.format(agent_id="dummy"))
            if potential_dir: # Only create if dirname is not empty
                os.makedirs(potential_dir, exist_ok=True)
                logger.info(f"Ensured directory exists: {potential_dir}")
        except Exception as e:
            # Log error but don't fail init; file access errors handled later
            logger.warning(f"Could not pre-create directory structure for {self._file_path_template}: {e}")


    def _get_agent_file_path(self, agent_id: str) -> str:
        """Resolves the file path for a specific agent."""
        try:
            return self._file_path_template.format(agent_id=agent_id)
        except KeyError:
            # If {agent_id} placeholder is missing, use the template directly
            # This might mean one file for all agents, which could cause issues
            logger.warning(f"File path template '{self._file_path_template}' does not contain '{{agent_id}}'. Using path directly.")
            return self._file_path_template

    def _read_file(self, file_path: str) -> Dict[str, Any]:
        """Reads and parses the JSON data from a file."""
        if not os.path.exists(file_path):
            return {} # Return empty dict if file doesn't exist
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                if not content.strip(): # Handle empty file
                    return {}
                data = json.loads(content)
                if not isinstance(data, dict):
                    raise MemoryError(f"Invalid format in memory file {file_path}: expected a JSON object (dict).")
                return data
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from memory file {file_path}: {e}")
            raise MemoryError(f"Error decoding JSON from memory file {file_path}: {e}") from e
        except IOError as e:
            logger.error(f"I/O error reading memory file {file_path}: {e}")
            raise MemoryError(f"I/O error reading memory file {file_path}: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error reading memory file {file_path}: {e}")
            raise MemoryError(f"Unexpected error reading memory file {file_path}: {e}") from e

    def _write_file(self, file_path: str, data: Dict[str, Any]) -> None:
        """Writes JSON data to a file."""
        try:
            # Ensure directory exists before writing
            dir_name = os.path.dirname(file_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4) # Use indent for readability
        except IOError as e:
            logger.error(f"I/O error writing memory file {file_path}: {e}")
            raise MemoryError(f"I/O error writing memory file {file_path}: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error writing memory file {file_path}: {e}")
            raise MemoryError(f"Unexpected error writing memory file {file_path}: {e}") from e

    def read_memory(self, agent_id: str, key: str) -> Optional[Any]:
        file_path = self._get_agent_file_path(agent_id)
        logger.debug(f"Reading memory for agent '{agent_id}', key '{key}' from file: {file_path}")
        try:
            agent_memory = self._read_file(file_path)
            return agent_memory.get(key)
        except MemoryError as e:
            # Log and re-raise specific memory errors
            logger.error(f"Failed to read memory for agent '{agent_id}', key '{key}': {e}")
            raise

    def write_memory(self, agent_id: str, key: str, value: Any) -> None:
        file_path = self._get_agent_file_path(agent_id)
        logger.debug(f"Writing memory for agent '{agent_id}', key '{key}' to file: {file_path}")
        try:
            # Read-modify-write to avoid overwriting other keys
            agent_memory = self._read_file(file_path)
            agent_memory[key] = value
            self._write_file(file_path, agent_memory)
        except MemoryError as e:
            logger.error(f"Failed to write memory for agent '{agent_id}', key '{key}': {e}")
            raise

    def load_agent_memory(self, agent_id: str) -> Dict[str, Any]:
        file_path = self._get_agent_file_path(agent_id)
        logger.debug(f"Loading entire memory for agent '{agent_id}' from file: {file_path}")
        try:
            return self._read_file(file_path)
        except MemoryError as e:
            logger.error(f"Failed to load memory state for agent '{agent_id}': {e}")
            raise

    def save_agent_memory(self, agent_id: str, memory_state: Dict[str, Any]) -> None:
        file_path = self._get_agent_file_path(agent_id)
        logger.debug(f"Saving entire memory for agent '{agent_id}' to file: {file_path}")
        try:
            self._write_file(file_path, memory_state)
        except MemoryError as e:
            logger.error(f"Failed to save memory state for agent '{agent_id}': {e}")
            raise

# --- Factory Function (Optional but Recommended) ---

def get_memory_manager(config: 'AgentConfigSchema') -> MemoryManager:
    """
    Factory function to create a MemoryManager instance based on configuration.

    Args:
        config: The AgentConfigSchema object containing memory settings.

    Returns:
        An instance of a MemoryManager subclass.

    Raises:
        ConfigurationError: If the memory_type is unsupported or configuration is invalid.
    """
    memory_type = config.memory_type
    memory_settings = config.memory_settings

    logger.info(f"Creating memory manager of type: {memory_type}")

    if memory_type == "in_memory":
        return InMemoryMemoryManager(memory_settings)
    elif memory_type == "file_based":
        return FileBasedMemoryManager(memory_settings)
    # elif memory_type == "database":
        # from .database_manager import DatabaseMemoryManager # Example
        # return DatabaseMemoryManager(memory_settings)
    else:
        raise ConfigurationError(f"Unsupported memory_type: {memory_type}")
