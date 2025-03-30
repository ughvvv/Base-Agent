# Multi-Agent System - Phase 1 Design Document

## 1. Introduction

This document outlines the design and architecture for Phase 1 of the multi-agent system foundation. The primary goal of this phase is to establish a modular, scalable, and maintainable core structure that allows for the easy development and integration of specialized agents in the future.

The design emphasizes separation of concerns, clear interfaces between modules, and integration with existing central infrastructure for logging, metrics, and error handling, as detailed in the Agent Onboarding Guide.

## 2. Architecture Overview

The system is composed of several core modules, each with distinct responsibilities:

-   **BaseAgent:** Defines the fundamental structure and lifecycle for all agents.
-   **AgentConfig:** Manages agent-specific configurations.
-   **MemoryManager:** Abstracts data storage and retrieval for agent context/memory.
-   **LLMManager:** Centralizes interaction with Language Model APIs.
-   **Utils:** Provides shared utilities and common error definitions.
-   **Testing:** Contains unit and integration tests for ensuring module correctness and system stability.

### 2.1. Module Interaction Diagram

```mermaid
graph LR
    subgraph User/System Input
        direction LR
        InputData[Input Data]
    end

    subgraph Agent Core
        direction LR
        BA(BaseAgent) -- Reads --> AC(AgentConfig)
        BA -- Uses --> MM(MemoryManager)
        BA -- Uses --> LM(LLMManager)
        BA -- Uses --> U(Utils)
        BA -- Logs Events --> Logging[Central Logging]
        BA -- Emits Metrics --> Metrics[Central Metrics]
    end

    subgraph Supporting Modules
        direction LR
        AC -- Defines Schema --> ConfigFile(config/*.yaml)
        MM(MemoryManager) -- Manages --> MemoryStore[Memory Backend (In-Mem/File/DB)]
        LM(LLMManager) -- Calls --> LLM_API[External LLM API]
        U(Utils)
    end

    subgraph Infrastructure
        direction LR
        Logging
        Metrics
    end

    InputData --> BA
    BA --> OutputData[Processed Output]

    style BA fill:#f9f,stroke:#333,stroke-width:2px
    style AC fill:#ccf,stroke:#333,stroke-width:1px
    style MM fill:#cfc,stroke:#333,stroke-width:1px
    style LM fill:#ffc,stroke:#333,stroke-width:1px
    style U fill:#eee,stroke:#333,stroke-width:1px
```

*Diagram Explanation:*
-   User input triggers the `BaseAgent`.
-   `BaseAgent` loads its configuration via `AgentConfig`.
-   `BaseAgent` interacts with `MemoryManager` to fetch/store context and `LLMManager` to process prompts.
-   All modules utilize shared functions and error classes from `Utils`.
-   Key events and metrics are sent to the central logging and metrics systems.
-   `MemoryManager` interacts with the chosen backend (file, memory, etc.).
-   `LLMManager` interacts with the external LLM service.

## 3. Module Details

### 3.1. BaseAgent (`agents/base_agent.py`)

-   **Purpose:** Provides the abstract base class for all agents. Enforces a common lifecycle and integrates core services.
-   **Key Methods:**
    -   `__init__(config_path)`: Loads config, initializes Memory and LLM managers, sets up logging.
    -   `fetch_context(key)`: Retrieves data from memory via MemoryManager.
    -   `run_llm(prompt, ...)`: Executes prompts via LLMManager, handles retries.
    -   `process(input_data)`: **Abstract method** to be implemented by subclasses for specific agent logic.
    -   `store_result(key, value)`: Stores data to memory via MemoryManager.
    -   `log_event(event_type, details)`: Interface for structured logging (to be integrated with the central JSON logger).
    -   `run(input_data)`: Entry point to execute the agent's `process` method with logging and error handling.
-   **Integration:** Uses `AgentConfig`, `MemoryManager`, `LLMManager`, `Utils`. Connects to central logging/metrics.

### 3.2. AgentConfig (`agents/agent_config.py`, `config/`)

-   **Purpose:** Standardizes agent configuration.
-   **Components:**
    -   `AgentConfigSchema` (dataclass): Defines the structure (agent ID, model params, memory settings, log level, tools, permissions, retry logic).
    -   `load_config` (function in `utils/utils.py`): Parses and validates configuration files (e.g., `config/agent_config.yaml`).
-   **Format:** YAML is the default format for configuration files.

### 3.3. MemoryManager (`memory_manager/memory_manager.py`)

-   **Purpose:** Abstracts agent memory operations.
-   **Interface (`MemoryManager` ABC):**
    -   `read_memory(agent_id, key)`
    -   `write_memory(agent_id, key, value)`
    -   `load_agent_memory(agent_id)`
    -   `save_agent_memory(agent_id, state)`
-   **Implementations:**
    -   `InMemoryMemoryManager`: Stores data in a dictionary (ephemeral).
    -   `FileBasedMemoryManager`: Stores data in JSON files (persistent).
-   **Extensibility:** Designed to allow adding database backends (e.g., PostgreSQL) later by implementing the `MemoryManager` interface.
-   **Integration:** Used by `BaseAgent`. Requires configuration from `AgentConfigSchema`.

### 3.4. LLMManager (`llm_manager/llm_manager.py`)

-   **Purpose:** Centralizes LLM API interactions and error handling.
-   **Interface (`LLMManager` ABC):**
    -   `call_llm(prompt, ...)`: Main method to call the LLM, includes retry logic based on config.
    -   `_call_api(prompt, ...)`: **Abstract method** for specific API call implementation by subclasses.
-   **Implementations:**
    -   `StubLLMManager`: Placeholder for testing, does not call real APIs.
    -   *(Future: `OpenAILLMManager`, `AnthropicLLMManager`, etc.)*
-   **Error Handling:** Implements retry logic (exponential backoff) for `TransientError` based on `AgentConfigSchema` settings. Distinguishes between `TransientError` and `PermanentError`.
-   **Integration:** Used by `BaseAgent`. Requires configuration from `AgentConfigSchema`.

### 3.5. Utils (`utils/utils.py`)

-   **Purpose:** Provides shared code and definitions.
-   **Components:**
    -   Custom Error Classes (`BaseAgentError`, `TransientError`, `PermanentError`, `ConfigurationError`, `MemoryError`, `LLMError`).
    -   Helper functions (e.g., `load_config`).
    -   *(Future: Common data validation, formatting routines, etc.)*

## 4. Integration with Central Infrastructure

### 4.1. Logging

-   **Mechanism:** Agents use the `log_event` method within `BaseAgent`.
-   **Implementation:** This method MUST be adapted to use the central JSON logging utility provided in the **Agent Onboarding Guide**. It should format logs according to the established schema.
-   **Events:** Key lifecycle events (init, start/end process, fetch/store context, LLM calls, errors) are logged with relevant details.

### 4.2. Metrics

-   **Mechanism:** Agents should expose Prometheus metrics as defined in the **Agent Onboarding Guide**.
-   **Implementation:** Requires setting up a Prometheus client/server (potentially shared or per-agent) and incrementing counters/gauges for events like:
    -   Tasks dispatched/completed/failed.
    -   LLM calls made/succeeded/failed.
    -   Memory operations.
    -   Errors encountered (categorized by type).
-   **Reference:** Follow the examples in the **Agent Onboarding Guide** for metric names and implementation details.

### 4.3. Error Handling

-   **Categorization:** Errors are categorized using the custom exceptions defined in `utils.utils` (`TransientError`, `PermanentError`).
-   **Retry Logic:** `LLMManager` implements retry logic with exponential backoff for `TransientError`s, configured via `AgentConfigSchema`.
-   **Reporting:** Errors are logged via `log_event`. Critical/permanent errors should potentially trigger alerts (integration with monitoring system TBD).

## 5. Future Considerations

-   **Database Memory:** Implement `DatabaseMemoryManager` using PostgreSQL (as per optional guide).
-   **Real LLM Managers:** Implement managers for specific LLM providers (OpenAI, Anthropic, etc.).
-   **Tool Integration:** Define a formal mechanism for agents to discover and use tools based on configuration/permissions.
-   **Asynchronous Operations:** Consider using `asyncio` for non-blocking I/O, especially for LLM calls and potentially memory operations.
-   **Agent Orchestration:** Develop a higher-level system for managing multiple agents, scheduling tasks, and inter-agent communication.

## 6. Developer Onboarding

Refer to the **Agent Onboarding Guide** for detailed instructions on:
-   Setting up the development environment.
-   Integrating with the central logging utility.
-   Implementing metrics collection.
-   Following error handling best practices.
-   (Optional) Integrating with the database.

To create a new specialized agent:
1.  Subclass `BaseAgent`.
2.  Implement the `process` method with the agent's core logic.
3.  Create a configuration file in `config/` for the new agent.
4.  Add unit tests for the new agent in the `tests/` directory.
5.  Update this design document if the new agent introduces significant architectural changes.
