"""Global state management for agentsec."""

import threading
from typing import Any, Dict, List, Optional


# Thread lock for state mutations
_state_lock = threading.Lock()

# Supported LLM providers
SUPPORTED_PROVIDERS = ["openai", "azure_openai", "vertexai", "bedrock", "google_genai"]

# Valid configuration values
VALID_API_MODES = {"off", "monitor", "enforce"}
VALID_INTEGRATION_MODES = {"api", "gateway"}
VALID_GATEWAY_MODES = {"off", "on"}

# Global state
_initialized: bool = False

# LLM rules for API mode inspection
_llm_rules: Optional[List[Any]] = None

# LLM entity types for filtering (new)
_llm_entity_types: Optional[List[str]] = None

# Metadata defaults (new)
_metadata_user: Optional[str] = None
_metadata_src_app: Optional[str] = None
_metadata_client_transaction_id: Optional[str] = None

# Retry configuration (new)
_retry_total: Optional[int] = None  # Default: 1 (no retry)
_retry_backoff: Optional[float] = None  # Default: 0 (no backoff)
_retry_status_codes: Optional[List[int]] = None  # Default: [500, 502, 503, 504]

# Connection pool configuration (new)
_pool_max_connections: Optional[int] = None  # Default: 100
_pool_max_keepalive: Optional[int] = None  # Default: 20

# Timeout configuration (new) - in seconds
_timeout: Optional[int] = None  # Default: 1 second

# Logger configuration (new)
_log_file: Optional[str] = None
_log_format: Optional[str] = None
_custom_logger: Optional[Any] = None  # Custom logger instance

# Integration modes (api vs gateway) - common for both
_llm_integration_mode: str = "api"
_mcp_integration_mode: str = "api"

# API mode configuration
_api_mode_llm: Optional[str] = None  # off/monitor/enforce
_api_mode_mcp: Optional[str] = None  # off/monitor/enforce
_api_mode_llm_endpoint: Optional[str] = None
_api_mode_llm_api_key: Optional[str] = None
_api_mode_mcp_endpoint: Optional[str] = None
_api_mode_mcp_api_key: Optional[str] = None
_api_mode_fail_open_llm: bool = True
_api_mode_fail_open_mcp: bool = True

# Gateway mode configuration (general)
_gateway_mode_llm: str = "on"  # off/on
_gateway_mode_mcp: str = "on"  # off/on
_gateway_mode_mcp_url: Optional[str] = None
_gateway_mode_mcp_api_key: Optional[str] = None
_gateway_mode_fail_open_llm: bool = True
_gateway_mode_fail_open_mcp: bool = True

# Provider-specific gateway configuration
# Each provider has its own gateway URL and API key
_provider_gateway_config: Dict[str, Dict[str, Optional[str]]] = {
    "openai": {"url": None, "api_key": None},
    "azure_openai": {"url": None, "api_key": None},
    "vertexai": {"url": None, "api_key": None},
    "bedrock": {"url": None, "api_key": None},
    "google_genai": {"url": None, "api_key": None},
}

# Provider-specific API configuration (for direct calls in API mode)
_provider_api_config: Dict[str, Dict[str, Optional[str]]] = {
    "openai": {"url": None, "api_key": None},
    "azure_openai": {"url": None, "api_key": None},
    "vertexai": {"url": None, "api_key": None},
    "bedrock": {"url": None, "api_key": None},
    "google_genai": {"url": None, "api_key": None},
}


def is_initialized() -> bool:
    """Check if agentsec has been initialized."""
    return _initialized


def get_llm_rules() -> Optional[List[Any]]:
    """Get the current LLM rules for API mode inspection."""
    return _llm_rules


def get_llm_entity_types() -> Optional[List[str]]:
    """Get the LLM entity types for filtering."""
    return _llm_entity_types


# Metadata getters (new)
def get_metadata_user() -> Optional[str]:
    """Get the default metadata user."""
    return _metadata_user


def get_metadata_src_app() -> Optional[str]:
    """Get the default metadata source application."""
    return _metadata_src_app


def get_metadata_client_transaction_id() -> Optional[str]:
    """Get the default metadata client transaction ID."""
    return _metadata_client_transaction_id


# Retry configuration getters (new)
def get_retry_total() -> Optional[int]:
    """Get the retry total count. None means use default (1)."""
    return _retry_total


def get_retry_backoff() -> Optional[float]:
    """Get the retry backoff factor. None means use default (0)."""
    return _retry_backoff


def get_retry_status_codes() -> Optional[List[int]]:
    """Get the retry status codes. None means use default ([500, 502, 503, 504])."""
    return _retry_status_codes


# Connection pool configuration getters (new)
def get_pool_max_connections() -> Optional[int]:
    """Get the pool max connections. None means use default (100)."""
    return _pool_max_connections


def get_pool_max_keepalive() -> Optional[int]:
    """Get the pool max keepalive connections. None means use default (20)."""
    return _pool_max_keepalive


# Timeout configuration getter (new)
def get_timeout() -> Optional[int]:
    """Get the timeout in seconds. None means use default (1)."""
    return _timeout


# Logger configuration getters (new)
def get_log_file() -> Optional[str]:
    """Get the log file path."""
    return _log_file


def get_log_format() -> Optional[str]:
    """Get the log format ('text' or 'json')."""
    return _log_format


def get_custom_logger() -> Optional[Any]:
    """Get the custom logger instance."""
    return _custom_logger


# Integration mode getters (common)
def get_llm_integration_mode() -> str:
    """Get the current LLM integration mode ('api' or 'gateway')."""
    return _llm_integration_mode


def get_mcp_integration_mode() -> str:
    """Get the current MCP integration mode ('api' or 'gateway')."""
    return _mcp_integration_mode


# API mode getters
def get_api_mode_llm() -> Optional[str]:
    """Get the current LLM API mode (off/monitor/enforce)."""
    return _api_mode_llm


def get_api_mode_mcp() -> Optional[str]:
    """Get the current MCP API mode (off/monitor/enforce)."""
    return _api_mode_mcp


def get_api_mode_llm_endpoint() -> Optional[str]:
    """Get the LLM API endpoint."""
    return _api_mode_llm_endpoint


def get_api_mode_llm_api_key() -> Optional[str]:
    """Get the LLM API key."""
    return _api_mode_llm_api_key


def get_api_mode_mcp_endpoint() -> Optional[str]:
    """Get the MCP API endpoint (falls back to LLM endpoint if not set)."""
    return _api_mode_mcp_endpoint or _api_mode_llm_endpoint


def get_api_mode_mcp_api_key() -> Optional[str]:
    """Get the MCP API key (falls back to LLM key if not set)."""
    return _api_mode_mcp_api_key or _api_mode_llm_api_key


def get_api_mode_fail_open_llm() -> bool:
    """Get the LLM API mode fail_open setting."""
    return _api_mode_fail_open_llm


def get_api_mode_fail_open_mcp() -> bool:
    """Get the MCP API mode fail_open setting."""
    return _api_mode_fail_open_mcp


# Gateway mode getters
def get_gateway_mode_llm() -> str:
    """Get the current LLM gateway mode ('off' or 'on')."""
    return _gateway_mode_llm


def get_gateway_mode_mcp() -> str:
    """Get the current MCP gateway mode ('off' or 'on')."""
    return _gateway_mode_mcp


def get_gateway_mode_mcp_url() -> Optional[str]:
    """Get the MCP gateway URL."""
    return _gateway_mode_mcp_url


def get_gateway_mode_mcp_api_key() -> Optional[str]:
    """Get the MCP gateway API key."""
    return _gateway_mode_mcp_api_key


def get_gateway_mode_fail_open_llm() -> bool:
    """Get the LLM gateway fail_open setting."""
    return _gateway_mode_fail_open_llm


def get_gateway_mode_fail_open_mcp() -> bool:
    """Get the MCP gateway fail_open setting."""
    return _gateway_mode_fail_open_mcp


# Provider-specific gateway config getters
def get_provider_gateway_url(provider: str) -> Optional[str]:
    """
    Get gateway URL for a specific provider.
    
    Args:
        provider: Provider name (openai, azure_openai, vertexai, bedrock)
        
    Returns:
        Gateway URL or None if not configured
    """
    return _provider_gateway_config.get(provider, {}).get("url")


def get_provider_gateway_api_key(provider: str) -> Optional[str]:
    """
    Get gateway API key for a specific provider.
    
    Args:
        provider: Provider name (openai, azure_openai, vertexai, bedrock)
        
    Returns:
        Gateway API key or None if not configured
    """
    return _provider_gateway_config.get(provider, {}).get("api_key")


def get_provider_api_url(provider: str) -> Optional[str]:
    """
    Get API URL for a specific provider (for direct calls in API mode).
    
    Args:
        provider: Provider name (openai, azure_openai, vertexai, bedrock)
        
    Returns:
        API URL or None if not configured
    """
    return _provider_api_config.get(provider, {}).get("url")


def get_provider_api_key(provider: str) -> Optional[str]:
    """
    Get API key for a specific provider (for direct calls in API mode).
    
    Args:
        provider: Provider name (openai, azure_openai, vertexai, bedrock)
        
    Returns:
        API key or None if not configured
    """
    return _provider_api_config.get(provider, {}).get("api_key")


def set_provider_gateway_config(provider: str, url: Optional[str], api_key: Optional[str]) -> None:
    """
    Set gateway configuration for a specific provider.
    
    Args:
        provider: Provider name (openai, azure_openai, vertexai, bedrock)
        url: Gateway URL
        api_key: Gateway API key
    """
    if provider in _provider_gateway_config:
        _provider_gateway_config[provider] = {"url": url, "api_key": api_key}


def set_provider_api_config(provider: str, url: Optional[str], api_key: Optional[str]) -> None:
    """
    Set API configuration for a specific provider (for direct calls in API mode).
    
    Args:
        provider: Provider name (openai, azure_openai, vertexai, bedrock)
        url: API URL
        api_key: API key
    """
    if provider in _provider_api_config:
        _provider_api_config[provider] = {"url": url, "api_key": api_key}


# Legacy getters (aliases for backward compatibility)
def get_llm_mode() -> Optional[str]:
    """Get the current LLM inspection mode (alias for get_api_mode_llm)."""
    return _api_mode_llm


def get_mcp_mode() -> Optional[str]:
    """Get the current MCP inspection mode (alias for get_api_mode_mcp)."""
    return _api_mode_mcp


def get_llm_api_endpoint() -> Optional[str]:
    """Get the LLM API endpoint (alias for get_api_mode_llm_endpoint)."""
    return _api_mode_llm_endpoint


def get_llm_api_key() -> Optional[str]:
    """Get the LLM API key (alias for get_api_mode_llm_api_key)."""
    return _api_mode_llm_api_key


def get_mcp_api_endpoint() -> Optional[str]:
    """Get the MCP API endpoint (alias for get_api_mode_mcp_endpoint)."""
    return _api_mode_mcp_endpoint or _api_mode_llm_endpoint


def get_mcp_api_key() -> Optional[str]:
    """Get the MCP API key (alias for get_api_mode_mcp_api_key)."""
    return _api_mode_mcp_api_key or _api_mode_llm_api_key


def get_mcp_gateway_url() -> Optional[str]:
    """Get the MCP gateway URL (alias for get_gateway_mode_mcp_url)."""
    return _gateway_mode_mcp_url


def get_mcp_gateway_api_key() -> Optional[str]:
    """Get the MCP gateway API key (alias for get_gateway_mode_mcp_api_key)."""
    return _gateway_mode_mcp_api_key


def get_llm_gateway_mode() -> str:
    """Get the LLM gateway mode (alias for get_gateway_mode_llm)."""
    return _gateway_mode_llm


def get_mcp_gateway_mode() -> str:
    """Get the MCP gateway mode (alias for get_gateway_mode_mcp)."""
    return _gateway_mode_mcp


def get_llm_gateway_fail_open() -> bool:
    """Get the LLM gateway fail_open (alias for get_gateway_mode_fail_open_llm)."""
    return _gateway_mode_fail_open_llm


def get_mcp_gateway_fail_open() -> bool:
    """Get the MCP gateway fail_open (alias for get_gateway_mode_fail_open_mcp)."""
    return _gateway_mode_fail_open_mcp


def set_state(
    initialized: bool,
    # LLM rules for API mode
    llm_rules: Optional[List[Any]] = None,
    # LLM entity types (new)
    llm_entity_types: Optional[List[str]] = None,
    # Integration modes
    llm_integration_mode: str = "api",
    mcp_integration_mode: str = "api",
    # API mode configuration
    api_mode_llm: Optional[str] = None,
    api_mode_mcp: Optional[str] = None,
    api_mode_llm_endpoint: Optional[str] = None,
    api_mode_llm_api_key: Optional[str] = None,
    api_mode_mcp_endpoint: Optional[str] = None,
    api_mode_mcp_api_key: Optional[str] = None,
    api_mode_fail_open_llm: bool = True,
    api_mode_fail_open_mcp: bool = True,
    # Gateway mode configuration
    gateway_mode_llm: str = "on",
    gateway_mode_mcp: str = "on",
    gateway_mode_mcp_url: Optional[str] = None,
    gateway_mode_mcp_api_key: Optional[str] = None,
    gateway_mode_fail_open_llm: bool = True,
    gateway_mode_fail_open_mcp: bool = True,
    # Provider-specific gateway configuration
    provider_gateway_config: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
    # Provider-specific API configuration
    provider_api_config: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
    # Metadata configuration (new)
    metadata_user: Optional[str] = None,
    metadata_src_app: Optional[str] = None,
    metadata_client_transaction_id: Optional[str] = None,
    # Retry configuration (new)
    retry_total: Optional[int] = None,
    retry_backoff: Optional[float] = None,
    retry_status_codes: Optional[List[int]] = None,
    # Connection pool configuration (new)
    pool_max_connections: Optional[int] = None,
    pool_max_keepalive: Optional[int] = None,
    # Timeout configuration (new)
    timeout: Optional[int] = None,
    # Logger configuration (new)
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    custom_logger: Optional[Any] = None,
) -> None:
    """
    Set the global state.
    
    Args:
        initialized: Whether agentsec has been initialized
        llm_rules: LLM rules for API mode inspection
        llm_entity_types: LLM entity types for filtering
        llm_integration_mode: Integration mode for LLM (api/gateway)
        mcp_integration_mode: Integration mode for MCP (api/gateway)
        api_mode_llm: Mode for LLM in API mode (off/monitor/enforce)
        api_mode_mcp: Mode for MCP in API mode (off/monitor/enforce)
        api_mode_llm_endpoint: API endpoint for LLM inspection
        api_mode_llm_api_key: API key for LLM inspection
        api_mode_mcp_endpoint: API endpoint for MCP inspection
        api_mode_mcp_api_key: API key for MCP inspection
        api_mode_fail_open_llm: Allow LLM requests on API errors
        api_mode_fail_open_mcp: Allow MCP calls on API errors
        gateway_mode_llm: Mode for LLM in gateway mode (off/on)
        gateway_mode_mcp: Mode for MCP in gateway mode (off/on)
        gateway_mode_mcp_url: Gateway URL for MCP calls
        gateway_mode_mcp_api_key: Gateway API key for MCP calls
        gateway_mode_fail_open_llm: Allow LLM requests on gateway errors
        gateway_mode_fail_open_mcp: Allow MCP calls on gateway errors
        provider_gateway_config: Per-provider gateway config {provider: {url, api_key}}
        provider_api_config: Per-provider API config {provider: {url, api_key}}
        metadata_user: Default metadata user
        metadata_src_app: Default metadata source application
        metadata_client_transaction_id: Default metadata client transaction ID
        retry_total: Retry total count
        retry_backoff: Retry backoff factor
        retry_status_codes: Status codes to retry on
        pool_max_connections: Pool max connections
        pool_max_keepalive: Pool max keepalive connections
        timeout: Timeout in seconds
        log_file: Log file path
        log_format: Log format ('text' or 'json')
        custom_logger: Custom logger instance
        
    Raises:
        ConfigurationError: If any configuration value is invalid
    """
    from .exceptions import ConfigurationError
    
    # Input validation
    if api_mode_llm is not None and api_mode_llm not in VALID_API_MODES:
        raise ConfigurationError(f"Invalid api_mode_llm: '{api_mode_llm}'. Must be one of: {', '.join(VALID_API_MODES)}")
    
    if api_mode_mcp is not None and api_mode_mcp not in VALID_API_MODES:
        raise ConfigurationError(f"Invalid api_mode_mcp: '{api_mode_mcp}'. Must be one of: {', '.join(VALID_API_MODES)}")
    
    if llm_integration_mode not in VALID_INTEGRATION_MODES:
        raise ConfigurationError(f"Invalid llm_integration_mode: '{llm_integration_mode}'. Must be one of: {', '.join(VALID_INTEGRATION_MODES)}")
    
    if mcp_integration_mode not in VALID_INTEGRATION_MODES:
        raise ConfigurationError(f"Invalid mcp_integration_mode: '{mcp_integration_mode}'. Must be one of: {', '.join(VALID_INTEGRATION_MODES)}")
    
    if gateway_mode_llm not in VALID_GATEWAY_MODES:
        raise ConfigurationError(f"Invalid gateway_mode_llm: '{gateway_mode_llm}'. Must be one of: {', '.join(VALID_GATEWAY_MODES)}")
    
    if gateway_mode_mcp not in VALID_GATEWAY_MODES:
        raise ConfigurationError(f"Invalid gateway_mode_mcp: '{gateway_mode_mcp}'. Must be one of: {', '.join(VALID_GATEWAY_MODES)}")
    
    # Validate provider configuration keys
    if provider_gateway_config:
        invalid_providers = set(provider_gateway_config.keys()) - set(SUPPORTED_PROVIDERS)
        if invalid_providers:
            raise ConfigurationError(f"Invalid provider names in provider_gateway_config: {invalid_providers}. Supported providers: {SUPPORTED_PROVIDERS}")
    
    if provider_api_config:
        invalid_providers = set(provider_api_config.keys()) - set(SUPPORTED_PROVIDERS)
        if invalid_providers:
            raise ConfigurationError(f"Invalid provider names in provider_api_config: {invalid_providers}. Supported providers: {SUPPORTED_PROVIDERS}")
    
    # Validate numeric parameters
    if retry_total is not None and retry_total < 1:
        raise ConfigurationError(f"Invalid retry_total: {retry_total}. Must be >= 1")
    
    if retry_backoff is not None and retry_backoff < 0:
        raise ConfigurationError(f"Invalid retry_backoff: {retry_backoff}. Must be >= 0")
    
    if timeout is not None and timeout <= 0:
        raise ConfigurationError(f"Invalid timeout: {timeout}. Must be > 0")
    
    if pool_max_connections is not None and pool_max_connections < 1:
        raise ConfigurationError(f"Invalid pool_max_connections: {pool_max_connections}. Must be >= 1")
    
    if pool_max_keepalive is not None and pool_max_keepalive < 0:
        raise ConfigurationError(f"Invalid pool_max_keepalive: {pool_max_keepalive}. Must be >= 0")
    
    global _initialized, _llm_rules, _llm_entity_types
    global _llm_integration_mode, _mcp_integration_mode
    global _api_mode_llm, _api_mode_mcp
    global _api_mode_llm_endpoint, _api_mode_llm_api_key
    global _api_mode_mcp_endpoint, _api_mode_mcp_api_key
    global _api_mode_fail_open_llm, _api_mode_fail_open_mcp
    global _gateway_mode_llm, _gateway_mode_mcp
    global _gateway_mode_mcp_url, _gateway_mode_mcp_api_key
    global _gateway_mode_fail_open_llm, _gateway_mode_fail_open_mcp
    global _provider_gateway_config, _provider_api_config
    global _metadata_user, _metadata_src_app, _metadata_client_transaction_id
    global _retry_total, _retry_backoff, _retry_status_codes
    global _pool_max_connections, _pool_max_keepalive
    global _timeout, _log_file, _log_format, _custom_logger
    
    # Thread-safe state mutation
    with _state_lock:
        _initialized = initialized
        _llm_rules = llm_rules
        _llm_entity_types = llm_entity_types
        _llm_integration_mode = llm_integration_mode
        _mcp_integration_mode = mcp_integration_mode
        _api_mode_llm = api_mode_llm
        _api_mode_mcp = api_mode_mcp
        _api_mode_llm_endpoint = api_mode_llm_endpoint
        _api_mode_llm_api_key = api_mode_llm_api_key
        _api_mode_mcp_endpoint = api_mode_mcp_endpoint
        _api_mode_mcp_api_key = api_mode_mcp_api_key
        _api_mode_fail_open_llm = api_mode_fail_open_llm
        _api_mode_fail_open_mcp = api_mode_fail_open_mcp
        _gateway_mode_llm = gateway_mode_llm
        _gateway_mode_mcp = gateway_mode_mcp
        _gateway_mode_mcp_url = gateway_mode_mcp_url
        _gateway_mode_mcp_api_key = gateway_mode_mcp_api_key
        _gateway_mode_fail_open_llm = gateway_mode_fail_open_llm
        _gateway_mode_fail_open_mcp = gateway_mode_fail_open_mcp
        
        # New state variables
        _metadata_user = metadata_user
        _metadata_src_app = metadata_src_app
        _metadata_client_transaction_id = metadata_client_transaction_id
        _retry_total = retry_total
        _retry_backoff = retry_backoff
        _retry_status_codes = retry_status_codes
        _pool_max_connections = pool_max_connections
        _pool_max_keepalive = pool_max_keepalive
        _timeout = timeout
        _log_file = log_file
        _log_format = log_format
        _custom_logger = custom_logger
        
        # Update provider-specific configs
        if provider_gateway_config:
            for provider, config in provider_gateway_config.items():
                if provider in _provider_gateway_config:
                    _provider_gateway_config[provider] = config
        
        if provider_api_config:
            for provider, config in provider_api_config.items():
                if provider in _provider_api_config:
                    _provider_api_config[provider] = config


def reset() -> None:
    """Reset global state. Useful for testing."""
    global _initialized, _llm_rules, _llm_entity_types
    global _llm_integration_mode, _mcp_integration_mode
    global _api_mode_llm, _api_mode_mcp
    global _api_mode_llm_endpoint, _api_mode_llm_api_key
    global _api_mode_mcp_endpoint, _api_mode_mcp_api_key
    global _api_mode_fail_open_llm, _api_mode_fail_open_mcp
    global _gateway_mode_llm, _gateway_mode_mcp
    global _gateway_mode_mcp_url, _gateway_mode_mcp_api_key
    global _gateway_mode_fail_open_llm, _gateway_mode_fail_open_mcp
    global _provider_gateway_config, _provider_api_config
    global _metadata_user, _metadata_src_app, _metadata_client_transaction_id
    global _retry_total, _retry_backoff, _retry_status_codes
    global _pool_max_connections, _pool_max_keepalive
    global _timeout, _log_file, _log_format, _custom_logger
    
    # Thread-safe state reset
    with _state_lock:
        _initialized = False
        _llm_rules = None
        _llm_entity_types = None
        _llm_integration_mode = "api"
        _mcp_integration_mode = "api"
        _api_mode_llm = None
        _api_mode_mcp = None
        _api_mode_llm_endpoint = None
        _api_mode_llm_api_key = None
        _api_mode_mcp_endpoint = None
        _api_mode_mcp_api_key = None
        _api_mode_fail_open_llm = True
        _api_mode_fail_open_mcp = True
        _gateway_mode_llm = "on"
        _gateway_mode_mcp = "on"
        _gateway_mode_mcp_url = None
        _gateway_mode_mcp_api_key = None
        _gateway_mode_fail_open_llm = True
        _gateway_mode_fail_open_mcp = True
        
        # Reset new state variables
        _metadata_user = None
        _metadata_src_app = None
        _metadata_client_transaction_id = None
        _retry_total = None
        _retry_backoff = None
        _retry_status_codes = None
        _pool_max_connections = None
        _pool_max_keepalive = None
        _timeout = None
        _log_file = None
        _log_format = None
        _custom_logger = None
        
        # Reset provider-specific configs
        _provider_gateway_config = {
            "openai": {"url": None, "api_key": None},
            "azure_openai": {"url": None, "api_key": None},
            "vertexai": {"url": None, "api_key": None},
            "bedrock": {"url": None, "api_key": None},
            "google_genai": {"url": None, "api_key": None},
        }
        _provider_api_config = {
            "openai": {"url": None, "api_key": None},
            "azure_openai": {"url": None, "api_key": None},
            "vertexai": {"url": None, "api_key": None},
            "bedrock": {"url": None, "api_key": None},
            "google_genai": {"url": None, "api_key": None},
        }
