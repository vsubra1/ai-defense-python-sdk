"""
agentsec - Agent Runtime Security Sensor SDK

Provides runtime security enforcement and monitoring for LLM and MCP
interactions with minimal integration effort.

Usage (Simple - with YAML config):
    import agentsec
    agentsec.protect(config="agentsec.yaml")

    # Now import your LLM client
    from openai import OpenAI

Usage (Programmatic):
    import agentsec
    agentsec.protect(
        llm_integration_mode="gateway",
        gateway_mode={
            "llm_gateways": {
                "openai-1": {
                    "gateway_url": "https://gw.aidefense.cisco.com/t1/conn/openai",
                    "gateway_api_key": "your-key",
                    "auth_mode": "api_key",
                    "provider": "openai",
                    "default": True,
                },
            },
        },
    )

Usage (Named gateways):
    import agentsec
    agentsec.protect(config="agentsec.yaml")

    with agentsec.gateway("math-gateway"):
        response = client.chat.completions.create(...)

For more information, see:
https://developer.cisco.com/docs/ai-defense/overview/
"""

import copy
import logging
import threading
from typing import Any, Dict, List, Optional

from . import _state
from .decision import Decision
from .exceptions import (
    AgentsecError,
    ConfigurationError,
    ValidationError,
    InspectionTimeoutError,
    InspectionNetworkError,
    SecurityPolicyError,
)
from ._logging import setup_logging
from ._context import (
    skip_inspection,
    no_inspection,
    set_metadata,
    gateway,
    use_gateway,
)

# Lock for thread-safe initialization of protect()
_protect_lock = threading.Lock()

__all__ = [
    "protect",
    "get_patched_clients",
    "set_metadata",
    "Decision",
    # Exceptions
    "AgentsecError",
    "ConfigurationError",
    "ValidationError",
    "InspectionTimeoutError",
    "InspectionNetworkError",
    "SecurityPolicyError",
    # Context managers / decorators
    "skip_inspection",
    "no_inspection",
    "gateway",
    "use_gateway",
]

__version__ = "0.1.0"

# Logger - use the centralized logging module
logger = logging.getLogger("aidefense.runtime.agentsec")


def _auto_load_dotenv() -> bool:
    """Automatically load .env file if python-dotenv is installed.

    Searches for .env starting from the current working directory (usecwd=True).

    Returns:
        True if .env was loaded, False otherwise.
    """
    try:
        from dotenv import load_dotenv, find_dotenv

        dotenv_path = find_dotenv(usecwd=True)
        if dotenv_path:
            load_dotenv(dotenv_path)
            logger.debug(f"Auto-loaded .env file: {dotenv_path}")
        else:
            load_dotenv()
            logger.debug("No .env file found in current directory")
        return True
    except ImportError:
        logger.debug("python-dotenv not installed, skipping .env auto-load")
        return False


def _apply_patches(api_mode_llm: Optional[str], api_mode_mcp: Optional[str]) -> None:
    """Apply client patches based on the effective mode for each integration path.

    For each category (LLM / MCP), patching is applied when:
    - Gateway integration is active AND gateway mode is "on", OR
    - API integration is active AND api_mode is not "off".
    """
    from .patchers import (
        patch_openai,
        patch_bedrock,
        patch_mcp,
        patch_vertexai,
        patch_google_genai,
        patch_cohere,
        patch_mistral,
        patch_litellm,
    )

    llm_integration = _state.get_llm_integration_mode()
    mcp_integration = _state.get_mcp_integration_mode()

    # Determine if LLM patching is needed
    llm_active = (
        (llm_integration == "gateway" and _state.get_gw_llm_mode() != "off")
        or (llm_integration == "api" and api_mode_llm != "off")
    )
    if llm_active:
        patch_openai()
        patch_bedrock()
        patch_vertexai()
        patch_google_genai()
        patch_cohere()
        patch_mistral()
        patch_litellm()

    # Determine if MCP patching is needed
    mcp_active = (
        (mcp_integration == "gateway" and _state.get_gw_mcp_mode() != "off")
        or (mcp_integration == "api" and api_mode_mcp != "off")
    )
    if mcp_active:
        patch_mcp()


def get_patched_clients() -> List[str]:
    """Get list of successfully patched clients.

    Returns:
        List of client names that have been patched.
    """
    from .patchers import get_patched_clients as _get_patched

    return _get_patched()


# =========================================================================
# Deep merge utility
# =========================================================================

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively deep-merge *override* into *base*.

    - Leaf values from *override* replace *base* values.
    - ``None`` values in *override* are **skipped** (base value preserved).
    - Nested dicts are merged recursively.

    Returns a **new** dict (neither *base* nor *override* is mutated).
    """
    result = copy.deepcopy(base)
    for key, val in override.items():
        if val is None:
            continue
        if isinstance(val, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


# =========================================================================
# protect() — public API
# =========================================================================

def protect(
    patch_clients: bool = True,
    *,
    auto_dotenv: bool = True,
    config: Optional[str] = None,
    llm_integration_mode: Optional[str] = None,
    mcp_integration_mode: Optional[str] = None,
    gateway_mode: Optional[dict] = None,
    api_mode: Optional[dict] = None,
    pool_max_connections: Optional[int] = None,
    pool_max_keepalive: Optional[int] = None,
    custom_logger: Optional[logging.Logger] = None,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
) -> None:
    """Enable agentsec protection for LLM and MCP interactions.

    This is the main entry point for agentsec. Call this once at the top
    of your application, BEFORE importing any LLM clients.

    Minimal usage (no config)::

        import agentsec
        agentsec.protect()

    YAML config::

        import agentsec
        agentsec.protect(config="agentsec.yaml")

    Programmatic::

        import agentsec
        agentsec.protect(
            llm_integration_mode="gateway",
            gateway_mode={
                "llm_gateways": {
                    "openai-1": {
                        "gateway_url": "https://...",
                        "gateway_api_key": "key",
                        "auth_mode": "api_key",
                        "provider": "openai",
                        "default": True,
                    },
                },
            },
        )

    This function is idempotent — calling it multiple times has no
    effect after the first successful call.

    Args:
        patch_clients: Whether to auto-patch LLM clients.
        auto_dotenv: Load .env before YAML parsing so ``${VAR}``
            references can resolve.
        config: Path to an ``agentsec.yaml`` configuration file.
        llm_integration_mode: ``"api"`` or ``"gateway"``.
        mcp_integration_mode: ``"api"`` or ``"gateway"``.
        gateway_mode: Dict matching the ``gateway_mode`` section in YAML
            (llm_defaults, mcp_defaults, llm_gateways, mcp_gateways).
        api_mode: Dict matching the ``api_mode`` section in YAML
            (llm_defaults, mcp_defaults, llm, mcp).
        pool_max_connections: Max HTTP connections (global).
        pool_max_keepalive: Max keepalive connections (global).
        custom_logger: Custom ``logging.Logger`` instance.
        log_file: Log file path.
        log_format: ``"text"`` or ``"json"``.

    Raises:
        ConfigurationError: If config file cannot be loaded or
            contains invalid values.
    """
    # Idempotency check
    if _state.is_initialized():
        logger.debug("agentsec already initialized, skipping")
        return

    with _protect_lock:
        # Double-check after acquiring lock
        if _state.is_initialized():
            logger.debug("agentsec already initialized (after lock), skipping")
            return

        _protect_impl(
            patch_clients=patch_clients,
            auto_dotenv=auto_dotenv,
            config=config,
            llm_integration_mode=llm_integration_mode,
            mcp_integration_mode=mcp_integration_mode,
            gateway_mode=gateway_mode,
            api_mode=api_mode,
            pool_max_connections=pool_max_connections,
            pool_max_keepalive=pool_max_keepalive,
            custom_logger=custom_logger,
            log_file=log_file,
            log_format=log_format,
        )


def _protect_impl(
    patch_clients: bool,
    auto_dotenv: bool,
    config: Optional[str],
    llm_integration_mode: Optional[str],
    mcp_integration_mode: Optional[str],
    gateway_mode: Optional[dict],
    api_mode: Optional[dict],
    pool_max_connections: Optional[int],
    pool_max_keepalive: Optional[int],
    custom_logger: Optional[logging.Logger],
    log_file: Optional[str],
    log_format: Optional[str],
) -> None:
    """Internal implementation of protect(), called under lock."""

    # Step 1: Load .env so ${VAR} substitution can resolve in YAML
    if auto_dotenv:
        _auto_load_dotenv()

    # Step 2: Build merged config: hardcoded defaults -> YAML -> kwargs
    merged: Dict[str, Any] = {}

    if config is not None:
        from .config_file import load_config_file

        merged = load_config_file(config)

    # Overlay protect() kwargs (non-None only) via deep merge
    kwargs_overlay: Dict[str, Any] = {}
    if llm_integration_mode is not None:
        kwargs_overlay["llm_integration_mode"] = llm_integration_mode
    if mcp_integration_mode is not None:
        kwargs_overlay["mcp_integration_mode"] = mcp_integration_mode
    if gateway_mode is not None:
        kwargs_overlay["gateway_mode"] = gateway_mode
    if api_mode is not None:
        kwargs_overlay["api_mode"] = api_mode

    if kwargs_overlay:
        merged = _deep_merge(merged, kwargs_overlay)

    # Step 3: Extract final values
    final_llm_integration = merged.get("llm_integration_mode", "api")
    final_mcp_integration = merged.get("mcp_integration_mode", "api")
    final_gateway_mode = merged.get("gateway_mode") or {}
    final_api_mode = merged.get("api_mode") or {}

    # Extract API mode strings for patching decisions
    api_llm_cfg = final_api_mode.get("llm") or {}
    api_mcp_cfg = final_api_mode.get("mcp") or {}
    api_mode_llm_str = api_llm_cfg.get("mode")
    api_mode_mcp_str = api_mcp_cfg.get("mode")

    # Extract logging config from YAML
    logging_cfg = merged.get("logging") or {}
    final_log_file = log_file or logging_cfg.get("file")
    final_log_format = log_format or logging_cfg.get("format")
    log_level = logging_cfg.get("level")

    # Step 4: Setup logging
    setup_logging(
        level=log_level,
        format_type=final_log_format,
        log_file=final_log_file,
        custom_logger=custom_logger,
    )

    # Step 5: Store state BEFORE patching
    _state.set_state(
        initialized=True,
        llm_rules=api_llm_cfg.get("rules"),
        llm_entity_types=api_llm_cfg.get("entity_types"),
        llm_integration_mode=final_llm_integration,
        mcp_integration_mode=final_mcp_integration,
        gateway_mode=final_gateway_mode,
        api_mode=final_api_mode,
        pool_max_connections=pool_max_connections,
        pool_max_keepalive=pool_max_keepalive,
        log_file=final_log_file,
        log_format=final_log_format,
        custom_logger=custom_logger,
    )

    # Step 6: Apply client patches
    patched: List[str] = []
    if patch_clients:
        logger.debug("Applying client patches...")
        _apply_patches(api_mode_llm_str, api_mode_mcp_str)
        patched = get_patched_clients()

    # Step 7: Log initialization summary
    gw_llm_mode = final_gateway_mode.get("llm_mode", "on")
    gw_mcp_mode = final_gateway_mode.get("mcp_mode", "on")

    # Build display strings per integration path
    if final_llm_integration == "gateway":
        llm_display = f"gateway ({gw_llm_mode})"
    else:
        llm_display = api_mode_llm_str or "not configured"

    if final_mcp_integration == "gateway":
        mcp_display = f"gateway ({gw_mcp_mode})"
    else:
        mcp_display = api_mode_mcp_str or "not configured"

    print(
        f"[agentsec] LLM: {llm_display} | MCP: {mcp_display} "
        f"| Patched: {patched}"
    )
    logger.info(
        f"agentsec initialized: llm={llm_display}, mcp={mcp_display}, "
        f"llm_integration={final_llm_integration}, "
        f"mcp_integration={final_mcp_integration}"
    )
