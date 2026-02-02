"""MCP Inspector for tool, prompt, and resource inspection using Cisco AI Defense MCP Inspection API."""

import itertools
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

import httpx

from ..decision import Decision
from ..exceptions import (
    SecurityPolicyError,
    InspectionTimeoutError,
    InspectionNetworkError,
)

logger = logging.getLogger("aidefense.runtime.agentsec.inspectors.mcp")


class MCPInspector:
    """
    Inspector for MCP (Model Context Protocol) operations using Cisco AI Defense.
    
    This class integrates with the Cisco AI Defense MCP Inspection API to
    inspect MCP tool calls, prompt retrievals, and resource reads for security
    policy violations.
    
    Supported MCP methods:
    - tools/call: Tool execution inspection
    - prompts/get: Prompt retrieval inspection
    - resources/read: Resource access inspection
    
    The API expects raw MCP JSON-RPC 2.0 messages and returns inspection results
    with is_safe boolean and action (Allow/Block).
    
    Attributes:
        api_key: API key for Cisco AI Defense MCP inspection
        endpoint: Base URL for the AI Defense MCP API
        timeout_ms: Request timeout in milliseconds
        retry_total: Total number of retry attempts (default 1 = no retry)
        retry_backoff: Exponential backoff factor in seconds (default 0 = no backoff)
        retry_status_codes: HTTP status codes to retry on
        fail_open: Whether to allow operations when API is unreachable
    """
    
    # Maximum backoff delay to prevent runaway waits
    MAX_BACKOFF_DELAY = 30.0
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        retry_attempts: Optional[int] = None,  # Deprecated, use retry_total
        retry_total: Optional[int] = None,
        retry_backoff: Optional[float] = None,
        retry_status_codes: Optional[List[int]] = None,
        pool_max_connections: Optional[int] = None,
        pool_max_keepalive: Optional[int] = None,
        fail_open: bool = True,
    ) -> None:
        """
        Initialize the MCP Inspector.
        
        Args:
            api_key: API key for Cisco AI Defense MCP inspection.
                     Falls back to AI_DEFENSE_API_MODE_MCP_API_KEY, then AI_DEFENSE_API_MODE_LLM_API_KEY env vars.
            endpoint: Base URL for the AI Defense MCP API.
                      Falls back to AI_DEFENSE_API_MODE_MCP_ENDPOINT, then AI_DEFENSE_API_MODE_LLM_ENDPOINT env vars.
            timeout_ms: Request timeout in milliseconds (default 1000)
            retry_attempts: Deprecated, use retry_total instead
            retry_total: Total number of retry attempts (default 1, no retry)
            retry_backoff: Exponential backoff factor in seconds (default 0, no backoff)
            retry_status_codes: HTTP status codes to retry on (default [500, 502, 503, 504])
            pool_max_connections: Maximum connections in the pool (default 100)
            pool_max_keepalive: Maximum keepalive connections (default 20)
            fail_open: If True, allow tool calls on API errors (default True)
        """
        from .. import _state
        
        # API key: explicit > state > MCP-specific env > general env
        self.api_key = (
            api_key 
            or _state.get_api_mode_mcp_api_key() 
            or os.environ.get("AI_DEFENSE_API_MODE_MCP_API_KEY") 
            or os.environ.get("AI_DEFENSE_API_MODE_LLM_API_KEY")
        )
        
        # Endpoint: explicit > state > MCP-specific env > general env
        raw_endpoint = (
            endpoint 
            or _state.get_api_mode_mcp_endpoint() 
            or os.environ.get("AI_DEFENSE_API_MODE_MCP_ENDPOINT") 
            or os.environ.get("AI_DEFENSE_API_MODE_LLM_ENDPOINT")
        )
        
        # Store base endpoint (strip any trailing /api/v1/inspect/mcp path)
        if raw_endpoint:
            self.endpoint = raw_endpoint.rstrip("/").removesuffix("/api/v1/inspect/mcp").removesuffix("/api")
        else:
            self.endpoint = None
        
        self.fail_open = fail_open
        
        # Timeout: explicit param > state > default (1000ms)
        if timeout_ms is not None:
            self.timeout_ms = timeout_ms
        else:
            state_timeout = _state.get_timeout()
            self.timeout_ms = (state_timeout * 1000) if state_timeout is not None else 1000
        
        # Retry configuration: explicit param > state > default
        if retry_total is not None:
            self.retry_total = max(1, retry_total)
        elif retry_attempts is not None:
            logger.debug("retry_attempts is deprecated, use retry_total instead")
            self.retry_total = max(1, retry_attempts)
        else:
            state_retry = _state.get_retry_total()
            self.retry_total = max(1, state_retry) if state_retry is not None else 1
        
        if retry_backoff is not None:
            self.retry_backoff = max(0.0, retry_backoff)
        else:
            state_backoff = _state.get_retry_backoff()
            self.retry_backoff = max(0.0, state_backoff) if state_backoff is not None else 0.0
        
        if retry_status_codes is not None:
            self.retry_status_codes = retry_status_codes
        else:
            state_codes = _state.get_retry_status_codes()
            self.retry_status_codes = state_codes if state_codes is not None else [500, 502, 503, 504]
        
        # Keep retry_attempts as alias for backward compatibility
        self.retry_attempts = self.retry_total
        
        # Connection pool configuration: explicit param > state > default
        if pool_max_connections is not None:
            self.pool_max_connections = pool_max_connections
        else:
            state_pool = _state.get_pool_max_connections()
            self.pool_max_connections = state_pool if state_pool is not None else 100
        
        if pool_max_keepalive is not None:
            self.pool_max_keepalive = pool_max_keepalive
        else:
            state_keepalive = _state.get_pool_max_keepalive()
            self.pool_max_keepalive = state_keepalive if state_keepalive is not None else 20
        
        # Thread-safe counter for JSON-RPC message IDs using itertools.count()
        self._request_id_counter = itertools.count(1)
        
        # Create HTTP client with configured limits
        timeout = httpx.Timeout(self.timeout_ms / 1000.0)
        limits = httpx.Limits(
            max_connections=self.pool_max_connections,
            max_keepalive_connections=self.pool_max_keepalive,
        )
        self._sync_client = httpx.Client(timeout=timeout, limits=limits, http2=False)
        
        # Async client is lazily created and reused per event loop
        # to avoid "attached to different event loop" errors
        self._async_client: Optional[httpx.AsyncClient] = None
        self._async_loop_id: Optional[int] = None
        self._async_client_lock = threading.Lock()  # Thread-safe async client creation
    
    def _get_next_id(self) -> int:
        """Get the next request ID for JSON-RPC messages (thread-safe)."""
        return next(self._request_id_counter)
    
    def _get_async_client(self) -> httpx.AsyncClient:
        """
        Get or create async client for the current event loop.
        
        The client is reused within the same event loop but recreated
        if the event loop changes (e.g., different threads or frameworks).
        
        Thread-safe: Uses lock to prevent race conditions during client creation.
        """
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        loop_id = id(loop) if loop else None
        
        # Fast path: check without lock if client exists for current loop
        if self._async_client is not None and self._async_loop_id == loop_id:
            return self._async_client
        
        # Slow path: acquire lock for thread-safe client creation
        with self._async_client_lock:
            # Double-check after acquiring lock
            if self._async_client is not None and self._async_loop_id == loop_id:
                return self._async_client
            
            # Discard old client if it exists (different loop)
            if self._async_client is not None:
                logger.debug("Discarding async client from different event loop")
            
            timeout = httpx.Timeout(self.timeout_ms / 1000.0)
            limits = httpx.Limits(
                max_connections=self.pool_max_connections,
                max_keepalive_connections=self.pool_max_keepalive,
            )
            self._async_client = httpx.AsyncClient(timeout=timeout, limits=limits, http2=False)
            self._async_loop_id = loop_id
            logger.debug(f"Created new async HTTP client for loop {loop_id}")
        
        return self._async_client
    
    def _build_request_message(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        method: str = "tools/call",
    ) -> Dict[str, Any]:
        """
        Build a JSON-RPC 2.0 request message for MCP inspection.
        
        Args:
            tool_name: Name of the tool/prompt/resource being accessed
            arguments: Arguments passed to the operation
            method: MCP method (tools/call, prompts/get, resources/read)
            
        Returns:
            JSON-RPC 2.0 request message dict
        """
        if method == "prompts/get":
            return {
                "jsonrpc": "2.0",
                "method": method,
                "params": {
                    "name": tool_name,
                    "arguments": arguments,
                },
                "id": self._get_next_id(),
            }
        elif method == "resources/read":
            return {
                "jsonrpc": "2.0",
                "method": method,
                "params": {
                    "uri": tool_name,  # For resources, the "name" is the URI
                },
                "id": self._get_next_id(),
            }
        else:
            # Default: tools/call
            return {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments,
                },
                "id": self._get_next_id(),
            }
    
    def _build_response_message(
        self,
        result: Any,
    ) -> Dict[str, Any]:
        """
        Build a JSON-RPC 2.0 response message for MCP tool response inspection.
        
        Args:
            result: The result returned by the tool
            
        Returns:
            JSON-RPC 2.0 response message dict
        """
        # Convert result to text content format expected by MCP
        if isinstance(result, str):
            text_content = result
        elif isinstance(result, (dict, list)):
            text_content = json.dumps(result)
        else:
            text_content = str(result)
        
        return {
            "jsonrpc": "2.0",
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": text_content,
                    }
                ]
            },
            "id": self._get_next_id(),
        }
    
    def _parse_mcp_response(self, response_data: Dict[str, Any]) -> Decision:
        """
        Parse the MCP Inspection API response into a Decision.
        
        The API returns a JSON-RPC 2.0 response with result containing:
        - is_safe: boolean (primary decision flag)
        - action: "Allow" or "Block"
        - severity: NONE_SEVERITY, LOW, MEDIUM, HIGH, CRITICAL
        - rules: list of rules that triggered
        - classifications: list of violation types
        - explanation: human-readable explanation
        - event_id: unique inspection event ID
        
        Args:
            response_data: JSON response from the API
            
        Returns:
            Decision based on API response
        """
        # Extract the result object from JSON-RPC response
        result = response_data.get("result", response_data)
        
        # Primary decision: action field (Allow/Block)
        action = result.get("action", "Allow")
        is_safe = result.get("is_safe", True)
        
        # Extract new fields for rich result processing
        severity = result.get("severity")
        classifications = result.get("classifications")
        rules_data = result.get("rules")
        explanation = result.get("explanation")
        event_id = result.get("event_id")
        
        # Build reasons from rules that triggered
        reasons: List[str] = []
        for rule in (rules_data or []):
            classification = rule.get("classification")
            if classification and classification != "NONE_VIOLATION":
                rule_name = rule.get("rule_name", "Unknown")
                reasons.append(f"{rule_name}: {classification}")
        
        # If no specific rules triggered but is_safe is false, add generic reason
        if not reasons and not is_safe:
            sev = severity or "UNKNOWN"
            attack_technique = result.get("attack_technique", "")
            if explanation:
                reasons.append(explanation)
            elif attack_technique and attack_technique != "NONE_ATTACK_TECHNIQUE":
                reasons.append(f"Attack technique: {attack_technique}")
            else:
                reasons.append(f"Unsafe content detected (severity: {sev})")
        
        # Log block decisions for debugging
        if action == "Block" or not is_safe:
            logger.debug(f"MCP Inspection BLOCK response: {response_data}")
        
        # Decision kwargs with all new fields
        decision_kwargs = {
            "reasons": reasons,
            "raw_response": response_data,
            "severity": severity,
            "classifications": classifications,
            "rules": rules_data,
            "explanation": explanation,
            "event_id": event_id,
        }
        
        # Decision based on action OR is_safe
        if action == "Block" or not is_safe:
            return Decision.block(**decision_kwargs)
        else:
            return Decision.allow(**decision_kwargs)
    
    def _get_backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay for a retry attempt."""
        if self.retry_backoff <= 0:
            return 0.0
        delay = self.retry_backoff * (2 ** attempt)
        return min(delay, self.MAX_BACKOFF_DELAY)
    
    def _should_retry(self, error: Exception) -> bool:
        """Determine if a request should be retried based on the error."""
        import json
        
        # Never retry on JSON decode errors (response is malformed, not transient)
        if isinstance(error, json.JSONDecodeError):
            logger.warning(f"JSON decode error (not retryable): {error}")
            return False
        
        if isinstance(error, (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError)):
            return True
        if isinstance(error, httpx.HTTPStatusError):
            return error.response.status_code in self.retry_status_codes
        return False
    
    def _handle_error(
        self,
        error: Exception,
        tool_name: str,
        context: Optional[str] = None,
    ) -> Decision:
        """
        Handle API errors based on fail_open config.
        
        Args:
            error: The exception that occurred
            tool_name: Name of the tool being inspected
            context: Optional context string (e.g., "inspect_request")
            
        Returns:
            Decision.allow() if fail_open
            
        Raises:
            InspectionTimeoutError: If fail_open=False and error is a timeout
            InspectionNetworkError: If fail_open=False and error is a network error
            SecurityPolicyError: If fail_open=False for other errors
        """
        error_type = type(error).__name__
        error_msg = str(error)
        
        ctx_str = f" [{context}]" if context else ""
        logger.warning(f"MCP inspection error for tool={tool_name}{ctx_str}: {error_type}: {error_msg}")
        logger.debug(f"Error details: {error}", exc_info=True)
        
        if self.fail_open:
            logger.warning(f"mcp_fail_open=True, allowing tool call '{tool_name}' despite error")
            return Decision.allow(reasons=[f"MCP inspection error ({error_type}), fail_open=True"])
        else:
            logger.error(f"mcp_fail_open=False, blocking tool call '{tool_name}' due to error")
            
            # Raise typed exceptions based on error type
            if isinstance(error, httpx.TimeoutException):
                raise InspectionTimeoutError(
                    f"MCP inspection timed out after {self.timeout_ms}ms: {error_msg}",
                    timeout_ms=self.timeout_ms,
                ) from error
            
            if isinstance(error, (httpx.ConnectError, httpx.NetworkError)):
                raise InspectionNetworkError(
                    f"Failed to connect to MCP inspection API: {error_msg}"
                ) from error
            
            decision = Decision.block(reasons=[f"MCP inspection error: {error_type}: {error_msg}"])
            raise SecurityPolicyError(decision, f"MCP inspection failed and fail_open=False: {error_msg}") from error
    
    def inspect_request(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        metadata: Dict[str, Any],
        method: str = "tools/call",
    ) -> Decision:
        """
        Inspect an MCP request before execution (sync).
        
        Sends the request to Cisco AI Defense MCP Inspection API for
        security analysis before execution.
        
        Args:
            tool_name: Name of the tool/prompt/resource being accessed
            arguments: Arguments passed to the operation
            metadata: Additional metadata about the request (not sent to API)
            method: MCP method (tools/call, prompts/get, resources/read)
            
        Returns:
            Decision indicating whether to allow or block the request
            
        Raises:
            SecurityPolicyError: If fail_open=False and API is unreachable
        """
        # If no API configured, allow by default (backward compatible)
        if not self.endpoint or not self.api_key:
            logger.debug(f"MCP request intercepted: {method}={tool_name}, allowing by default (no API configured)")
            return Decision.allow()
        
        # Build JSON-RPC request message
        mcp_message = self._build_request_message(tool_name, arguments, method)
        logger.debug(f"MCP inspection request: {method}={tool_name}")
        
        headers = {
            "X-Cisco-AI-Defense-API-Key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        last_error: Optional[Exception] = None
        
        for attempt in range(self.retry_total):
            try:
                response = self._sync_client.post(
                    f"{self.endpoint}/api/v1/inspect/mcp",
                    json=mcp_message,
                    headers=headers,
                )
                response.raise_for_status()
                return self._parse_mcp_response(response.json())
            except Exception as e:
                last_error = e
                logger.debug(f"Attempt {attempt + 1}/{self.retry_total} failed: {e}")
                
                is_last_attempt = attempt >= self.retry_total - 1
                if is_last_attempt or not self._should_retry(e):
                    break
                
                delay = self._get_backoff_delay(attempt)
                if delay > 0:
                    logger.debug(f"Retrying in {delay:.2f}s...")
                    time.sleep(delay)
        
        return self._handle_error(last_error, tool_name, context="inspect_request")  # type: ignore
    
    def inspect_response(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        metadata: Dict[str, Any],
        method: str = "tools/call",
    ) -> Decision:
        """
        Inspect an MCP response after execution (sync).
        
        Sends the response to Cisco AI Defense MCP Inspection API for
        security analysis after execution.
        
        Args:
            tool_name: Name of the tool/prompt/resource that was accessed
            arguments: Arguments that were passed to the operation
            result: The result returned by the operation
            metadata: Additional metadata about the request (not sent to API)
            method: MCP method (tools/call, prompts/get, resources/read)
            
        Returns:
            Decision indicating whether to allow or block the response
            
        Raises:
            SecurityPolicyError: If fail_open=False and API is unreachable
        """
        # If no API configured, allow by default (backward compatible)
        if not self.endpoint or not self.api_key:
            logger.debug(f"MCP response intercepted: {method}={tool_name}, allowing by default (no API configured)")
            return Decision.allow()
        
        # Build JSON-RPC response message
        mcp_message = self._build_response_message(result)
        logger.debug(f"MCP inspection response: {method}={tool_name}")
        
        headers = {
            "X-Cisco-AI-Defense-API-Key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        last_error: Optional[Exception] = None
        
        for attempt in range(self.retry_total):
            try:
                response = self._sync_client.post(
                    f"{self.endpoint}/api/v1/inspect/mcp",
                    json=mcp_message,
                    headers=headers,
                )
                response.raise_for_status()
                return self._parse_mcp_response(response.json())
            except Exception as e:
                last_error = e
                logger.debug(f"Attempt {attempt + 1}/{self.retry_total} failed: {e}")
                
                is_last_attempt = attempt >= self.retry_total - 1
                if is_last_attempt or not self._should_retry(e):
                    break
                
                delay = self._get_backoff_delay(attempt)
                if delay > 0:
                    logger.debug(f"Retrying in {delay:.2f}s...")
                    time.sleep(delay)
        
        return self._handle_error(last_error, tool_name, context="inspect_response")  # type: ignore
    
    async def ainspect_request(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        metadata: Dict[str, Any],
        method: str = "tools/call",
    ) -> Decision:
        """
        Inspect an MCP request before execution (async).
        
        Args:
            tool_name: Name of the tool/prompt/resource being accessed
            arguments: Arguments passed to the operation
            metadata: Additional metadata about the request (not sent to API)
            method: MCP method (tools/call, prompts/get, resources/read)
            
        Returns:
            Decision indicating whether to allow or block the request
            
        Raises:
            SecurityPolicyError: If fail_open=False and API is unreachable
        """
        # If no API configured, allow by default (backward compatible)
        if not self.endpoint or not self.api_key:
            logger.debug(f"MCP request intercepted: {method}={tool_name}, allowing by default (no API configured)")
            return Decision.allow()
        
        # Build JSON-RPC request message
        mcp_message = self._build_request_message(tool_name, arguments, method)
        logger.debug(f"MCP async inspection request: {method}={tool_name}")
        
        headers = {
            "X-Cisco-AI-Defense-API-Key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        import asyncio
        last_error: Optional[Exception] = None
        
        # Reuse async client for better performance
        client = self._get_async_client()
        
        for attempt in range(self.retry_total):
            try:
                response = await client.post(
                    f"{self.endpoint}/api/v1/inspect/mcp",
                    json=mcp_message,
                    headers=headers,
                )
                response.raise_for_status()
                return self._parse_mcp_response(response.json())
            except Exception as e:
                last_error = e
                logger.debug(f"Attempt {attempt + 1}/{self.retry_total} failed: {e}")
                
                is_last_attempt = attempt >= self.retry_total - 1
                if is_last_attempt or not self._should_retry(e):
                    break
                
                delay = self._get_backoff_delay(attempt)
                if delay > 0:
                    logger.debug(f"Retrying in {delay:.2f}s...")
                    await asyncio.sleep(delay)
        
        return self._handle_error(last_error, tool_name, context="ainspect_request")  # type: ignore
    
    async def ainspect_response(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        metadata: Dict[str, Any],
        method: str = "tools/call",
    ) -> Decision:
        """
        Inspect an MCP response after execution (async).
        
        Args:
            tool_name: Name of the tool/prompt/resource that was accessed
            arguments: Arguments that were passed to the operation
            result: The result returned by the operation
            metadata: Additional metadata about the request (not sent to API)
            method: MCP method (tools/call, prompts/get, resources/read)
            
        Returns:
            Decision indicating whether to allow or block the response
            
        Raises:
            SecurityPolicyError: If fail_open=False and API is unreachable
        """
        # If no API configured, allow by default (backward compatible)
        if not self.endpoint or not self.api_key:
            logger.debug(f"MCP response intercepted: {method}={tool_name}, allowing by default (no API configured)")
            return Decision.allow()
        
        # Build JSON-RPC response message
        mcp_message = self._build_response_message(result)
        logger.debug(f"MCP async inspection response: {method}={tool_name}")
        
        headers = {
            "X-Cisco-AI-Defense-API-Key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        import asyncio
        last_error: Optional[Exception] = None
        
        # Reuse async client for better performance
        client = self._get_async_client()
        
        for attempt in range(self.retry_total):
            try:
                response = await client.post(
                    f"{self.endpoint}/api/v1/inspect/mcp",
                    json=mcp_message,
                    headers=headers,
                )
                response.raise_for_status()
                return self._parse_mcp_response(response.json())
            except Exception as e:
                last_error = e
                logger.debug(f"Attempt {attempt + 1}/{self.retry_total} failed: {e}")
                
                is_last_attempt = attempt >= self.retry_total - 1
                if is_last_attempt or not self._should_retry(e):
                    break
                
                delay = self._get_backoff_delay(attempt)
                if delay > 0:
                    logger.debug(f"Retrying in {delay:.2f}s...")
                    await asyncio.sleep(delay)
        
        return self._handle_error(last_error, tool_name, context="ainspect_response")  # type: ignore
    
    def close(self) -> None:
        """Close HTTP clients."""
        try:
            self._sync_client.close()
        except Exception as e:
            logger.warning(f"Error closing sync HTTP client: {e}")
        
        # Clear async client reference
        with self._async_client_lock:
            self._async_client = None
            self._async_loop_id = None
    
    async def aclose(self) -> None:
        """Close async resources."""
        with self._async_client_lock:
            if self._async_client is not None:
                try:
                    await self._async_client.aclose()
                except Exception as e:
                    logger.debug(f"Error closing async HTTP client: {e}")
                finally:
                    self._async_client = None
                    self._async_loop_id = None
