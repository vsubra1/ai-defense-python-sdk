# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""LLM Inspector for Cisco AI Defense Chat Inspection API."""

import logging
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

logger = logging.getLogger("aidefense.runtime.agentsec.inspectors.llm")


class LLMInspector:
    """
    Inspector for LLM conversations using Cisco AI Defense Chat Inspection API.
    
    This class integrates with the Cisco AI Defense Chat Inspection API to
    inspect LLM requests and responses for security policy violations.
    
    See: https://developer.cisco.com/docs/ai-defense/overview/
    
    Attributes:
        api_key: API key for Cisco AI Defense
        endpoint: Base URL for the AI Defense API
        timeout_ms: Request timeout in milliseconds
        retry_total: Total number of retry attempts (default 1 = no retry)
        retry_backoff: Exponential backoff factor in seconds (default 0 = no backoff)
        retry_status_codes: HTTP status codes to retry on
        fail_open: Whether to allow requests when API is unreachable
        entity_types: Entity types to filter for (e.g., ["EMAIL", "PHONE_NUMBER"])
    """
    
    # Maximum backoff delay to prevent runaway waits
    MAX_BACKOFF_DELAY = 30.0
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        default_rules: Optional[List[Any]] = None,
        entity_types: Optional[List[str]] = None,
        timeout_ms: Optional[int] = None,
        retry_attempts: Optional[int] = None,  # Deprecated, use retry_total
        retry_total: Optional[int] = None,
        retry_backoff: Optional[float] = None,
        retry_status_codes: Optional[List[int]] = None,
        pool_max_connections: Optional[int] = None,
        pool_max_keepalive: Optional[int] = None,
        fail_open: bool = True,
    ):
        """
        Initialize the LLM Inspector.
        
        Args:
            api_key: API key for Cisco AI Defense (or from AI_DEFENSE_API_MODE_LLM_API_KEY env)
            endpoint: Base URL for the AI Defense API (or from AI_DEFENSE_API_MODE_LLM_ENDPOINT env)
            default_rules: Default rules for inspection
            entity_types: Entity types to filter for (e.g., ["EMAIL", "PHONE_NUMBER"])
            timeout_ms: Request timeout in milliseconds (default 1000)
            retry_attempts: Deprecated, use retry_total instead
            retry_total: Total number of retry attempts (default 1, no retry)
            retry_backoff: Exponential backoff factor in seconds (default 0, no backoff)
            retry_status_codes: HTTP status codes to retry on (default [500, 502, 503, 504])
            pool_max_connections: Maximum connections in the pool (default 100)
            pool_max_keepalive: Maximum keepalive connections (default 20)
            fail_open: Whether to allow requests when API is unreachable (default True)
        """
        import os
        from .. import _state
        
        # Priority: explicit param > state > env var
        self.api_key = api_key or _state.get_api_mode_llm_api_key() or os.environ.get("AI_DEFENSE_API_MODE_LLM_API_KEY")
        self.endpoint = endpoint or _state.get_api_mode_llm_endpoint() or os.environ.get("AI_DEFENSE_API_MODE_LLM_ENDPOINT")
        self.default_rules = default_rules or []
        self.entity_types = entity_types or _state.get_llm_entity_types()
        self.fail_open = fail_open
        
        # Timeout: explicit param > state > default (1000ms)
        if timeout_ms is not None:
            self.timeout_ms = timeout_ms
        else:
            state_timeout = _state.get_timeout()
            # State stores timeout in seconds, convert to ms
            self.timeout_ms = (state_timeout * 1000) if state_timeout is not None else 1000
        
        # Retry configuration: explicit param > state > default
        # Handle deprecated retry_attempts parameter
        if retry_total is not None:
            self.retry_total = max(1, retry_total)
        elif retry_attempts is not None:
            # Deprecated parameter - use it but log warning
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
        
        # Create sync HTTP client with configured limits
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
            
            old_client = None
            # Close old client if it exists (different loop)
            if self._async_client is not None:
                old_client = self._async_client
                logger.debug("Replacing async client from different event loop")
            
            timeout = httpx.Timeout(self.timeout_ms / 1000.0)
            limits = httpx.Limits(
                max_connections=self.pool_max_connections,
                max_keepalive_connections=self.pool_max_keepalive,
            )
            self._async_client = httpx.AsyncClient(timeout=timeout, limits=limits, http2=False)
            self._async_loop_id = loop_id
            logger.debug(f"Created new async HTTP client for loop {loop_id}")
        
        # Attempt to close old client outside the lock to avoid blocking
        if old_client is not None:
            self._close_stale_async_client(old_client)
        
        return self._async_client
    
    def _close_stale_async_client(self, client: httpx.AsyncClient) -> None:
        """
        Attempt to close a stale async client that was created for a different event loop.
        
        This is a best-effort cleanup - if the client can't be closed properly,
        it will be garbage collected with its connections.
        """
        try:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                # Schedule close on current loop - this is async-safe
                loop.create_task(self._safe_aclose(client))
            except RuntimeError:
                # No running loop - try sync close via internal transport
                # httpx.AsyncClient doesn't have a sync close, so just log and let GC handle it
                logger.warning(
                    "Stale async HTTP client from different event loop could not be closed properly. "
                    "Connections may leak until garbage collection."
                )
        except Exception as e:
            logger.warning(f"Error closing stale async client: {e}")
    
    async def _safe_aclose(self, client: httpx.AsyncClient) -> None:
        """Safely close an async client, catching any errors."""
        try:
            await client.aclose()
            logger.debug("Successfully closed stale async client")
        except Exception as e:
            logger.debug(f"Error closing stale async client: {e}")
    
    def _build_request_payload(
        self,
        messages: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build the Chat Inspection API request payload.
        
        Args:
            messages: List of conversation messages with role and content
            metadata: Additional metadata (user, src_app, transaction_id, etc.)
            
        Returns:
            Request payload dict for the API
        """
        payload: Dict[str, Any] = {
            "messages": messages,
            "metadata": metadata,
        }
        
        # Build rules with entity_types if configured
        if self.default_rules:
            rules = []
            for rule in self.default_rules:
                if isinstance(rule, dict):
                    # Rule is already a dict (e.g., {"rule_name": "PII", "entity_types": [...]})
                    rule_dict = dict(rule)
                else:
                    # Rule is a string (e.g., "PII")
                    rule_dict = {"rule_name": rule}
                
                # Add entity_types if configured and not already in rule
                if self.entity_types and "entity_types" not in rule_dict:
                    rule_dict["entity_types"] = self.entity_types
                
                rules.append(rule_dict)
            payload["rules"] = rules
        elif self.entity_types:
            # No rules configured, but entity_types are - this might be used
            # by the API to filter entities across all rules
            payload["entity_types"] = self.entity_types
        
        return payload
    
    def _parse_response(self, response_data: Dict[str, Any]) -> Decision:
        """
        Parse the API response into a Decision.
        
        Args:
            response_data: JSON response from the API
            
        Returns:
            Decision based on API response
        """
        # The API returns action capitalized (Allow, Block, etc.) - normalize to lowercase
        action = response_data.get("action", "allow").lower()
        reasons = response_data.get("reasons", [])
        sanitized_content = response_data.get("sanitized_content")
        
        # Extract new fields for rich result processing
        severity = response_data.get("severity")
        classifications = response_data.get("classifications")
        explanation = response_data.get("explanation")
        event_id = response_data.get("event_id")
        
        # Parse rules from response
        rules_data = response_data.get("rules") or response_data.get("processed_rules")
        
        # Log full response for debugging block decisions
        if action == "block":
            logger.debug(f"AI Defense BLOCK response: {response_data}")
        
        # Extract reasons from "rules" field (primary source of violations)
        if not reasons and rules_data:
            for rule in rules_data:
                classification = rule.get("classification")
                if classification and classification not in ("NONE_VIOLATION", "NONE_SEVERITY"):
                    reasons.append(f"{rule.get('rule_name')}: {classification}")
        
        # Map API action to Decision with all fields
        decision_kwargs = {
            "reasons": reasons,
            "raw_response": response_data,
            "severity": severity,
            "classifications": classifications,
            "rules": rules_data,
            "explanation": explanation,
            "event_id": event_id,
        }
        
        if action == "block":
            return Decision.block(**decision_kwargs)
        elif action == "sanitize":
            return Decision.sanitize(
                sanitized_content=sanitized_content,
                **decision_kwargs,
            )
        elif action == "monitor_only":
            return Decision.monitor_only(**decision_kwargs)
        else:
            return Decision.allow(**decision_kwargs)
    
    def _get_backoff_delay(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay for a retry attempt.
        
        Args:
            attempt: The attempt number (0-indexed)
            
        Returns:
            Delay in seconds, capped at MAX_BACKOFF_DELAY
        """
        if self.retry_backoff <= 0:
            return 0.0
        
        # Exponential backoff: backoff_factor * (2 ** attempt)
        delay = self.retry_backoff * (2 ** attempt)
        
        # Cap the delay to prevent runaway waits
        return min(delay, self.MAX_BACKOFF_DELAY)
    
    def _should_retry(self, error: Exception, status_code: Optional[int] = None) -> bool:
        """
        Determine if a request should be retried based on the error.
        
        Args:
            error: The exception that occurred
            status_code: Optional HTTP status code from the response
            
        Returns:
            True if the request should be retried
        """
        import json
        
        # Never retry on JSON decode errors (response is malformed, not transient)
        if isinstance(error, json.JSONDecodeError):
            logger.warning(f"JSON decode error (not retryable): {error}")
            return False
        
        # Always retry on timeout or network errors
        if isinstance(error, (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError)):
            return True
        
        # Retry on configured status codes
        if isinstance(error, httpx.HTTPStatusError):
            return error.response.status_code in self.retry_status_codes
        
        # Don't retry on other errors
        return False
    
    def _handle_error(
        self,
        error: Exception,
        context: Optional[str] = None,
        message_count: int = 0,
    ) -> Decision:
        """
        Handle API errors based on fail_open config.
        
        Centralizes error handling for all API-related errors. Logs with context
        and respects fail_open setting. When fail_open=False, raises typed
        exceptions (InspectionTimeoutError, InspectionNetworkError) to allow
        callers to handle specific error types.
        
        Args:
            error: The exception that occurred
            context: Optional context string (e.g., "inspect_conversation")
            message_count: Number of messages in the request for logging
            
        Returns:
            Decision.allow() if fail_open
            
        Raises:
            InspectionTimeoutError: If fail_open=False and error is a timeout
            InspectionNetworkError: If fail_open=False and error is a network error
            SecurityPolicyError: If fail_open=False for other errors
        """
        # Classify the error type for better logging
        error_type = type(error).__name__
        error_msg = str(error)
        
        # Build context string for logging
        ctx_parts = []
        if context:
            ctx_parts.append(f"operation={context}")
        if message_count > 0:
            ctx_parts.append(f"messages={message_count}")
        ctx_str = f" [{', '.join(ctx_parts)}]" if ctx_parts else ""
        
        # Log at WARNING level (will be upgraded to ERROR if fail_open=False)
        logger.warning(f"AI Defense API error{ctx_str}: {error_type}: {error_msg}")
        
        # Log stack trace at DEBUG level only
        logger.debug(f"Error details: {error}", exc_info=True)
        
        if self.fail_open:
            logger.warning("fail_open=True, allowing request despite API error")
            return Decision.allow(reasons=[f"API error ({error_type}), fail_open=True"])
        else:
            logger.error("fail_open=False, blocking request due to API error")
            
            # Raise typed exceptions based on error type
            if isinstance(error, httpx.TimeoutException):
                raise InspectionTimeoutError(
                    f"Inspection timed out after {self.timeout_ms}ms: {error_msg}",
                    timeout_ms=self.timeout_ms,
                ) from error
            
            if isinstance(error, (httpx.ConnectError, httpx.NetworkError)):
                raise InspectionNetworkError(
                    f"Failed to connect to inspection API: {error_msg}"
                ) from error
            
            # For other errors, raise SecurityPolicyError
            decision = Decision.block(reasons=[f"API error: {error_type}: {error_msg}"])
            raise SecurityPolicyError(decision, f"AI Defense API unavailable and fail_open=False: {error_msg}") from error
    
    def inspect_conversation(
        self,
        messages: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> Decision:
        """
        Inspect an LLM conversation for security violations (sync).
        
        Args:
            messages: List of conversation messages with role and content
            metadata: Additional metadata (user, src_app, transaction_id, etc.)
            
        Returns:
            Decision indicating whether to allow, block, or sanitize
            
        Raises:
            InspectionTimeoutError: If fail_open=False and request times out
            InspectionNetworkError: If fail_open=False and network error occurs
            SecurityPolicyError: If fail_open=False and other API errors occur
        """
        if not self.endpoint or not self.api_key:
            logger.debug("No API endpoint/key configured, allowing by default")
            return Decision.allow()
        
        logger.info(f"Request inspection: {len(messages)} messages")
        payload = self._build_request_payload(messages, metadata)
        logger.debug(f"AI Defense request: {len(messages)} messages, metadata={list(metadata.keys())}")
        logger.debug(f"AI Defense request payload: {payload}")
        headers = {
            "X-Cisco-AI-Defense-API-Key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        last_error: Optional[Exception] = None
        
        for attempt in range(self.retry_total):
            try:
                response = self._sync_client.post(
                    f"{self.endpoint}/v1/inspect/chat",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                response_json = response.json()
                logger.debug(f"AI Defense response: {response_json}")
                decision = self._parse_response(response_json)
                logger.info(f"Request decision: {decision.action}")
                return decision
            except Exception as e:
                last_error = e
                logger.debug(f"Attempt {attempt + 1}/{self.retry_total} failed: {e}")
                
                # Check if we should retry
                is_last_attempt = attempt >= self.retry_total - 1
                if is_last_attempt or not self._should_retry(e):
                    break
                
                # Apply exponential backoff before next retry
                delay = self._get_backoff_delay(attempt)
                if delay > 0:
                    logger.debug(f"Retrying in {delay:.2f}s...")
                    time.sleep(delay)
        
        return self._handle_error(
            last_error,  # type: ignore
            context="inspect_conversation",
            message_count=len(messages),
        )
    
    async def ainspect_conversation(
        self,
        messages: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> Decision:
        """
        Inspect an LLM conversation for security violations (async).
        
        Args:
            messages: List of conversation messages with role and content
            metadata: Additional metadata (user, src_app, transaction_id, etc.)
            
        Returns:
            Decision indicating whether to allow, block, or sanitize
            
        Raises:
            InspectionTimeoutError: If fail_open=False and request times out
            InspectionNetworkError: If fail_open=False and network error occurs
            SecurityPolicyError: If fail_open=False and other API errors occur
        """
        import asyncio
        
        if not self.endpoint or not self.api_key:
            logger.debug("No API endpoint/key configured, allowing by default")
            return Decision.allow()
        
        logger.info(f"Request inspection: {len(messages)} messages")
        payload = self._build_request_payload(messages, metadata)
        logger.debug(f"AI Defense async request: {len(messages)} messages, metadata={list(metadata.keys())}")
        logger.debug(f"AI Defense async request payload: {payload}")
        headers = {
            "X-Cisco-AI-Defense-API-Key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        last_error: Optional[Exception] = None
        
        # Reuse async client per event loop to improve performance
        client = self._get_async_client()
        
        for attempt in range(self.retry_total):
            try:
                response = await client.post(
                    f"{self.endpoint}/v1/inspect/chat",
                    json=payload,
                    headers=headers,
                )
                if response.status_code != 200:
                    logger.debug(f"AI Defense async response error: {response.status_code} - {response.text[:500]}")
                response.raise_for_status()
                response_json = response.json()
                logger.debug(f"AI Defense async response: {response_json}")
                decision = self._parse_response(response_json)
                logger.info(f"Request decision: {decision.action}")
                return decision
            except Exception as e:
                last_error = e
                logger.debug(f"Attempt {attempt + 1}/{self.retry_total} failed: {e}")
                
                # Check if we should retry
                is_last_attempt = attempt >= self.retry_total - 1
                if is_last_attempt or not self._should_retry(e):
                    break
                
                # Apply exponential backoff before next retry (async sleep)
                delay = self._get_backoff_delay(attempt)
                if delay > 0:
                    logger.debug(f"Retrying in {delay:.2f}s...")
                    await asyncio.sleep(delay)
        
        return self._handle_error(
            last_error,  # type: ignore
            context="ainspect_conversation",
            message_count=len(messages),
        )
    
    def close(self) -> None:
        """
        Close HTTP clients.
        
        Closes the sync client immediately. For the async client, attempts
        to close it if we're in an event loop context, otherwise marks it
        for garbage collection.
        """
        # Close sync client
        try:
            self._sync_client.close()
        except Exception as e:
            logger.warning(f"Error closing sync HTTP client: {e}")
        
        # Attempt to close async client if it exists
        with self._async_client_lock:
            if self._async_client is not None:
                client_to_close = self._async_client
                self._async_client = None
                self._async_loop_id = None
                
                try:
                    import asyncio
                    # Try to get the running loop - if we're in async context, schedule close
                    try:
                        loop = asyncio.get_running_loop()
                        # Schedule the close on the current loop
                        loop.create_task(self._safe_aclose(client_to_close))
                    except RuntimeError:
                        # No running loop - just clear the reference
                        # The client will be garbage collected
                        logger.debug("No running event loop, async client will be garbage collected")
                except Exception as e:
                    logger.warning(f"Error scheduling async client close: {e}")
    
    async def aclose(self) -> None:
        """Close async resources."""
        if self._async_client is not None:
            try:
                await self._async_client.aclose()
            except Exception as e:
                logger.debug(f"Error closing async HTTP client: {e}")
            finally:
                self._async_client = None
                self._async_loop_id = None
