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

"""
Runtime module with lazy imports to avoid loading dependencies at import time.

NOTE: Consider reverting to direct imports when raising the final PR if:
      - The real fix (requirements.txt packaging + lazy agentsec.protect()) is sufficient
      - Lazy imports add unnecessary complexity
      - Testing shows deployment works without these lazy imports
      
      Original imports were:
        from .http_inspect import HttpInspectionClient
        from .chat_inspect import ChatInspectionClient, Message, Role, ChatInspectRequest
        from .mcp_inspect import MCPInspectionClient
        from .models import (Action, Rule, Classification, ...)
        from .http_models import (HttpInspectRequest, HttpReqObject, ...)
        from .mcp_models import (MCPMessage, MCPError, ...)
        from .utils import to_base64_bytes
        from .agentsec import (protect, get_patched_clients, ...)
"""

from typing import Any


__all__ = [
    # http_inspect
    "HttpInspectionClient",
    # chat_inspect
    "ChatInspectionClient",
    "Message",
    "Role",
    "ChatInspectRequest",
    # mcp_inspect
    "MCPInspectionClient",
    # models
    "Action",
    "Rule",
    "Classification",
    "RuleName",
    "InspectionConfig",
    "Metadata",
    "InspectResponse",
    # http_models
    "HttpInspectRequest",
    "HttpReqObject",
    "HttpResObject",
    "HttpMetaObject",
    "HttpHdrObject",
    "HttpHdrKvObject",
    # mcp_models
    "MCPMessage",
    "MCPError",
    "MCPInspectResponse",
    "MCPInspectError",
    # utils
    "to_base64_bytes",
    # agentsec
    "protect",
    "get_patched_clients",
    "Decision",
    "SecurityPolicyError",
    "skip_inspection",
    "no_inspection",
    # Special attributes for agentsec module
    "agentsec",
]


def __getattr__(name: str) -> Any:
    """Lazy import handler for runtime module."""
    import importlib
    
    # Map of attribute names to (module_name, attr_name)
    _import_map = {
        # http_inspect
        "HttpInspectionClient": ("aidefense.runtime.http_inspect", "HttpInspectionClient"),
        # chat_inspect
        "ChatInspectionClient": ("aidefense.runtime.chat_inspect", "ChatInspectionClient"),
        "Message": ("aidefense.runtime.chat_inspect", "Message"),
        "Role": ("aidefense.runtime.chat_inspect", "Role"),
        "ChatInspectRequest": ("aidefense.runtime.chat_inspect", "ChatInspectRequest"),
        # mcp_inspect
        "MCPInspectionClient": ("aidefense.runtime.mcp_inspect", "MCPInspectionClient"),
        # models
        "Action": ("aidefense.runtime.models", "Action"),
        "Rule": ("aidefense.runtime.models", "Rule"),
        "Classification": ("aidefense.runtime.models", "Classification"),
        "RuleName": ("aidefense.runtime.models", "RuleName"),
        "InspectionConfig": ("aidefense.runtime.models", "InspectionConfig"),
        "Metadata": ("aidefense.runtime.models", "Metadata"),
        "InspectResponse": ("aidefense.runtime.models", "InspectResponse"),
        # http_models
        "HttpInspectRequest": ("aidefense.runtime.http_models", "HttpInspectRequest"),
        "HttpReqObject": ("aidefense.runtime.http_models", "HttpReqObject"),
        "HttpResObject": ("aidefense.runtime.http_models", "HttpResObject"),
        "HttpMetaObject": ("aidefense.runtime.http_models", "HttpMetaObject"),
        "HttpHdrObject": ("aidefense.runtime.http_models", "HttpHdrObject"),
        "HttpHdrKvObject": ("aidefense.runtime.http_models", "HttpHdrKvObject"),
        # mcp_models
        "MCPMessage": ("aidefense.runtime.mcp_models", "MCPMessage"),
        "MCPError": ("aidefense.runtime.mcp_models", "MCPError"),
        "MCPInspectResponse": ("aidefense.runtime.mcp_models", "MCPInspectResponse"),
        "MCPInspectError": ("aidefense.runtime.mcp_models", "MCPInspectError"),
        # utils
        "to_base64_bytes": ("aidefense.runtime.utils", "to_base64_bytes"),
        # agentsec
        "protect": ("aidefense.runtime.agentsec", "protect"),
        "get_patched_clients": ("aidefense.runtime.agentsec", "get_patched_clients"),
        "Decision": ("aidefense.runtime.agentsec", "Decision"),
        "SecurityPolicyError": ("aidefense.runtime.agentsec", "SecurityPolicyError"),
        "skip_inspection": ("aidefense.runtime.agentsec", "skip_inspection"),
        "no_inspection": ("aidefense.runtime.agentsec", "no_inspection"),
    }
    
    # Special case for agentsec module itself
    if name == "agentsec":
        module = importlib.import_module("aidefense.runtime.agentsec")
        globals()[name] = module
        return module
    
    if name in _import_map:
        module_name, attr_name = _import_map[name]
        module = importlib.import_module(module_name)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    
    raise AttributeError(f"module 'aidefense.runtime' has no attribute '{name}'")
