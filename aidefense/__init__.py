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
AI Defense Python SDK
Convenient imports for all major SDK components.

This module uses lazy imports to avoid loading dependencies at package import time.
This is particularly important for deployment environments like Vertex AI Agent Engine
where the package may be bundled before dependencies are installed.

NOTE: Consider reverting to direct imports when raising the final PR if:
      - The real fix (requirements.txt packaging in agent code) is sufficient
      - Lazy imports add unnecessary complexity to the SDK
      - Performance testing shows no benefit
      
      Original imports were:
        from .runtime import *
        from .config import Config, AsyncConfig
        from .exceptions import ValidationError, ApiError, SDKError
        from .modelscan import ModelScanClient
        from .management import (...)
"""

import sys
from typing import Any


# Define what should be available when doing `from aidefense import *`
__all__ = [
    # runtime
    "LLMClient",
    "AsyncLLMClient",
    "MCPClient",
    "AsyncMCPClient",
    # config
    "Config",
    "AsyncConfig",
    # exceptions
    "ValidationError",
    "ApiError",
    "SDKError",
    # modelscan
    "ModelScanClient",
    # management
    "ManagementClient",
    "ApplicationManagementClient",
    "ConnectionManagementClient",
    "PolicyManagementClient",
    "EventManagementClient",
]


# Module-level lazy import handler
def __getattr__(name: str) -> Any:
    """
    Lazy import handler that loads modules only when attributes are accessed.
    This prevents importing dependencies (like aiohttp) at package import time.
    """
    # Map attribute names to their source modules
    _import_map = {
        # From runtime
        "LLMClient": ("aidefense.runtime", "LLMClient"),
        "AsyncLLMClient": ("aidefense.runtime", "AsyncLLMClient"),
        "MCPClient": ("aidefense.runtime", "MCPClient"),
        "AsyncMCPClient": ("aidefense.runtime", "AsyncMCPClient"),
        # From config
        "Config": ("aidefense.config", "Config"),
        "AsyncConfig": ("aidefense.config", "AsyncConfig"),
        # From exceptions
        "ValidationError": ("aidefense.exceptions", "ValidationError"),
        "ApiError": ("aidefense.exceptions", "ApiError"),
        "SDKError": ("aidefense.exceptions", "SDKError"),
        # From modelscan
        "ModelScanClient": ("aidefense.modelscan", "ModelScanClient"),
        # From management
        "ManagementClient": ("aidefense.management", "ManagementClient"),
        "ApplicationManagementClient": ("aidefense.management", "ApplicationManagementClient"),
        "ConnectionManagementClient": ("aidefense.management", "ConnectionManagementClient"),
        "PolicyManagementClient": ("aidefense.management", "PolicyManagementClient"),
        "EventManagementClient": ("aidefense.management", "EventManagementClient"),
    }
    
    if name in _import_map:
        module_name, attr_name = _import_map[name]
        # Import the module
        import importlib
        module = importlib.import_module(module_name)
        # Get the attribute from the module
        attr = getattr(module, attr_name)
        # Cache it in this module's namespace for future access
        globals()[name] = attr
        return attr
    
    raise AttributeError(f"module 'aidefense' has no attribute '{name}'")
