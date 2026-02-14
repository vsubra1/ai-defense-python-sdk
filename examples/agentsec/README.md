# agentsec Examples

**agentsec** accelerates your integration with **Cisco AI Defense** when building AI applications using:

- **Chat Completions** - Direct LLM calls (OpenAI, Azure, Bedrock, Vertex AI)
- **Agentic Frameworks** - LangChain, LangGraph, CrewAI, AutoGen, Strands, OpenAI Agents SDK
- **Agentic Runtimes (PaaS)** - AWS Bedrock AgentCore, GCP Vertex AI Agent Engine, Microsoft Azure AI Foundry

### How It Works

agentsec uses **dynamic code modification** to automatically intercept LLM and MCP tool calls, routing traffic through:

| Integration Mode | Description |
|------------------|-------------|
| **Cisco AI Defense Gateway** | Routes all LLM traffic through a secure proxy for centralized policy enforcement |
| **Cisco AI Defense API** | Inspects requests/responses via API calls while connecting directly to LLM providers |

With just **a few lines of code**, your existing AI application gains enterprise-grade security:

```python
from aidefense.runtime import agentsec
agentsec.protect(config="agentsec.yaml")  
```

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Environment Variables Quick Reference](#environment-variables-quick-reference)
- [Overview](#overview)
- [Core Concepts](#core-concepts)
- [Integration Pattern](#integration-pattern)
- [Directory Structure](#directory-structure)
- [1. Simple Examples](#1-simple-examples)
- [2. Agent Frameworks](#2-agent-frameworks)
- [3. Agent Runtimes](#3-agent-runtimes)
- [Configuration](#configuration)
- [Advanced Usage](#advanced-usage)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Requirement | Version | Installation |
|-------------|---------|--------------|
| Python | 3.10+ | [python.org](https://www.python.org/downloads/) |
| Poetry | 1.5+ | `curl -sSL https://install.python-poetry.org \| python3 -` |
| Git | Any | `brew install git` or [git-scm.com](https://git-scm.com/) |

---

## Quick Start

All paths below are relative to `examples/agentsec/`.

```bash
# 1. Configure credentials and settings
cp .env.example .env
# Edit .env with your API keys and provider credentials (secrets only)
# Edit agentsec.yaml for integration modes, gateway URLs, and inspection settings

# 2. Run a simple example
cd 1-simple && poetry install && poetry run python basic_protection.py

# 3. Or run an agent framework example
cd 2-agent-frameworks/strands-agent && poetry install && ./scripts/run.sh --openai

# 4. Run all tests (from the repo root: ai-defense-python-sdk/)
./scripts/run-unit-tests.sh           # ~1130 unit tests
./scripts/run-integration-tests.sh    # Full integration tests
```

---

## Environment Variables Quick Reference

Configuration is split between two files:

| File | Contains | Example |
|------|----------|---------|
| **`agentsec.yaml`** | Integration modes, gateway URLs, timeouts, inspection modes | `llm_integration_mode: gateway` |
| **`.env`** | Secrets only (API keys, credentials) | `OPENAI_API_KEY=sk-...` |

> **Tip**: `agentsec.yaml` can reference secrets from `.env` using `${VAR_NAME}` syntax.

### Core `.env` Variables

| Variable | Required | Description |
|----------|:--------:|-------------|
| `AI_DEFENSE_API_MODE_LLM_API_KEY` | Yes | AI Defense API key for LLM inspection |
| `AI_DEFENSE_API_MODE_MCP_API_KEY` | For MCP | AI Defense API key for MCP inspection |

> **Note**: Log level, endpoints, and inspection modes are configured in `agentsec.yaml` (not `.env`). Log level can optionally be overridden via the `AGENTSEC_LOG_LEVEL` environment variable.

### By Integration Mode

| Mode | Secrets in `.env` | Settings in `agentsec.yaml` |
|------|-------------------|----------------------------|
| **API Mode** | `AI_DEFENSE_API_MODE_LLM_API_KEY` | `api_mode.llm.endpoint`, `api_mode.llm.mode` |
| **Gateway Mode** | Provider-specific gateway API keys (see below) | `gateway_mode.llm_gateways.*` (URLs, providers) |

### By LLM Provider

| Provider | API Mode (`.env`) | Gateway Mode (`.env`) |
|----------|-------------------|----------------------|
| **OpenAI** | `OPENAI_API_KEY` | `OPENAI_API_KEY` (same key for both) |
| **Azure OpenAI** | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT_NAME` | `AGENTSEC_AZURE_OPENAI_GATEWAY_API_KEY` |
| **AWS Bedrock** | `AWS_REGION`, `AWS_PROFILE` (or access keys) | AWS Sig V4 (no API key needed) |
| **GCP Vertex AI** | `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` + ADC | ADC OAuth2 (no API key needed) |
| **Cohere** | `COHERE_API_KEY` | `AGENTSEC_COHERE_GATEWAY_API_KEY` |
| **Mistral AI** | `MISTRAL_API_KEY` | `AGENTSEC_MISTRAL_GATEWAY_API_KEY` |

> All gateway URLs are configured in `agentsec.yaml` (not `.env`). MCP server URLs are configured per-example or in test scripts.

### By Example Path

| Example | AI Defense | LLM Provider | Extra |
|---------|------------|--------------|-------|
| `1-simple/basic_protection.py` | API or Gateway | OpenAI | - |
| `1-simple/openai_example.py` | API or Gateway | OpenAI | - |
| `1-simple/streaming_example.py` | API or Gateway | OpenAI | - |
| `1-simple/gateway_mode_example.py` | Gateway | OpenAI | Gateway API key |
| `1-simple/skip_inspection_example.py` | API or Gateway | OpenAI | - |
| `1-simple/cohere_example.py` | API or Gateway | Cohere | - |
| `1-simple/mistral_example.py` | API or Gateway | Mistral AI | - |
| `1-simple/simple_strands_bedrock.py` | API or Gateway | Bedrock | - |
| `1-simple/mcp_example.py` | API or Gateway | OpenAI + MCP | MCP server URL |
| `2-agent-frameworks/*/--openai` | API or Gateway | OpenAI | - |
| `2-agent-frameworks/*/--azure` | API or Gateway | Azure OpenAI | - |
| `2-agent-frameworks/*/--bedrock` | API or Gateway | Bedrock | - |
| `2-agent-frameworks/*/--vertex` | API or Gateway | Vertex AI | - |
| `3-agent-runtimes/microsoft-foundry/` | API or Gateway | Azure OpenAI | `AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP`, etc. |
| `3-agent-runtimes/amazon-bedrock-agentcore/` | API or Gateway | Bedrock | AWS deployment vars |
| `3-agent-runtimes/gcp-vertex-ai-agent-engine/` | API or Gateway | Vertex AI | `GKE_AUTHORIZED_NETWORKS` |

> **Tip**: Check the section headers in `.env.example` - each section shows which examples require those variables.

---

## Overview

| Category | Description | Examples |
|----------|-------------|----------|
| **1-simple/** | Standalone examples for core features | 9 examples |
| **2-agent-frameworks/** | Agent frameworks with MCP tools | 6 frameworks |
| **3-agent-runtimes/** | Cloud deployment with AI Defense | 3 runtimes, 9 modes |

---

## Core Concepts

### Integration Modes

agentsec supports two ways to integrate with Cisco AI Defense:

| Mode | How It Works | When to Use |
|------|--------------|-------------|
| **API Mode** | SDK inspects requests via AI Defense API, then calls LLM directly | Most deployments; simple setup |
| **Gateway Mode** | SDK routes all traffic through AI Defense Gateway proxy | Centralized policy enforcement |

Set in `agentsec.yaml`:
```yaml
llm_integration_mode: gateway
mcp_integration_mode: api
```

### Supported LLM Providers

agentsec automatically patches these LLM client libraries:

| Provider | Package | Patched Methods |
|----------|---------|-----------------|
| **OpenAI** | `openai` | `chat.completions.create()` |
| **Azure OpenAI** | `openai` | `chat.completions.create()` (with Azure endpoint) |
| **AWS Bedrock** | `boto3` | `converse()`, `converse_stream()` |
| **Google Vertex AI** | `google-cloud-aiplatform` | `ChatVertexAI`, `generate_content()` |
| **Google GenAI** | `google-genai` | `generate_content()`, `generate_content_async()` |
| **Cohere** | `cohere` | `V2Client.chat()`, `V2Client.chat_stream()`, `AsyncV2Client.chat()`, `AsyncV2Client.chat_stream()` |
| **Mistral AI** | `mistralai` | `Chat.complete()`, `Chat.stream()`, `Chat.complete_async()`, `Chat.stream_async()` |

### MCP Tool Inspection

agentsec also patches the MCP (Model Context Protocol) client for tool call inspection:

| Package | Patched Methods |
|---------|-----------------|
| `mcp` | `ClientSession.call_tool()`, `get_prompt()`, `read_resource()` |

### Protection Coverage

agentsec inspects both **requests** (user prompts) and **responses** (LLM outputs):

| Inspection | What It Checks | When It Happens |
|------------|----------------|-----------------|
| **LLM Request** | Prompt injection, jailbreak, PII in prompts | Before LLM call |
| **LLM Response** | Sensitive data leakage, harmful content | After LLM response |
| **MCP Request** | Tool call arguments, prompt injection | Before MCP tool call |
| **MCP Response** | Tool response content, data leakage | After MCP tool response |

### Inspection Modes

| Mode | Behavior | Use Case |
|------|----------|----------|
| `off` | No inspection | Disabled |
| `monitor` | Inspect & log, never block | Testing, observability |
| `enforce` | Inspect & block violations | Production |

Set in `agentsec.yaml`:
```yaml
api_mode:
  llm:
    mode: enforce    # off | monitor | enforce
  mcp:
    mode: monitor
```

### Integration Mode Prerequisites

These prerequisites apply to both **Agent Frameworks** and **Agent Runtimes**.

#### API Mode Prerequisites

| Component | Required | `.env` (secrets) | `agentsec.yaml` (settings) |
|-----------|:--------:|------------------|---------------------------|
| **Cisco AI Defense API** | Yes | `AI_DEFENSE_API_MODE_LLM_API_KEY` | `api_mode.llm.endpoint`, `api_mode.llm.mode` |
| **OpenAI** | If using | `OPENAI_API_KEY` | - |
| **Azure OpenAI** | If using | `AZURE_OPENAI_API_KEY` | - |
| **AWS Bedrock** | If using | AWS credentials (profile or env vars) | - |
| **Vertex AI** | If using | GCP ADC (`gcloud auth application-default login`) | - |
| **Cohere** | If using | `COHERE_API_KEY` | - |
| **Mistral AI** | If using | `MISTRAL_API_KEY` | - |

> Azure OpenAI also requires `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT_NAME`, and `AZURE_OPENAI_API_VERSION` in `.env`.

#### Gateway Mode Prerequisites

Gateway URLs are configured in `agentsec.yaml` under `gateway_mode.llm_gateways`. Only secrets (API keys) go in `.env`.

| Provider | Gateway URL | API Key (`.env`) |
|----------|-------------|------------------|
| **OpenAI** | `agentsec.yaml` | `OPENAI_API_KEY` |
| **Azure OpenAI** | `agentsec.yaml` | `AGENTSEC_AZURE_OPENAI_GATEWAY_API_KEY` |
| **AWS Bedrock** | `agentsec.yaml` | AWS Sig V4 (no API key) |
| **Vertex AI** | `agentsec.yaml` | ADC OAuth2 (no API key) |
| **Cohere** | `agentsec.yaml` | `AGENTSEC_COHERE_GATEWAY_API_KEY` |
| **Mistral AI** | `agentsec.yaml` | `AGENTSEC_MISTRAL_GATEWAY_API_KEY` |

---

## Integration Pattern

The standard pattern for integrating agentsec:

```python
# 1. Import and enable protection BEFORE importing LLM clients
from aidefense.runtime import agentsec

# Option A: YAML config (recommended for production)
agentsec.protect(config="agentsec.yaml")

# Option B: Inline config (for quick testing)
agentsec.protect(api_mode={"llm": {"mode": "enforce"}, "mcp": {"mode": "enforce"}})

# 2. Import and use LLM clients normally - they're now protected
from openai import OpenAI
client = OpenAI()

# 3. All calls are inspected by AI Defense
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Gateway Mode Pattern

```python
from aidefense.runtime import agentsec
agentsec.protect(
    llm_integration_mode="gateway",
    gateway_mode={
        "llm_gateways": {
            "openai-default": {
                "gateway_url": "https://gateway.../openai-conn",
                "gateway_api_key": "your-key",
                "provider": "openai",
                "default": True,
            },
        },
    },
)
```

### Skip Inspection Pattern

```python
from aidefense.runtime.agentsec import skip_inspection, no_inspection

# Context manager - skip specific calls
with skip_inspection():
    response = client.chat.completions.create(...)

# Decorator - skip entire function
@no_inspection()
def my_function():
    response = client.chat.completions.create(...)
```

**Key Rule**: Always call `agentsec.protect()` **before** importing LLM client libraries.

---

## Directory Structure

<details>
<summary>Click to expand project layout</summary>

```
ai-defense-python-sdk/
├── scripts/
│   ├── run-unit-tests.sh           # Run all unit tests
│   └── run-integration-tests.sh    # Run integration tests
│
└── examples/agentsec/
    ├── agentsec.yaml               # Configuration (modes, gateways, timeouts)
    ├── .env.example                # Template - copy to .env (secrets only)
    ├── README.md                   # This file
    │
    ├── 1-simple/                   # Standalone examples
    │   ├── basic_protection.py     # Minimal setup
    │   ├── openai_example.py       # OpenAI client
    │   ├── cohere_example.py       # Cohere v2 client
    │   ├── mistral_example.py      # Mistral AI client
    │   ├── streaming_example.py    # Streaming responses
    │   ├── mcp_example.py          # MCP tool inspection
    │   ├── gateway_mode_example.py # Gateway mode
    │   ├── skip_inspection_example.py  # Per-call exclusion
    │   ├── simple_strands_bedrock.py   # Strands + Bedrock
    │   └── tests/
    │
    ├── 2-agent-frameworks/         # Agent framework examples
    │   ├── run-all-integration-tests.sh  # Run all framework tests
    │   ├── strands-agent/          # AWS Strands SDK
    │   ├── langgraph-agent/        # LangGraph
    │   ├── langchain-agent/        # LangChain
    │   ├── crewai-agent/           # CrewAI
    │   ├── autogen-agent/          # AutoGen
    │   ├── openai-agent/           # OpenAI Agents SDK
    │   └── _shared/                # Shared provider configs & tests
    │
    └── 3-agent-runtimes/           # Cloud runtime examples
        ├── run-all-integration-tests.sh  # Run all runtime tests
        ├── amazon-bedrock-agentcore/
        ├── gcp-vertex-ai-agent-engine/
        └── microsoft-foundry/
```

</details>

---

## 1. Simple Examples

Standalone examples demonstrating core agentsec features without agent frameworks.

> **Prerequisites**: See [Integration Mode Prerequisites](#integration-mode-prerequisites) for API/Gateway configuration.

| Example | Description | Request | Response |
|---------|-------------|:-------:|:--------:|
| `basic_protection.py` | Minimal setup - modes, patched clients | ✅ | ✅ |
| `openai_example.py` | OpenAI client with automatic inspection | ✅ | ✅ |
| `cohere_example.py` | Cohere v2 client with automatic inspection | ✅ | ✅ |
| `mistral_example.py` | Mistral AI client with automatic inspection | ✅ | ✅ |
| `streaming_example.py` | Streaming responses with chunk inspection | ✅ | ✅ |
| `mcp_example.py` | MCP tool call inspection (pre & post) | ✅ | ✅ |
| `gateway_mode_example.py` | Gateway mode configuration | ✅ | ✅ |
| `skip_inspection_example.py` | Per-call exclusion with context manager | ✅ | ✅ |
| `simple_strands_bedrock.py` | Strands agent with Bedrock Claude | ✅ | ✅ |

### Run Examples

```bash
cd 1-simple
poetry install  # First time only

# Run individual examples
poetry run python basic_protection.py
poetry run python openai_example.py
poetry run python cohere_example.py
poetry run python mistral_example.py
poetry run python streaming_example.py
poetry run python mcp_example.py
poetry run python gateway_mode_example.py
poetry run python skip_inspection_example.py
poetry run python simple_strands_bedrock.py

# Run integration tests (from repo root)
cd /path/to/ai-defense-python-sdk
./scripts/run-integration-tests.sh --simple
```

---

## 2. Agent Frameworks

Agent framework examples with MCP tool support and multi-provider configuration.

> **Prerequisites**: See [Integration Mode Prerequisites](#integration-mode-prerequisites) for API/Gateway configuration.

### Supported Frameworks

| Framework | Package | Framework Prerequisites |
|-----------|---------|------------------------|
| **Strands** | `strands-agents` | Poetry |
| **LangGraph** | `langgraph` | Poetry |
| **LangChain** | `langchain` | Poetry |
| **CrewAI** | `crewai` | Poetry |
| **AutoGen** | `ag2` | Poetry |
| **OpenAI Agents** | `openai` | Poetry |

### Provider Support

All frameworks support multiple LLM providers:

| Provider | Flag | Auth Methods | Config File |
|----------|------|--------------|-------------|
| **OpenAI** | `--openai` | API key | `config-openai.yaml` |
| **AWS Bedrock** | `--bedrock` | default, profile, session_token, iam_role | `config-bedrock.yaml` |
| **Azure OpenAI** | `--azure` | api_key, managed_identity, cli | `config-azure.yaml` |
| **GCP Vertex AI** | `--vertex` | adc, service_account | `config-vertex.yaml` |

### Run Examples

```bash
cd 2-agent-frameworks

# Run with specific provider
./strands-agent/scripts/run.sh --openai
./strands-agent/scripts/run.sh --bedrock
./langgraph-agent/scripts/run.sh --azure
./crewai-agent/scripts/run.sh --vertex

# Run integration tests (from repo root)
cd /path/to/ai-defense-python-sdk
./scripts/run-integration-tests.sh --agents                # All frameworks, all providers
./scripts/run-integration-tests.sh strands                 # Specific framework
./scripts/run-integration-tests.sh --agents --api          # All frameworks, API mode only
./scripts/run-integration-tests.sh --agents --gateway      # All frameworks, Gateway mode only
./scripts/run-integration-tests.sh strands --gateway       # Specific framework + mode
```

You can also run all framework tests directly from the `2-agent-frameworks/` directory:

```bash
cd examples/agentsec/2-agent-frameworks
./run-all-integration-tests.sh                # All frameworks, all providers, both modes
./run-all-integration-tests.sh --quick        # All frameworks, OpenAI only, both modes
./run-all-integration-tests.sh --api          # API mode only
./run-all-integration-tests.sh langgraph      # Specific framework
```

### Protection Coverage

All agent frameworks support both request and response inspection:

- **LLM Calls**: Every `chat.completions.create()` / `converse()` / `generate_content()` is inspected
- **MCP Tool Calls**: Tool request and response payloads are inspected (when MCP is used)
- **Streaming**: Response chunks are buffered and inspected at completion

---

## 3. Agent Runtimes

Cloud deployment examples with full AI Defense protection.

> **AI Defense Prerequisites**: In addition to the deploy prerequisites below, see [Integration Mode Prerequisites](#integration-mode-prerequisites) for API/Gateway configuration.

### Test Modes

All agent runtimes support two test modes:

| Mode | Flag | What It Tests | Cloud Required |
|------|------|---------------|----------------|
| **Local** | `--local` | Agent code directly with LLM provider | LLM credentials only |
| **Deploy** | `--deploy` | Full cloud deployment and endpoints | Cloud infrastructure |

### AWS Bedrock AgentCore

Three deployment modes for AWS AgentCore agents:

| Mode | Description | Local Prerequisites | Deploy Prerequisites |
|------|-------------|---------------------|---------------------|
| **Direct Deploy** | Agent with Bedrock | AWS Bedrock credentials | AWS CLI, AgentCore CLI |
| **Container Deploy** | Docker container | AWS Bedrock credentials | Docker, ECR permissions |
| **Lambda Deploy** | AWS Lambda function | AWS Bedrock credentials | Lambda/ECR permissions |

```bash
cd 3-agent-runtimes/amazon-bedrock-agentcore
poetry install

# Run LOCAL tests (default) - no AWS deployment needed
./tests/integration/test-all-modes.sh              # All modes, local
./tests/integration/test-all-modes.sh --local      # Explicit local mode
./tests/integration/test-all-modes.sh --api        # API mode only, local

# Run DEPLOY tests - deploys to AWS and tests real endpoints
./tests/integration/test-all-modes.sh --deploy     # All modes, deploy to AWS
./tests/integration/test-all-modes.sh --deploy direct --api  # Direct deploy, API only
```

### GCP Vertex AI Agent Engine

Three deployment modes for GCP Vertex AI agents:

| Mode | Description | Local Prerequisites | Deploy Prerequisites |
|------|-------------|---------------------|---------------------|
| **Agent Engine** | Google's managed agent service | GCP ADC credentials | gcloud CLI, Vertex AI permissions |
| **Cloud Run** | Serverless containers | GCP ADC credentials | Docker, Cloud Run permissions |
| **GKE** | Kubernetes deployment | GCP ADC credentials | Docker, kubectl, GKE permissions |

**Supported Google AI SDKs:**

| SDK | Package | Status | Environment Variable |
|-----|---------|--------|----------------------|
| **vertexai** | `google-cloud-aiplatform` | Legacy (default) | `GOOGLE_AI_SDK=vertexai` |
| **google-genai** | `google-genai` | Modern (recommended) | `GOOGLE_AI_SDK=google_genai` |

```bash
cd 3-agent-runtimes/gcp-vertex-ai-agent-engine
poetry install

# Choose SDK (optional, defaults to vertexai)
export GOOGLE_AI_SDK=google_genai  # Use modern SDK

# Run LOCAL tests (default) - no GCP deployment needed
./tests/integration/test-all-modes.sh              # All modes, local
./tests/integration/test-all-modes.sh --local      # Explicit local mode
./tests/integration/test-all-modes.sh --api        # API mode only, local

# Run DEPLOY tests - deploys to GCP and tests real endpoints
./tests/integration/test-all-modes.sh --deploy     # All modes, deploy to GCP
./tests/integration/test-all-modes.sh --deploy --mode cloud-run  # Cloud Run only
./tests/integration/test-all-modes.sh --deploy --cleanup  # Deploy, test, and cleanup
```

### Microsoft Azure AI Foundry

Three deployment modes for Microsoft Azure AI Foundry agents:

| Mode | Description | Local Prerequisites | Deploy Prerequisites |
|------|-------------|---------------------|---------------------|
| **Foundry Agent App** | Azure AI Foundry managed endpoint | Azure OpenAI credentials | Azure CLI, ML extension |
| **Azure Functions** | Serverless functions | Azure OpenAI credentials | Functions Core Tools |
| **Foundry Container** | Container deployment | Azure OpenAI credentials | Docker, ACR permissions |

```bash
cd 3-agent-runtimes/microsoft-foundry
poetry install

# Run DEPLOY tests (default for this script) - deploys to Azure and tests real endpoints
./tests/integration/test-all-modes.sh              # All modes, deploy to Azure
./tests/integration/test-all-modes.sh agent-app --api  # Agent app, API only

# Run LOCAL tests - no Azure deployment needed
./tests/integration/test-all-modes.sh --local      # All modes, local
./tests/integration/test-all-modes.sh --local --api # API mode only, local
```

> **Note**: Unlike Bedrock and GCP (which default to local), Foundry's test script defaults to deploy mode since it tests against already-deployed Azure endpoints.

### Runtime Integration Tests

There are two ways to run runtime integration tests:

**Option 1: Top-level script** (recommended — uses each runtime's own default mode)

```bash
# From the repo root
cd /path/to/ai-defense-python-sdk

./scripts/run-integration-tests.sh --runtimes            # All runtimes
./scripts/run-integration-tests.sh amazon-bedrock-agentcore  # Specific runtime
./scripts/run-integration-tests.sh --runtimes --api      # API mode only
./scripts/run-integration-tests.sh --runtimes --deploy   # Force deploy mode for all
```

**Option 2: Runtimes directory script** (interactive — prompts before running)

```bash
cd examples/agentsec/3-agent-runtimes

# Default: --deploy for AgentCore/Vertex AI, --local for Foundry
./run-all-integration-tests.sh                     # Default test mode

# Override modes
./run-all-integration-tests.sh --local             # All runtimes, local mode
./run-all-integration-tests.sh --deploy            # All runtimes, deploy mode

# Run specific runtime
./run-all-integration-tests.sh amazon-bedrock-agentcore        # AgentCore only
./run-all-integration-tests.sh --local microsoft-foundry       # Foundry, local mode

# Quick mode (API mode only, 1 test per runtime)
./run-all-integration-tests.sh --quick             # Quick tests for all runtimes
```

> **Note**: Default test modes differ by script. The top-level `scripts/run-integration-tests.sh` uses each runtime's own default (local for Bedrock/Vertex AI, deploy for Foundry). The `run-all-integration-tests.sh` in `3-agent-runtimes/` inverts this (deploy for Bedrock/Vertex AI, local for Foundry).

---

## Configuration

agentsec uses two configuration files:

| File | Purpose | Contents |
|------|---------|----------|
| **`agentsec.yaml`** | All non-secret settings | Integration modes, gateway URLs, endpoints, timeouts, retry policy, inspection modes |
| **`.env`** | Secrets only | API keys, credentials (referenced via `${VAR}` in YAML) |

### Getting Started

```bash
# Copy the template files
cp .env.example .env
# Edit .env — fill in your API keys and credentials
# Edit agentsec.yaml — configure integration modes, gateway URLs, inspection modes
```

### Secrets in `.env`

#### AI Defense API Keys (Required for API integration)

| Variable | Required | Description |
|----------|:--------:|-------------|
| `AI_DEFENSE_API_MODE_LLM_API_KEY` | Yes | AI Defense API key for LLM inspection |
| `AI_DEFENSE_API_MODE_MCP_API_KEY` | For MCP | AI Defense API key for MCP inspection |

#### Gateway API Keys (Required for Gateway integration)

| Variable | When Required | Description |
|----------|:-------------:|-------------|
| `OPENAI_API_KEY` | Gateway + OpenAI | OpenAI API key (used for both direct and gateway mode) |
| `AGENTSEC_AZURE_OPENAI_GATEWAY_API_KEY` | Gateway + Azure | Azure OpenAI gateway API key |
| `AGENTSEC_COHERE_GATEWAY_API_KEY` | Gateway + Cohere | Cohere gateway API key |
| `AGENTSEC_MISTRAL_GATEWAY_API_KEY` | Gateway + Mistral | Mistral AI gateway API key |

> All gateway URLs are configured in `agentsec.yaml` — only API keys go in `.env`.
> Bedrock uses AWS Sig V4 auth and Vertex AI uses ADC OAuth2 — neither requires a gateway API key.

### Settings in `agentsec.yaml`

All non-secret configuration lives in `agentsec.yaml`. Key sections:

| Section | What It Configures | Example |
|---------|-------------------|---------|
| `llm_integration_mode` | API or Gateway for LLM calls | `gateway` |
| `mcp_integration_mode` | API or Gateway for MCP calls | `api` |
| `api_mode.llm` | API inspection endpoint, mode, API key ref | `mode: enforce` |
| `api_mode.mcp` | MCP inspection endpoint, mode, API key ref | `mode: monitor` |
| `gateway_mode.llm_gateways` | Named gateway entries with URLs and providers | See `agentsec.yaml` |
| `gateway_mode.mcp_gateways` | MCP server → gateway URL mappings | See `agentsec.yaml` |
| `*.defaults` | Timeout, retry, fail_open per mode | `timeout: 5` |
| `logging` | Log level and format | `level: DEBUG` |

> See `agentsec.yaml` for the full configuration reference with inline documentation.

### Provider Credentials

<details>
<summary><strong>OpenAI</strong></summary>

```bash
OPENAI_API_KEY=sk-your-openai-api-key
```

</details>

<details>
<summary><strong>Azure OpenAI</strong></summary>

```bash
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-azure-openai-key
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name  # e.g., gpt-4o
AZURE_OPENAI_API_VERSION=2024-08-01-preview
```

</details>

<details>
<summary><strong>AWS Bedrock</strong></summary>

```bash
AWS_AUTH_METHOD=profile    # profile | session_token
AWS_REGION=us-east-1

# For profile auth (recommended)
AWS_PROFILE=default

# For session_token auth (uncomment if needed)
# AWS_ACCESS_KEY_ID=your-access-key
# AWS_SECRET_ACCESS_KEY=your-secret-key
# AWS_SESSION_TOKEN=your-session-token
```

You can also use SSO: `aws sso login`

</details>

<details>
<summary><strong>GCP Vertex AI / Google GenAI</strong></summary>

```bash
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-central1

# Authenticate via gcloud CLI (ADC)
gcloud auth application-default login

# Choose SDK (optional)
GOOGLE_AI_SDK=google_genai  # Modern SDK (recommended)
# or
GOOGLE_AI_SDK=vertexai      # Legacy SDK (default)
```

</details>

<details>
<summary><strong>Cohere</strong></summary>

```bash
COHERE_API_KEY=your-cohere-api-key
```

**How to get a Cohere API key**

1. Sign up at [Cohere Dashboard](https://dashboard.cohere.com/welcome/register) (email/password or Google/GitHub).
2. Log in and go to [API keys](https://dashboard.cohere.com/api-keys).
3. Create a key (e.g. **Trial** for limited free usage or **Production** for higher limits).
4. Copy the key into `.env` as `COHERE_API_KEY`.

</details>

<details>
<summary><strong>Mistral AI</strong></summary>

```bash
MISTRAL_API_KEY=your-mistral-api-key
```

**How to get a Mistral API key**

1. Create an account at [Mistral Console](https://console.mistral.ai).
2. (Optional) Set up billing at [admin.mistral.ai](https://admin.mistral.ai) (Organization → Billing). You can use the **Experiment** (free) or **Scale** (pay-as-you-go) plan.
3. In your workspace, open **API Keys** and click **Create new key**.
4. Copy the key into `.env` as `MISTRAL_API_KEY` (it is shown only once).

</details>

### Programmatic Configuration

Configuration can be provided via `protect()` parameters using structured dicts:

```python
from aidefense.runtime import agentsec

# Option 1: YAML config file (recommended)
agentsec.protect(config="agentsec.yaml")

# Option 2: Inline configuration
agentsec.protect(
    api_mode={
        "llm": {
            "mode": "enforce",
            "endpoint": "https://preview.api.inspect.aidefense.aiteam.cisco.com/api",
            "api_key": "your-api-key",
        },
        "mcp": {"mode": "monitor"},
        "llm_defaults": {"fail_open": True, "timeout": 5},
        "mcp_defaults": {"fail_open": True},
    },
    gateway_mode={
        "llm_gateways": {
            "openai-default": {
                "gateway_url": "https://gateway.../openai-conn",
                "gateway_api_key": "your-key",
                "provider": "openai",
                "default": True,
            },
        },
        "mcp_gateways": {
            "https://remote.mcpservers.org/fetch/mcp": {"gateway_url": "https://gateway.../mcp", "gateway_api_key": "your-key"},
        },
    },
    # Connection pool (global)
    pool_max_connections=20,
    pool_max_keepalive=10,
    # Custom logger
    custom_logger=my_logger,
    log_file="/var/log/agentsec.log",
)
```

See `agentsec.yaml` for a complete YAML configuration reference.

### YAML Config (Agent Frameworks)

Each agent framework has config files in `config/`:

```yaml
# config/config-bedrock.yaml
provider: bedrock
bedrock:
  model_id: anthropic.claude-3-haiku-20240307-v1:0
  region: us-east-1
  auth:
    method: default  # or: profile, session_token, iam_role
```

Select config via flag:
```bash
./scripts/run.sh --bedrock   # Uses config/config-bedrock.yaml
./scripts/run.sh --azure     # Uses config/config-azure.yaml
./scripts/run.sh --vertex    # Uses config/config-vertex.yaml
./scripts/run.sh --openai    # Uses config/config-openai.yaml
```

---

## Advanced Usage

### Error Handling

agentsec provides typed exceptions for granular error handling:

```python
from aidefense.runtime.agentsec import (
    AgentsecError,           # Base exception
    SecurityPolicyError,     # Policy violation (blocked content)
    InspectionTimeoutError,  # API timeout
    InspectionNetworkError,  # Network/connection failure
    ConfigurationError,      # Invalid configuration
    ValidationError,         # Input validation failure
)

try:
    response = client.chat.completions.create(...)
except SecurityPolicyError as e:
    print(f"Blocked: {e.decision.reasons}")
except InspectionTimeoutError as e:
    print(f"Timeout after {e.timeout_ms}ms")
except InspectionNetworkError:
    print("Network error - check connectivity")
```

### Request Context

Add metadata to inspection requests for better tracking:

```python
from aidefense.runtime.agentsec import set_metadata

# Set context before LLM calls
set_metadata(
    user="user@example.com",
    src_app="my-chatbot",
    client_transaction_id="txn-12345",
    custom_field="any-value",
)
```

### Inspection Results

The `Decision` object includes detailed inspection results:

```python
decision = inspector.inspect_conversation(messages, metadata)

decision.action          # "allow", "block", "sanitize", "monitor_only"
decision.is_safe         # True if action != "block"
decision.reasons         # List of reasons
decision.severity        # "low", "medium", "high", "critical"
decision.classifications # ["pii", "prompt_injection", ...]
decision.event_id        # Unique event identifier for tracking
decision.explanation     # Human-readable explanation
```

---

## Testing

### Test Summary

| Category | Test Type | Test Count | What It Validates |
|----------|-----------|:----------:|-------------------|
| **Core SDK** | Unit | ~670 | Patching, inspection, decisions, config |
| **Simple Examples** | Unit | ~85 | Example file structure, syntax |
| **Simple Examples** | Integration | 18 | 9 examples x 2 modes (API + Gateway) |
| **Agent Frameworks** | Unit | ~210 | Agent setup, provider configs |
| **Agent Frameworks** | Integration | ~40 | 6 frameworks x (2-4 providers) x 2 modes* |
| **AgentCore** | Unit | ~65 | Deploy scripts, protection setup |
| **AgentCore** | Integration | 8 | (3 deploy x 2 modes) + 2 MCP tests |
| **Vertex AI** | Unit | ~50 | Deploy scripts, SDK selection |
| **Vertex AI** | Integration | 16 | (3 deploy x 2 modes) + 2 MCP tests** |
| **Azure AI Foundry** | Unit | ~50 | Deploy scripts, agent factory, endpoints |
| **Azure AI Foundry** | Integration | 8 | (3 deploy x 2 modes) + 2 MCP tests |

**Total: ~1130 unit tests**

*Provider support varies by framework: OpenAI Agents supports 2 providers (openai, azure), others support 4 (openai, azure, vertex, bedrock)

**Vertex AI counts individual check assertions as passed tests

### Run Tests

```bash
# From project root
cd /path/to/ai-defense-python-sdk

# All unit tests (~1130 tests)
./scripts/run-unit-tests.sh

# All integration tests
./scripts/run-integration-tests.sh

# Specific test categories
./scripts/run-integration-tests.sh --simple      # Simple examples only
./scripts/run-integration-tests.sh --agents      # Agent frameworks only
./scripts/run-integration-tests.sh --runtimes    # Agent runtimes only

# Specific mode
./scripts/run-integration-tests.sh --api         # API mode only
./scripts/run-integration-tests.sh --gateway     # Gateway mode only

# Specific framework/runtime
./scripts/run-integration-tests.sh strands       # Strands agent only
./scripts/run-integration-tests.sh amazon-bedrock-agentcore  # AgentCore only
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'aidefense'` | Run `poetry install` in the example directory |
| `ModuleNotFoundError: No module named 'dotenv'` | Run `poetry add python-dotenv` or `pip install python-dotenv` |
| AWS auth fails | Run `aws sts get-caller-identity` to verify credentials |
| Azure 401 Unauthorized | Check endpoint format: `https://your-resource.openai.azure.com` |
| GCP auth fails | Run `gcloud auth application-default login` |
| OpenAI 401 Unauthorized | Verify key at https://platform.openai.com/api-keys |
| `SecurityPolicyError` raised | Expected in enforce mode when content violates policies |
| `[BLOCKED] Prompt Injection` | AI Defense detected prompt injection - this is working correctly |
| No inspection happening | Ensure `agentsec.protect()` is called BEFORE importing LLM clients |
| MCP tool calls not inspected | Ensure `mcp` package is installed and `api_mode.mcp.mode` is set in `agentsec.yaml` |
| Poetry version error | Remove `package-mode = false` from pyproject.toml if using older Poetry |

### Debug Logging

By default, agentsec operates quietly (log level `WARNING`). To see what's happening under the hood - including messages sent to AI Defense, responses received, and inspection decisions - enable debug logging.

#### Logging Environment Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `AGENTSEC_LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `WARNING` | Controls verbosity |
| `AGENTSEC_LOG_FORMAT` | `text`, `json` | `text` | Output format (use `json` for log aggregators) |
| `AGENTSEC_LOG_FILE` | file path | None | Optional file to write logs to |

#### How to Enable Debug Logging

**Option 1: Environment variable (recommended)**
```bash
export AGENTSEC_LOG_LEVEL=DEBUG
python your_agent.py
```

**Option 2: In your code (before `agentsec.protect()`)**
```python
import os
os.environ["AGENTSEC_LOG_LEVEL"] = "DEBUG"

from aidefense.runtime import agentsec
agentsec.protect()
```

**Option 3: Command line (one-off)**
```bash
AGENTSEC_LOG_LEVEL=DEBUG python your_agent.py
```

#### What You'll See with DEBUG Logging

**1. Patched LLM Call Flow**
```
╔══════════════════════════════════════════════════════════════
║ [PATCHED] LLM CALL: gpt-4o-mini
║ Operation: OpenAI.chat.completions.create | LLM Mode: enforce | Integration: api | Provider: openai
╚══════════════════════════════════════════════════════════════
[aidefense.runtime.agentsec] DEBUG: [PATCHED CALL] OpenAI.chat.completions.create - Request inspection (3 messages)
[aidefense.runtime.agentsec] DEBUG: [PATCHED CALL] OpenAI.chat.completions.create - Request decision: allow
```

**2. Request Payload Sent to AI Defense**
```
[aidefense.runtime.agentsec] DEBUG: AI Defense request: 3 messages, metadata=['user', 'src_app', 'transaction_id']
[aidefense.runtime.agentsec] DEBUG: AI Defense request payload: {
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "metadata": {"user": "test-user", "src_app": "my-agent"}
}
```

**3. AI Defense Response**
```
[aidefense.runtime.agentsec] DEBUG: AI Defense response: {
  "action": "allow",
  "reasons": [],
  "sanitized_content": null
}
```

**4. Response Inspection**
```
[aidefense.runtime.agentsec] DEBUG: [PATCHED CALL] OpenAI.chat.completions.create - Response inspection (response: 142 chars)
[aidefense.runtime.agentsec] DEBUG: [PATCHED CALL] OpenAI.chat.completions.create - Response decision: allow
[aidefense.runtime.agentsec] DEBUG: [PATCHED CALL] OpenAI.chat.completions.create - complete
```

**5. Gateway Mode (if using)**
```
[aidefense.runtime.agentsec] DEBUG: [GATEWAY] Sending request to openai gateway: https://gateway.example.com/v1/chat/completions
[aidefense.runtime.agentsec] DEBUG: [GATEWAY] Received response from openai gateway
```

**6. MCP Tool Call Inspection**
```
[aidefense.runtime.agentsec] DEBUG: [PATCHED CALL] MCP.call_tool(fetch_url) - Request inspection
[aidefense.runtime.agentsec] DEBUG: [PATCHED CALL] MCP.call_tool(fetch_url) - Request decision: allow
[aidefense.runtime.agentsec] DEBUG: [PATCHED CALL] MCP.call_tool(fetch_url) - Response inspection
[aidefense.runtime.agentsec] DEBUG: [PATCHED CALL] MCP.call_tool(fetch_url) - Response decision: allow
```

**7. Block Decision (when content violates policy)**
```
[aidefense.runtime.agentsec] DEBUG: AI Defense BLOCK response: {
  "action": "block",
  "reasons": ["prompt_injection_detected"],
  "sanitized_content": null
}
[aidefense.runtime.agentsec] WARNING: [BLOCKED] Content blocked by AI Defense: prompt_injection_detected
```

#### JSON Log Format

For production environments or log aggregation systems, use JSON format:

```bash
export AGENTSEC_LOG_LEVEL=DEBUG
export AGENTSEC_LOG_FORMAT=json
```

Output:
```json
{"timestamp": "2025-01-24T10:30:45.123Z", "level": "DEBUG", "logger": "aidefense.runtime.agentsec", "message": "AI Defense response: {'action': 'allow'}"}
```

### Getting Help

- Check the individual example READMEs for detailed setup instructions
- Review the main [README.md](../../README.md) for SDK configuration
- Enable DEBUG logging to trace inspection flow
- Run `./scripts/run-unit-tests.sh` to verify SDK installation
