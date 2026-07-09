# TechSpec: Hybrid Cloud Platform Models for AutoManager

## Executive Summary

AutoManager will support Codex, Claude Code, and Google Antigravity by supervising CLIProxyAPI as a shared local HTTP sidecar. Platform integrations will appear beside local models, can be started from the UI, and will be exposed through AutoManager's existing `/v1` API and smart proxy flow.

The primary trade-off is adding one managed sidecar process and startup-only detection in exchange for reusing CLIProxyAPI's provider authentication, translation, model registry, and subscription-tool behavior without duplicating that logic in Python.

## System Architecture

### Component Overview

- `PlatformIntegrationManager`: owns startup-time detection, platform card state, and active platform backend state.
- `CLIProxySidecarManager`: starts, stops, health-checks, and tracks the shared CLIProxyAPI HTTP sidecar.
- `HybridBackendRegistry`: resolves local `model_path` backends and platform `backend_id` backends behind one proxy-facing contract.
- `ConfigManager`: persists platform preferences under `platform_configs` and keeps local `model_configs` unchanged.
- `llama_manager.py`: extends `/models`, `/status`, `/v1/*`, and proxy config routes to include platform backends.
- `proxy_router.py`: routes smart proxy decisions to local llama-server ports or the shared sidecar port.
- Frontend model cards: render platform cards in the same catalog area as local models, with platform state labels and proxy eligibility controls.

Data flow: startup detection creates platform catalog entries; the operator starts a platform card; AutoManager starts the shared sidecar if needed; `/v1/models` queries local instances and the sidecar; request forwarding routes local model IDs to llama-server and platform model IDs to the sidecar.

## Implementation Design

### Core Interfaces

The wire contract should be backend-aware. Go struct shown for explicit field types:

```go
type HybridBackend struct {
    BackendID     string `json:"backend_id"`
    BackendType   string `json:"backend_type"`
    Provider      string `json:"provider,omitempty"`
    ModelPath     string `json:"model_path,omitempty"`
    Model         string `json:"model"`
    Port          int    `json:"port"`
    Status        string `json:"status"`
    ProxyEligible bool   `json:"proxy_eligible"`
}
```

Python implementation should use lightweight dataclasses or Pydantic schemas with the same fields. Local backends keep `model_path`; platform backends use `backend_id`.

### Data Models

- `PlatformDefinition`: static MVP definition for `platform:codex`, `platform:claude-code`, and `platform:google-antigravity`.
- `PlatformDetection`: `backend_id`, `provider`, `display_name`, `detected`, `executable_path`, `status`, `reason`.
- `PlatformRuntimeState`: `backend_id`, `provider`, `active`, `sidecar_port`, `status`, `last_error`.
- `platform_configs`: persisted map keyed by `backend_id`, containing `proxy_eligible`, `max_parallel_requests`, and optional display preferences.
- `smart_proxy.primary_backend_id`: new backend-aware primary selector. Existing `primary_model_path` remains for local-model compatibility.

### API Endpoints

- `GET /models`: returns existing `models`, `projectors`, and `storage`, plus a new `platforms` array.
- `GET /status`: returns existing process status plus platform runtime state and merged hybrid instances.
- `POST /platforms/{backend_id}/start`: starts the selected platform integration and shared sidecar if needed.
- `POST /platforms/{backend_id}/stop`: deactivates the selected platform integration; stops the sidecar when no platform remains active.
- `POST /models/proxy`: accepts either `model_path` for local models or `backend_id` for platform integrations.
- `POST /proxy/config`: accepts `primary_backend_id` while preserving `primary_model_path`.
- `GET /v1/models`: aggregates local llama-server models and sidecar `/v1/models`, preserving sidecar model IDs.
- `/v1/{path}`: forwards local model requests to llama-server ports and platform model requests to the sidecar port.

## Integration Points

CLIProxyAPI is the only external runtime integration in the MVP. AutoManager will detect its executable from `PATH` and known local install locations, then start it bound to `127.0.0.1` on an AutoManager-managed port.

AutoManager will not collect provider API keys, OAuth tokens, or provider logins. CLIProxyAPI uses existing local authentication state. Remote management endpoints should remain disabled unless a later phase explicitly needs them.

Failures should surface as concise UI reasons: missing executable, sidecar failed to start, sidecar unhealthy, provider not ready, or `/v1/models` unavailable.

## Impact Analysis

| Component | Impact Type | Description and Risk | Required Action |
|-----------|-------------|---------------------|-----------------|
| `llama_manager.py` | Modified | Adds platform routes and hybrid `/v1` aggregation. Medium risk. | Extend route handlers behind backend-aware helpers. |
| `proxy_router.py` | Modified | Supports platform backends using sidecar port and `backend_id`. Medium risk. | Add backend identity handling without breaking local routing. |
| `config_manager.py` | Modified | Persists `platform_configs` and `primary_backend_id`. Medium risk. | Add migration-safe defaults. |
| `model_manager.py` | Modified | Catalog response includes platform entries. Low risk. | Merge startup detection output into `/models`. |
| `static/js/models.js` | Modified | Renders platform cards and start/stop states. Medium risk. | Add card variant without changing local card behavior. |
| `static/js/proxy.js` | Modified | Shows local versus platform proxy eligibility. Low risk. | Use `backend_id` when present. |
| CLIProxyAPI sidecar | New | Adds supervised process dependency. Medium risk. | Add start, stop, health, and generated config handling. |

## Testing Approach

### Unit Tests

- Platform detector: detected, missing, and not-ready cases.
- Config migration: existing local configs remain unchanged; `platform_configs` defaults are added.
- Backend identity resolver: local `model_path` and platform `backend_id` both resolve correctly.
- Sidecar manager: command construction, port selection, generated config path, health failure handling.
- Proxy eligibility: platform backends are excluded unless explicitly enabled.

### Integration Tests

- Fake sidecar HTTP server exposing `/v1/models` and `/v1/chat/completions`.
- `/v1/models` aggregation with local-only, platform-only, and mixed backends.
- Smart proxy routing to a sidecar-backed platform backend.
- UI contract tests for `/models`, `/status`, and proxy config payloads.

## Development Sequencing

### Build Order

1. Add hybrid backend schemas and config defaults - no dependencies.
2. Add startup-only platform detector - depends on step 1.
3. Add CLIProxyAPI sidecar manager - depends on steps 1 and 2.
4. Extend `/models` and `/status` responses - depends on steps 1, 2, and 3.
5. Extend `/v1/models` aggregation and request forwarding - depends on steps 3 and 4.
6. Extend smart proxy config and router backend selection - depends on steps 1, 3, and 5.
7. Update model catalog and proxy UI controls - depends on steps 4 and 6.
8. Add unit, integration, and UI contract tests - depends on steps 2 through 7.

### Technical Dependencies

- CLIProxyAPI executable discovery or bundling strategy.
- Known command names and install paths for Codex, Claude Code, and Google Antigravity.
- Localhost port allocation that does not collide with llama-server instances.
- Windows and Linux path behavior for executable detection.

## Monitoring and Observability

Log structured events for platform detection, sidecar start, sidecar stop, sidecar health failure, platform activation, model aggregation failure, and smart proxy routing to platform backends.

Track counts for detected platforms, active platforms, sidecar start failures, sidecar health failures, `/v1/models` sidecar failures, and platform-routed requests.

## Technical Considerations

### Key Decisions

- Decision: Use CLIProxyAPI as a shared HTTP sidecar. Rationale: reuses provider support and fits AutoManager's port-based proxy model.
- Decision: Use `backend_id` and `backend_type`. Rationale: platform integrations are not filesystem paths.
- Decision: Preserve sidecar model IDs. Rationale: API clients see real model names from CLIProxyAPI.
- Decision: Detect platforms only at AutoManager startup. Rationale: predictable MVP behavior selected during technical review.

### Known Risks

- Sidecar availability can fail independently from AutoManager. Mitigation: health-check and display not-ready reasons.
- Sidecar model IDs can collide with local model IDs. Mitigation: deterministic aggregation and future aliasing if needed.
- Startup-only detection may surprise operators after installing a tool. Mitigation: document restart requirement for MVP.
- Platform activation and shared sidecar state can diverge. Mitigation: keep one sidecar process state and separate per-platform runtime state.

## Architecture Decision Records

- [ADR-001: Unified Catalog MVP for Hybrid Platform Models](adrs/adr-001.md) - Treat detected cloud tools as first-class catalog cards.
- [ADR-002: CLIProxyAPI HTTP Sidecar for Platform Backends](adrs/adr-002.md) - Use CLIProxyAPI as the managed local HTTP sidecar.
- [ADR-003: Stable Backend Identity for Platform Integrations](adrs/adr-003.md) - Use `backend_id` and `backend_type` instead of pseudo-paths.
- [ADR-004: Shared Sidecar Lifecycle and Real Model Discovery](adrs/adr-004.md) - Run one sidecar and preserve model IDs from `/v1/models`.
- [ADR-005: Startup-Only Platform Detection with Persisted Preferences](adrs/adr-005.md) - Detect once at startup and persist only operator preferences.
