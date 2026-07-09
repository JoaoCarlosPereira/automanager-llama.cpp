# PRD: Hybrid Cloud Platform Models for AutoManager

## Overview

AutoManager currently helps a local operator discover, configure, start, and expose local llama.cpp models through a unified panel and local OpenAI-compatible API. This feature extends that model-management experience to cloud subscription tools already installed on the server: Codex, Claude Code, and Google Antigravity.

The problem is that subscription-based AI tools can be available on the same machine, but they are not visible or usable through AutoManager's existing model catalog, API surface, or smart proxy workflow. Operators must manage local models and cloud tools separately, even when they want one operational gateway for the machine.

The feature is valuable because it turns AutoManager into a hybrid operator console. The local operator can see local models and detected platform integrations in one catalog, start a detected cloud integration from the same UI, and make it available through AutoManager's `/v1` model surface without entering separate API keys or configuring provider credentials inside AutoManager.

## Goals

- Show Codex, Claude Code, and Google Antigravity as first-class catalog cards when their applications are detected on the server.
- Preserve the current local-model workflow without requiring existing local users to change behavior.
- Let the local operator start a detected cloud integration from the AutoManager UI.
- Make a started cloud integration appear as an available model through AutoManager's `/v1/models` surface.
- Show unavailable cloud integrations as disabled cards with a clear reason instead of hiding them.
- Let the operator explicitly decide whether a cloud integration is eligible for smart proxy routing.
- Require no AutoManager-specific login, API key entry, or manual provider setup for Codex, Claude Code, or Google Antigravity.

## User Stories

### Local Operator

- As a local operator, I want AutoManager to detect installed cloud coding tools so that I can see all available inference options in one panel.
- As a local operator, I want detected cloud integrations to appear as cards alongside local models so that I do not need a separate mental model for cloud providers.
- As a local operator, I want unavailable integrations to explain why they cannot be started so that I can diagnose missing installation or runtime readiness quickly.
- As a local operator, I want to start a cloud integration from the same place where I start local models so that the operational flow stays consistent.
- As a local operator, I want a started cloud integration to appear in `/v1/models` so that existing API clients can discover and use it.
- As a local operator, I want cloud integrations to join the smart proxy only when I mark them eligible so that subscription usage is deliberate.

### API Client User

- As an API client user, I want AutoManager's `/v1/models` response to include cloud integrations that the operator has started so that my client can select them like other models.
- As an API client user, I want existing local-model API behavior to remain stable so that the hybrid feature does not break current workflows.

## Core Features

### 1. Hybrid Model Catalog

AutoManager must add platform integrations to the same catalog experience used for local models. Codex, Claude Code, and Google Antigravity are required for the MVP.

Functional requirements:
- Display local models and detected cloud integrations in the same model/card area.
- Use visual labels that distinguish local models from platform integrations without creating a separate workflow.
- Show provider identity, readiness state, and a concise reason when unavailable.
- Keep local models visible and behaviorally unchanged.

### 2. Automatic Platform Detection

AutoManager must detect whether Codex, Claude Code, and Google Antigravity are installed or otherwise available on the server.

Functional requirements:
- A detected application is enough for the integration card to appear.
- Credential validation must not be required before the card exists.
- If the integration cannot be started or used, the card must become disabled or not-ready with a simple reason.
- The operator must not need to enter provider credentials inside AutoManager.

### 3. Platform Start Flow

The operator must be able to start a detected cloud integration from its card.

Functional requirements:
- The start action should feel consistent with local model start actions.
- Starting a platform integration should create an operational backend visible to AutoManager.
- If start fails, AutoManager must show a concise provider-specific failure reason where available.
- The UI must avoid implying that AutoManager owns the external subscription account.

### 4. API Model Availability

A successfully started cloud integration must be discoverable through AutoManager's existing model API surface.

Functional requirements:
- Started platform integrations must appear in `/v1/models`.
- Model names must be clear enough for API clients to select intentionally.
- Existing local models must continue to appear as they do today.
- When a platform integration stops or becomes unavailable, its API availability must reflect that state.

### 5. Smart Proxy Participation

Cloud integrations must be eligible for the existing smart proxy only when the operator explicitly enables them.

Functional requirements:
- The operator can mark a cloud integration as proxy-eligible.
- The operator can choose the primary model according to existing smart proxy behavior.
- Cloud integrations must not silently become fallback capacity.
- The proxy view must make local versus cloud backend participation clear.

### 6. No Manual Provider Configuration

The MVP must not require AutoManager-specific setup for supported cloud integrations.

Functional requirements:
- No provider API key entry inside AutoManager.
- No AutoManager login flow for Codex, Claude Code, or Google Antigravity.
- No required manual mapping file for the MVP.
- AutoManager should rely on already-installed tools and their existing local authentication state.

## User Experience

### Primary Flow: First Discovery

1. The operator opens AutoManager.
2. The model catalog shows existing local models.
3. AutoManager also shows cards for detected Codex, Claude Code, and Google Antigravity installations.
4. Cards that are detected but not operational appear disabled or not-ready with a short reason.
5. The operator can immediately understand which options are local and which are platform integrations.

### Primary Flow: Start Cloud Integration

1. The operator selects a detected platform card.
2. The operator starts the integration from the card.
3. AutoManager shows a starting state.
4. When ready, the card indicates that the integration is available.
5. The integration appears in `/v1/models`.
6. Existing API clients can discover the integration as an available model.

### Primary Flow: Proxy Eligibility

1. The operator reviews local and cloud model cards.
2. The operator marks a cloud integration as proxy-eligible only if subscription usage is acceptable.
3. The smart proxy panel shows the cloud integration as an eligible backend.
4. The operator can use existing primary-model selection behavior.
5. AutoManager routes according to explicit operator configuration.

### UX Requirements

- Use simple state labels such as Detected, Ready, Running, Not Ready, and Missing.
- Disabled cards must show a reason, not only a disabled control.
- Platform integrations must not be hidden solely because they are not ready.
- Local model cards must remain visually familiar.
- The UI must avoid asking for cloud provider credentials.

## High-Level Technical Constraints

- AutoManager must continue to support local llama.cpp models exactly as before.
- The MVP must include Codex, Claude Code, and Google Antigravity.
- Cloud integrations must use already available local installation and authentication state.
- AutoManager must expose started cloud integrations through its existing local API model discovery experience.
- The smart proxy must only use cloud integrations when explicitly enabled by the operator.
- The product must account for provider readiness being different from provider detection.

## Non-Goals (Out of Scope)

- Building a full CLIProxyAPI management center inside AutoManager.
- Managing multiple cloud accounts per provider in the MVP.
- Editing provider OAuth credentials inside AutoManager.
- Creating new provider login flows inside AutoManager.
- Showing detailed quota dashboards in the MVP.
- Implementing advanced cost optimization or automatic cloud fallback policies in the MVP.
- Supporting every CLIProxyAPI provider in the MVP.
- Replacing the existing local model start/configuration flow.
- Requiring external clients to change how they call AutoManager's local API.

## Phased Rollout Plan

### MVP (Phase 1)

Included:
- Detect Codex, Claude Code, and Google Antigravity.
- Show detected integrations as catalog cards.
- Show unavailable integrations as disabled/not-ready cards with reasons.
- Allow the operator to start a detected integration.
- Expose started integrations through `/v1/models`.
- Allow explicit proxy eligibility for cloud integrations.
- Preserve local model behavior.

Success criteria:
- A local operator can open AutoManager, see at least one detected platform integration, start it, and then see it in `/v1/models`.
- Existing local model workflows continue to work without new required steps.
- A not-ready integration provides a reason that helps the operator understand the state.

### Phase 2

Included:
- Improve readiness diagnostics for provider-specific failures.
- Show basic health and availability in the proxy panel.
- Add clearer model aliasing and display names for platform integrations.
- Add Gemini CLI if product validation confirms it should be separate from Google Antigravity.
- Improve operator controls for including or excluding cloud integrations from API model listings.

Success criteria:
- Operators can distinguish installation issues, authentication issues, and provider availability issues.
- API clients can reliably identify cloud-backed model names.
- Proxy routing decisions are explainable from the UI.

### Phase 3

Included:
- Add quota and usage visibility where provider data is available.
- Support additional CLIProxyAPI-backed providers.
- Support richer routing policies involving local and cloud backends.
- Add account-pool visibility if multi-account operation becomes a product requirement.
- Add advanced fallback and preference policies for local-first, cloud-first, or task-specific routing.

Success criteria:
- AutoManager becomes a complete hybrid model operations panel for local and subscription-backed inference.
- Operators can manage availability, usage, and routing policies without leaving AutoManager.

## Success Metrics

Primary MVP metric:
- The operator can activate a detected cloud integration and see it in `/v1/models`.

Supporting metrics:
- Detection rate for installed Codex, Claude Code, and Google Antigravity tools.
- Percentage of unavailable integration cards that show a clear reason.
- Number of successful cloud integration starts from the UI.
- Time from opening AutoManager to seeing a started cloud integration in `/v1/models`.
- Regression rate for existing local model workflows.
- Percentage of proxy-eligible cloud integrations explicitly enabled by the operator.

## Risks and Mitigations

### User Confusion About Detection Versus Readiness

Risk: Operators may assume that a detected integration is fully authenticated and ready.

Mitigation: Use explicit states and reason text. A detected card can exist without being startable.

### Surprise Subscription Usage

Risk: Cloud integrations may consume limited subscription quota or trigger provider limits.

Mitigation: Do not include cloud integrations in smart proxy routing unless the operator explicitly marks them eligible.

### Provider Behavior Changes

Risk: Codex, Claude Code, and Google Antigravity may change authentication, availability, model naming, or usage limits.

Mitigation: Keep MVP status language provider-neutral and avoid promising detailed quota or account management in Phase 1.

### Scope Expansion Into Full Cloud Management

Risk: The feature could grow into a complete replacement for CLIProxyAPI dashboards before the core AutoManager use case is validated.

Mitigation: Keep Phase 1 focused on detection, cards, start, `/v1/models`, and explicit proxy eligibility.

### Trust and Privacy Concerns

Risk: Operators may worry that AutoManager is collecting provider credentials.

Mitigation: State clearly in the UI and documentation that AutoManager uses existing local tool authentication and does not ask for provider API keys in the MVP.

## Architecture Decision Records

- [ADR-001: Unified Catalog MVP for Hybrid Platform Models](adrs/adr-001.md) - AutoManager will treat detected cloud tools as first-class catalog cards and expose started integrations through the existing model/API flow.

## Open Questions

- Should Gemini CLI be a separate first-class integration from Google Antigravity in Phase 2?
- What exact card labels should be used for provider states?
- Which model names should be shown for each provider in the MVP?
- Should stopped cloud integrations remain visible in `/v1/models`, or only running integrations?
- Should the UI include a short privacy note explaining that AutoManager does not collect provider credentials?
