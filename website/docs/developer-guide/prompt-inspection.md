# Inspecting resolved cron prompt state

`agent.prompt_inspection.inspect_cron_prompt` renders the existing native
system-prompt builder from resolved cron inputs without creating an `AIAgent`,
provider client, context compressor, memory provider, MCP connection or session
database. It is a construction primitive, not a source-admission or request
attestation API.

The normal constructor and inspection share prompt configuration and tool-state
initializers in `agent.prompt_state`. The scheduler and inspection also share
cron's prompt-affecting constructor flags: preserve native SOUL identity,
disable memory, and discover project context only for a configured workdir.
Existing ordinary constructor and system-prompt behavior remains the default.

Callers must supply the actual native-resolved model/provider, context length,
tool schemas, merged constructor configuration, session identity, workdir state
and prefill messages. The API retains nonempty schemas and prefills. Passing an
empty list to avoid resolving tools, dropping prefills, or substituting a new
identity does not establish equivalence to a real cron run.

Inspection resolves SOUL, context, skill, environment and subscription guidance
through the same native helpers. Its read-only source scope skips profile
preparation and optional skills-cache persistence, which are separate from
rendering. An uninitialized profile refuses rather than generating a different
identity fallback. Exiting the scope restores ordinary secure profile
preparation, including after refusal.

Dynamic dependencies are still explicit. Credential or entitlement resolution,
environment probes, tool availability checks, uncached imports and caller-side
source resolution may require operations that a strict inspector cannot permit.
The public inspector installs a native synchronous audit scope that refuses
observed socket/process operations, credential-file opens and persistent-write
operations before action. Mandatory denial bypasses native optional
`except Exception` fallbacks and becomes
`PromptInspectionUnavailable("native_instruction_inspection_unavailable")` at
the public boundary. `PromptSourceUnavailable` is the internal mandatory
source-refusal signal. Ordinary execution outside the scope is unchanged.

Use a separate isolated process and qualified native owners as well. The audit
scope is a diagnostic boundary for trusted native Python code, not a sandbox
for hostile native extensions, preopened resources or arbitrary plugin threads.
A cold import or unsupported dynamic dependency can therefore remain explicitly
unavailable; it must never produce an equivalent empty fragment.

A returned string establishes construction only. Independent provenance,
source/currentness binding, exact cron job-prompt composition, instruction
comparison and complete final request comparison remain the caller's proof
obligations. A first observed request or a dictionary containing prompt hashes
cannot replace those obligations.

The focused real-constructor control uses a disposable initialized profile,
explicit context configuration, configured environment probing disabled and
nonempty native file-tool schemas. It compares exact native instructions and
tool schemas while denying network/process/credential/write operations during
the inspection interval and checking the fixture postimage. This supports that
configuration in a warm native process; it does not qualify all production
configurations or a cold-process provenance witness.
