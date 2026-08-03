### 2026-08-02: Coordinator canary absence is unobservable from a spawned agent
**By:** Fact Checker
**What:** A spawned agent must not treat absence of `SQUAD_COORDINATOR_CANARY_a8f3` from its own
context as evidence that the coordinator file is truncated. Spawned agents do not receive the
coordinator's agent instructions, so the observation is invariant under both healthy and truncated
coordinator states.
**Why:** The check fired falsely in this audit and reportedly in five of six prior parallel
delegations. Its triggering condition is guaranteed true in the observer's frame, so it cannot
convict. The check must run in the coordinator's frame or carry a coordinator-generated attestation
into the spawn; otherwise it should report `UNOBSERVABLE`, not block.
