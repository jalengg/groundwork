# KMGen Agent Team Communication DB

This directory is the **shared memory and communication layer** for the KMGen agent team. All teammates read from and write to files here. It serves as a broadcast channel, shared knowledge store, and coordination mechanism.

## File Structure

### Broadcast Channel
- `broadcast.md` — Append-only message log. Any teammate can post. All teammates read before acting. Format: `[TIMESTAMP] [AGENT_NAME]: message`

### Shared Knowledge
- `baseline.md` — Current IAE numbers per plot. Updated by Quality Gate after each successful cycle. READ by all agents before proposing changes.
- `edge-cases.md` — Known failure modes, their categories, and status (open/investigating/resolved). Updated by Visual QA.
- `prompt-evolution.md` — Log of what skill changes helped/hurt. Each entry: date, change, IAE delta, verdict. Updated by Documentation agent.
- `decisions.md` — Architectural decisions and their rationale. Any agent can propose, Challenger can contest.

### Gate State
- `gate-status.md` — Current cycle state: which phase we're in, what's pending, who's blocked. Updated by coordinator.
- `challenger-verdicts.md` — History of Challenger verdicts. Other agents reference this to avoid re-proposing rejected ideas.

### Session Context
- `session-log.md` — Last 10 session summaries. Compacted by Documentation agent.

## Rules
1. **Read before write**: Always read `broadcast.md` and `baseline.md` before starting work.
2. **Append-only broadcasting**: Never delete from `broadcast.md`. Only Documentation agent compacts it (monthly).
3. **Structured output**: All agents use structured formats (VERDICT, FAILURES, etc.) so other agents can parse.
4. **No file >300 lines**: Documentation agent enforces compaction.
5. **Timestamp everything**: Use ISO 8601 format.
