Control and view epistemic verification (hallucination detection).

## Usage

- `/verify` — Show current verification status and configuration
- `/verify on` — Enable verification for this session
- `/verify off` — Disable verification for this session
- `/verify report` — Show the last verification report
- `/verify claim "..."` — Verify a specific claim against context
- `/verify feedback <claim_id> correct|incorrect` — Provide feedback on verification accuracy
- `/verify stats` — Show verification feedback statistics
- `/verify mode <full|sample|critical>` — Set verification mode

## Verification Modes

| Mode | Description | Cost |
|------|-------------|------|
| `full` | Verify all claims | Highest |
| `sample` | Verify critical + sample of claims (default) | Medium |
| `critical` | Only verify claims marked as critical | Lowest |

## Configuration

Check `~/.claude/rlm-config.json` for verification settings:
- `verification.enabled`: true | false
- `verification.mode`: "full" | "sample" | "critical_only"
- `verification.support_threshold`: 0.0-1.0 (default 0.7)
- `verification.sample_rate`: 0.0-1.0 (default 0.3)
- `verification.on_failure`: "flag" | "retry" | "ask"

## When to Use

Enable verification when:
- Accuracy is critical (production documentation, code review)
- Working with unfamiliar codebases
- Generating technical specifications
- Debugging claims that seem wrong

Disable verification when:
- Quick iterations where speed matters more than precision
- Creative/exploratory tasks
- You're confident in the context

## Feedback Loop

Use `/verify feedback` to improve verification accuracy:
```
/verify feedback c1 correct      # The claim was accurately verified
/verify feedback c2 incorrect    # The claim was incorrectly flagged (false positive)
```

This feedback is stored and used to calibrate verification thresholds over time.

## Verification Report

The report shows:
- **Claims**: Total claims extracted from response
- **Verified**: Claims that passed verification
- **Flagged**: Claims that failed (with reasons)
- **Gaps**: Evidence gaps identified (unsupported, contradicted, phantom citations)
- **Confidence**: Overall confidence score

## Example Output

```
Verification Report
───────────────────
Response: resp-abc123
Mode: sample (30% sampling)

Claims: 5 total, 4 verified, 1 flagged
Confidence: 0.85

Flagged Claims:
  [c3] "The API returns XML data"
       Reason: unsupported
       Suggestion: Provide supporting evidence or remove claim

Evidence Gaps:
  - partial_support (c2): Claim goes beyond available evidence
```
