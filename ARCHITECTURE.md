# Architecture Principles

This document pins the architectural principles that govern changes to this
repo. It exists so that future changes — by you, by a collaborator, or by
an agent acting on your behalf — start from a fixed reference point instead
of whichever framework was most recently quoted.

The format borrows from ATAM (Architecture Tradeoff Analysis Method,
Kazman/Klein/Clements, SEI 2000): a *utility tree* ranks the quality
attributes you care about, and every architectural decision is classified
as a *sensitivity point* or a *tradeoff point* against that tree.

## Utility tree

In priority order, highest first. When a decision improves a higher-priority
attribute at the cost of a lower one, the decision is acceptable; the
reverse is not.

1. **Maintainability** — small surface area, clear data flow, code an
   individual can hold in their head. This repo is agent-assisted hobby
   work; the maintainer's available time is the binding constraint.
2. **Retrieval precision (MRR)** — when the right answer is found, it
   should rank first. Downstream consumers (Codex, query CLIs) act on the
   top result.
3. **Retrieval recall (full-coverage)** — the right answer should be
   findable in the returned set, even if not at rank 1.
4. **Reproducibility** — manifests pin the source SHA, embedding model,
   and chunk parameters so a query against the index is the same query
   tomorrow.
5. **Latency** — local-loop iteration speed matters more than per-query
   latency, but slow queries still erode the loop.
6. **Multilingual capability** — the embedding model supports it; not
   currently exercised by the eval harness.

## Principle hierarchy

When two design principles point in opposite directions, the higher number
yields to the lower.

1. **Simple over easy (Hickey)** — never complect unrelated concerns.
   Data and serialization are separate. Parsing and I/O are separate.
   Configuration and behavior are separate. This is the strongest principle
   because complecting is invisible until you try to change something.
2. **Deep modules (Ousterhout)** — minimize public API surface; hide
   implementation behind simple interfaces. Two functions that do the same
   work for different callers should be one function.
3. **Functional core, imperative shell (FCIS, Bernhardt)** — keep pure
   transformation separate from I/O. Apply when it doesn't violate #1 or
   #2; specifically, do not split a deep module just to surface its purity.

These three almost always agree on what to do. The hierarchy only matters
when they conflict.

## Decision process

Before any non-trivial architectural change, classify it.

**Sensitivity point**: affects exactly one utility-tree attribute.
- Example: swapping `mxbai-embed-large` for `bge-m3` only changes
  retrieval quality. The decision rule is purely empirical — does the eval
  go up or down?

**Tradeoff point**: affects two or more attributes in opposing directions.
- Example: removing the hybrid retrieval pipeline improved
  maintainability and MRR but hurt recall. The decision rule is the
  utility tree — does the change push higher-priority attributes up at
  the cost of lower ones?

**Risk**: the change makes a known case worse and you accept it.
- Example: post-refactor, two eval cases (`for-range`, `iota`) systematically
  miss. Documented in the commit message rather than hidden.

**Non-risk**: the change cannot regress any attribute under any reasonable
scenario. (Most refactors that pass quality gates fall here.)

For each proposed change, write a short paragraph in the commit message
that names: which utility-tree attributes it touches, in which direction,
and which principle resolves the call. The discipline is the deliverable;
the document is the reference.

## Worked example

The hybrid → pure-cosine refactor from this codebase's history is a
canonical tradeoff point.

| Aspect | Direction | Magnitude |
|--------|-----------|-----------|
| Maintainability | ↑ | -983 LOC, -1 module, -26 tests |
| MRR | ↑ | 0.746 → 0.889 |
| Recall (full coverage) | ↓ | 19/21 → 17/21 |
| Latency | ↑ | one fewer index hit per query |
| Reproducibility | flat | manifest still pins everything |

**Classification**: tradeoff point. Maintainability (priority 1) and MRR
(priority 2) improved; recall (priority 3) regressed by ~10%.

**Resolution**: maintainability and MRR are higher in the utility tree
than recall, so the change is acceptable. The two now-failing cases were
documented as risks rather than hidden.

**Principle alignment**: the refactor primarily removed *complecting*
(score blending mixed cosine and rank into one number; the mode dispatch
braided four behaviors in one function). That's a Hickey-simple win,
not just a code-volume cut.

## What this isn't

- **Not a process for multi-stakeholder enterprise architecture.** Real
  ATAM is a multi-day workshop with named risks and stakeholder utility
  votes. This is a one-page decision framework adapted from it.
- **Not a substitute for measurement.** The utility tree tells you *what*
  to optimize; the eval harness, contracts, and quality gates tell you
  *whether* a change actually does what its commit message claims.
- **Not immutable.** When you find that the priority order produces a
  decision you disagree with, the right move is to update this document
  and re-justify, not to override silently.

## Pointers

- Hickey, *Simple Made Easy* (2011) — InfoQ talk. The "complecting" diagnostic.
- Ousterhout, *A Philosophy of Software Design* (2018) — chapters 4–8 on
  modular design and interface depth.
- Bernhardt, *Boundaries* (2012) — the original FCIS talk.
- Kazman/Klein/Clements, *ATAM: Method for Architecture Evaluation*
  (CMU/SEI-2000-TR-004) — sensitivity points, tradeoff points, utility
  trees.
