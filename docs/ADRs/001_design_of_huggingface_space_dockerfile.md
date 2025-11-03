# Design of HuggingFace Space Dockerfile

- Status: Ongoing
- Deciders: Martin Lellep (@PellelNitram)
- Drivers: Martin Lellep (@PellelNitram)
- PRD: None
- Date: 2025-10-04

## Context

*Explain the background and the context in which the decision is being made. Include any relevant information about the problem, constraints, or goals.*

## Decisions

*State the decision that has been made. Be clear and concise.*

- In the future, download models at build time into the Docker image from Github release page. In the
  very far future, pull them from HuggingFace at run-time.
- Add `xournalpp` binary to Docker image so that the `xopp` file can be exported as PDF prior to
  execution of the HTR pipeline.

## Consequences

*Describe the consequences of the decision. Include both positive and negative outcomes, as well as any trade-offs.*

## Alternatives Considered

*List and briefly describe other options that were considered and why they were not chosen.*

## References

*Include links or references to any supporting documentation, discussions, or resources.*