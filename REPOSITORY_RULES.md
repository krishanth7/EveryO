# Repository Rules

These rules apply to issues, discussions, pull requests, reviews, and other participation in EveryO. They complement the [Code of Conduct](CODE_OF_CONDUCT.md), [Contributing Guide](CONTRIBUTING.md), and [Security Policy](SECURITY.md).

## Participation

- Be respectful, specific, and focused on the project.
- Search existing issues and discussions before opening a duplicate.
- Use the provided templates and include enough information to reproduce or evaluate the request.
- Keep one primary bug or proposal per issue.
- Do not use issues for private vulnerability reports, personal support, advertising, or unrelated promotion.
- Do not post secrets, credentials, private datasets, personal information, or proprietary code without authorization.

## Technical contributions

- Open an issue or discussion before large architectural work.
- Keep pull requests focused and explain the problem, approach, and trade-offs.
- Add tests for behavioral changes. Gradient work should include numerical checks where practical.
- Preserve CPU-only operation unless a change is explicitly scoped to an optional backend.
- Keep CUDA and TensorFlow optional.
- Do not claim performance, accuracy, compatibility, or hardware support without reproducible evidence.
- Run the documented test, lint, formatting, and compilation checks.
- Update documentation when public behavior changes.

## Prohibited submissions

The project will reject:

- malicious code, credential harvesting, telemetry added without approval, or undisclosed network activity;
- copied code or data without compatible licensing and attribution;
- fabricated benchmarks, test results, citations, contributors, or adoption claims;
- generated bulk changes that have not been reviewed and validated;
- spam, deceptive promotion, referral links, paid-placement requests, or advertisements disguised as contributions;
- harassment, discrimination, doxxing, or threats.

## Reviews and merging

Approval is based on correctness, scope, maintainability, tests, documentation, security, and project direction. Passing CI does not guarantee acceptance. Maintainers may request changes, close stale or out-of-scope work, squash commits, or postpone a proposal.

Direct pushes to the default branch should be avoided. Changes should normally arrive through a reviewed pull request with passing required checks.

## Enforcement

Maintainers may edit, hide, lock, close, or remove content that violates these rules and may restrict participation for repeated or serious violations. Security and safety concerns may be handled privately.

By participating, you agree to follow these rules and the Code of Conduct.
