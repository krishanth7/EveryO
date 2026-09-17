# Security Policy

EveryO is designed so that the interesting attack surface is small: it needs no
credentials, makes no network calls from its core, and never unpickles a model.
This document says what is supported, how to report a problem, and what the
threat model actually is.

## Supported versions

| Version | Supported |
| --- | --- |
| 0.1.x | ✅ Security fixes applied |
| < 0.1 | ❌ Pre-release, unsupported |

Security fixes land on the latest release.

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Use GitHub's private vulnerability reporting, which is visible only to the
maintainers:

> **[Report a vulnerability →](https://github.com/krishanth7/EveryO/security/advisories/new)**
>
> (Repository → **Security** tab → **Report a vulnerability**)

If private reporting is unavailable to you, open a regular issue containing
**only** a request for a private contact channel — no details, no reproduction —
and a maintainer will follow up.

No dedicated security email address exists for this project, and none is
invented here.

### What to include

* EveryO version (`everyo info`), Python version and platform
* A minimal reproduction
* What an attacker gains — code execution, data exfiltration, memory corruption
* Any proposed fix, if you have one

### What to expect

| Stage | Target |
| --- | --- |
| Acknowledgement | within 7 days |
| Initial assessment | within 14 days |
| Fix or mitigation plan | communicated once assessed |

EveryO is a small volunteer project, so these are honest targets rather than a
contractual SLA. You will be credited in the advisory and the changelog unless
you ask not to be.

**Please give us a reasonable window to ship a fix before disclosing publicly.**
We will keep you updated and will not ask for an open-ended embargo.

## Threat model

### Model files are the main attack surface — and they are pickle-free by design

An `.evo` file is a ZIP archive holding:

* `manifest.json` — metadata and the architecture as plain JSON, and
* `parameters.npz` — arrays read with **`allow_pickle=False`**.

Models are rebuilt by looking class names up in a registry of modules that
explicitly opted in with `@register_module`; an unknown name is refused. Nothing
in a model file is executed, evaluated or unpickled.

Loading a malicious archive should therefore do no worse than raise
`EveryOSerializationError`. **If you find a way to make `everyo.load()` execute
attacker-controlled code, that is a vulnerability — please report it.**

Even so, treat model files like any other untrusted input.

### No credentials, no network, no telemetry

EveryO requires no API keys, tokens or accounts. The core library makes no
network requests; its datasets are generated locally from a seed. Nothing about
your usage is collected or transmitted, by the library or by the CLI.

If you ever observe EveryO making an unexpected outbound connection, treat that
as a security bug and report it.

### Native code

The CUDA extension is optional and built from source in this repository. Kernels
use bounds checking, CUDA error checking and RAII-owned device memory, but they
are native code: bugs there could cause memory corruption. Report anything that
reads or writes out of bounds.

### Dependencies

The core depends only on NumPy and Matplotlib. TensorFlow, PyYAML and pybind11
are optional extras. Vulnerabilities in those projects should be reported to
them directly; if EveryO *uses* one of them unsafely, that is ours.

## Scope

**In scope**

* Code execution or memory corruption from loading a model file
* Code execution from processing input data or configuration
* Memory safety issues in the CUDA extension
* Anything causing EveryO to transmit data off the machine
* A bypass of the pickle-free loading guarantee

**Out of scope**

* Denial of service from deliberately enormous inputs (allocate what you ask for)
* Numerical inaccuracy that is not a memory-safety problem
* Vulnerabilities in dependencies, unless EveryO misuses them
* Issues requiring an already-compromised machine or a malicious local user

## Hardening the repository itself

Contributors: never commit `.env` files, tokens, keys or credentials. The
[`.gitignore`](.gitignore) blocks the usual suspects, and EveryO needs none of
them — if a change appears to require a secret, that is a design problem worth
raising first.
