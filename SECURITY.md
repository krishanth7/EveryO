# Security policy

## Supported versions

EveryO is at version 0.1.0. Security fixes are applied to the latest release.

| Version | Supported |
| --- | --- |
| 0.1.x | Yes |

## Reporting a vulnerability

Please report security issues privately rather than in a public issue.

Use GitHub's private vulnerability reporting: open the repository's
**Security** tab and choose **Report a vulnerability**. That channel is visible
only to the repository maintainers.

If private reporting is not enabled on the repository, open a regular issue
saying only that you have found a security problem and asking for a private
contact — please do not include the details in the public issue.

No dedicated security email address exists for this project, and none is
invented here.

When reporting, please include:

* the affected version and platform,
* a minimal reproduction,
* what an attacker could achieve.

You can expect an acknowledgement as soon as a maintainer sees the report. As a
small project there is no guaranteed response time.

## Security-relevant design notes

**Model loading never executes code.** An `.evo` archive is a ZIP holding JSON
metadata and a NumPy archive read with `allow_pickle=False`. Models are rebuilt
by looking class names up in a registry of modules that opted in with
`@register_module`; an unknown name is refused. Loading a malicious file should
at worst raise `EveryOSerializationError`. If you find a way to make
`everyo.load()` execute attacker-controlled code, that is a vulnerability —
please report it.

Even so, treat model files like any other untrusted input and prefer archives
from sources you trust.

**No credentials, no network, no telemetry.** EveryO needs no API keys, tokens
or accounts. The core library makes no network requests, and the datasets it
ships are generated locally from a seed. Nothing about your usage is collected
or transmitted.

**The CUDA extension is native code.** It is optional and built from source in
this repository. Bugs there could cause memory errors; kernels use bounds
checking and CUDA error checking, and device memory is owned by RAII wrappers.

**Dependencies.** The core depends on NumPy and Matplotlib. TensorFlow, PyYAML
and pybind11 are optional extras.

## Scope

In scope: code execution or memory corruption from loading a model file,
processing input data, or using the CUDA extension; anything that would cause
EveryO to transmit data off the machine.

Out of scope: denial of service from deliberately enormous inputs; numerical
inaccuracy that is not a memory-safety problem; vulnerabilities in NumPy,
TensorFlow or other dependencies (report those upstream).
