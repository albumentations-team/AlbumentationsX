# Security Policy

AlbumentationsX takes security reports seriously. Please report suspected vulnerabilities privately so users have time to update before details are disclosed publicly.

## Report A Vulnerability

Use GitHub private vulnerability reporting for this repository whenever possible:

- Open the repository Security tab.
- Use `Report a vulnerability`.
- Include a minimal reproduction, affected version, impact, and any proposed fix or mitigation.

If GitHub private reporting is unavailable, email `vladimir@albumentations.ai` with the subject line `AlbumentationsX security report`.

Do not open public GitHub issues for suspected vulnerabilities.

## Scope

This policy covers vulnerabilities in code, packaging, release automation, or project-controlled infrastructure for the open-source AlbumentationsX distribution, including:

- Python package contents published from this repository
- GitHub Actions release and packaging workflows
- Project-maintained metadata, manifests, and release artifacts

The following are generally out of scope unless they create a project-controlled exploit path:

- Vulnerabilities in third-party dependencies with no AlbumentationsX-specific exposure
- Security issues in downstream applications that only use AlbumentationsX as a dependency
- Misconfiguration of user environments outside project-controlled releases

## Supported Versions

AlbumentationsX uses the following support policy for security fixes:

| Version line | Security support |
| --- | --- |
| Latest stable minor release line | Supported |
| Previous stable minor release line | Best effort for high-severity issues |
| Older releases | Not supported |

If a fix is not backported, the remediation path is to upgrade to the latest supported release.

## Coordinated Disclosure

Please keep reports private until a fix, mitigation, or advisory is ready.

AlbumentationsX will aim to:

- acknowledge receipt within 5 business days
- complete initial triage within 10 business days
- provide a status update after triage, including severity and planned remediation path
- publish a GitHub Security Advisory and/or release note after a fix or mitigation ships

Fix timelines depend on severity, exploitability, and maintainer availability. High-severity issues in supported releases take priority over feature work.

## Report Handling

When a report is accepted as a vulnerability, maintainers will:

- confirm affected versions and impact
- prepare and validate a fix or mitigation
- coordinate embargo timing with the reporter when practical
- release the fix through the normal release process
- publish public guidance once users can act on it

## Safe Harbor

Good-faith security research intended to improve the project is welcome. Please avoid privacy violations, destructive testing, social engineering, and service disruption.
