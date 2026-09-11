# Telemetry and data handling

Maintained by Albumentations, LLC. Contact: <vladimir@albumentations.ai>.
Source reviewed: repository version 2.4.9, September 11, 2026.

AlbumentationsX performs augmentation in the environment where you run Python.
Its telemetry is **enabled by default** and sends usage metadata to Mixpanel.
We collect these statistics solely to support future product analysis and
development, including transform priorities and environment support.

## Disable telemetry

Set either environment variable **before importing `albumentations`**, including
in notebook kernels and worker processes:

```bash
export ALBUMENTATIONS_NO_TELEMETRY=1
```

`ALBUMENTATIONS_OFFLINE=1` also disables library telemetry. Both variables accept
`1` or `true` (case-insensitive). Restart an already running process or notebook
kernel after changing its environment.

To disable telemetry for an individual pipeline:

```python
import albumentations as A

transform = A.Compose([A.HorizontalFlip(p=0.5)], telemetry=False)
```

Apply `telemetry=False` to each independently constructed pipeline whose events
you want to suppress. The environment variables disable telemetry globally.
Disabling telemetry does not change augmentation results.

## What is sent

The library can send a `Compose Init` event when a pipeline is constructed.
It does not send an event for each image processed. The outbound event contains:

| Data | Contents |
| --- | --- |
| Identifier | A randomly generated UUID, reused across events and processes when its local file is available |
| Event metadata | Event name, timestamp, and a fresh UUID for event deduplication |
| Software | AlbumentationsX version, Python major/minor version, and operating system description |
| Hardware | CPU model or architecture, first CUDA GPU name when available, and total RAM when available |
| Environment | A detected category such as local, Docker, Jupyter, Colab, or Kaggle |
| Pipeline | Transform class names, their count, and a SHA-256 hash of the ordered names |
| Annotation configuration | Whether bbox processors, keypoint processors, both, or neither are configured |

The payload contains no image pixels, masks, annotation coordinates, filenames,
or transform parameter values. Custom transform class names are included too.
The pipeline hash is derived from names, not from image data or parameter values.
Local execution traces and saved applied parameters are not part of this event.

The library does not link the UUID to your name or email and does not add an IP
address field to the event. The UUID allows events from the same installation
to be associated over time.

The client sends events by HTTPS to `https://api.mixpanel.com/track`.
[Mixpanel documents](https://docs.mixpanel.com/docs/tracking-best-practices/geolocation)
default enrichment of events with country, region, and city derived from the
connection's source IP address; it states that the IP is then removed from the
event before ingestion. The AX request does not explicitly disable this
geolocation behavior. This describes the provider's documented default; actual
stored events and project settings have not been verified for this notice.

The client skips telemetry when it detects CI or pytest, deduplicates transform
lists within a process, and rate-limits sends. These are implementation controls;
use an explicit opt-out when you want telemetry disabled.

## Local storage and previously collected data

The identifier is stored in `albumentationsx/user_id.json` below the user config
directory. `ALBUMENTATIONS_CONFIG_DIR` can override that base directory; otherwise
the library uses `APPDATA` on Windows or `XDG_CONFIG_HOME` on Unix, with platform
defaults when those variables are absent. The file remains until removed or
changed. Removing it can cause a new identifier to be created on the next enabled
run; removal is not an opt-out.

Opting out stops future telemetry; it does not delete events already received by
Mixpanel. Provider-side retention, storage region, and deletion settings have not
been verified for this notice, so no fixed retention period or storage-region
guarantee is stated here.

For telemetry questions or requests concerning previously collected data, email
<vladimir@albumentations.ai>. The local UUID can help locate associated events;
do not post it in a public issue. No dataset or image upload is needed for the
request.

## Optional network integrations

The optional Hugging Face Hub integration can upload or download pipeline
configurations when you invoke its methods. `ALBUMENTATIONS_OFFLINE` controls
library telemetry; it does not block those separately invoked Hub operations
or network activity in your own code and dependencies.

## Implementation references

- [Event fields](../albumentations/core/analytics/events.py) and [collectors](../albumentations/core/analytics/collectors.py)
- [Mixpanel request](../albumentations/core/analytics/backends/mixpanel.py)
- [Settings and environment controls](../albumentations/core/analytics/settings.py)
- [Telemetry client](../albumentations/core/analytics/telemetry.py) and [identifier storage](../albumentations/core/analytics/user_id.py)
- [Optional Hub integration](../albumentations/core/hub_mixin.py)
