# AlbumentationsX MCP integration

This guide explains how to use AlbumentationsX from MCP-capable AI assistants for interactive augmentation review.

The integration is useful when you want an assistant to help with questions like:

- "Show several stronger distortion variants for these images."
- "This preview is too noisy; reduce that while keeping the useful geometric variation."
- "Build a conservative segmentation-safe pipeline and export the final Python code."
- "Compare two preview runs and summarize what changed."

The current community MCP server is [AlbumentationsX MCP](https://github.com/dKosarevsky/albu-mcp). It wraps
AlbumentationsX workflows in a typed [Model Context Protocol](https://modelcontextprotocol.io/) server so MCP hosts can
inspect transforms, validate pipelines, render bounded local previews, collect feedback, and export reproducible
pipelines.

> AlbumentationsX MCP is a community integration, not a replacement for the Python API. Use it as an assistant-facing
> review layer around local AlbumentationsX workflows.

## What MCP adds

AlbumentationsX is already a Python library with a direct API. MCP adds a structured interface that AI assistants can
call safely and repeatedly.

With AlbumentationsX MCP, a host can:

- discover AlbumentationsX transforms and supported targets;
- recommend starter pipelines for classification, detection, segmentation, OCR, or balanced workflows;
- validate pipelines before rendering;
- validate local preview requests before reading image files;
- render deterministic preview batches and contact sheets;
- render annotation overlays for bounding boxes, masks, and keypoints;
- compare baseline and candidate preview runs;
- collect concrete feedback tags such as `too_noisy:high`, `too_blurry`, or `too_distorted`;
- export accepted pipelines as Python, JSON, or YAML;
- keep a short tuning-session record for review handoff.

This is especially useful for the common human-in-the-loop flow:

1. The assistant proposes a first augmentation pipeline.
2. You inspect a small preview sheet.
3. You say what is wrong in concrete terms.
4. The assistant adjusts the pipeline.
5. You compare the previous and adjusted preview runs.
6. You export the accepted pipeline for training or review.

## Installation

Install the MCP server from PyPI with `uvx`:

```bash
uvx --from albumentationsx-mcp albumentationsx-mcp
```

For local preview work, always bound filesystem access explicitly:

```bash
uvx --from albumentationsx-mcp albumentationsx-mcp \
  --allowed-root /absolute/path/to/images-or-dataset \
  --artifact-root /absolute/path/to/albumentationsx-mcp-artifacts
```

`--allowed-root` limits which local images and masks the server can read. `--artifact-root` controls where generated
previews, contact sheets, manifests, and reports are written.

The server also supports environment variables for host setups that prefer config-only launch commands:

- `ALBU_MCP_ALLOWED_ROOTS`: `os.pathsep`-separated list of allowed local roots;
- `ALBU_MCP_ARTIFACT_ROOT`: artifact output directory;
- `ALBU_MCP_MAX_PREVIEW_RUNS`: optional preview-run retention limit.

## Host configuration

The exact MCP config format depends on the host. The examples below use a PyPI install and local preview bounds.

### Claude Desktop

```json
{
  "mcpServers": {
    "albumentationsx": {
      "command": "uvx",
      "args": [
        "--from",
        "albumentationsx-mcp",
        "albumentationsx-mcp",
        "--allowed-root",
        "/absolute/path/to/images-or-dataset",
        "--artifact-root",
        "/absolute/path/to/albumentationsx-mcp-artifacts"
      ]
    }
  }
}
```

Restart Claude Desktop after editing its MCP config.

### Cursor

Cursor uses a similar JSON server config:

```json
{
  "mcpServers": {
    "albumentationsx": {
      "command": "uvx",
      "args": [
        "--from",
        "albumentationsx-mcp",
        "albumentationsx-mcp",
        "--allowed-root",
        "/absolute/path/to/images-or-dataset",
        "--artifact-root",
        "/absolute/path/to/albumentationsx-mcp-artifacts"
      ]
    }
  }
}
```

Refresh MCP discovery after changing the config.

### Claude Code

```bash
claude mcp add-json albumentationsx '{
  "type": "stdio",
  "command": "uvx",
  "args": [
    "--from",
    "albumentationsx-mcp",
    "albumentationsx-mcp",
    "--allowed-root",
    "/absolute/path/to/images-or-dataset",
    "--artifact-root",
    "/absolute/path/to/albumentationsx-mcp-artifacts"
  ]
}'
```

### Codex

```toml
[mcp_servers.albumentationsx]
command = "uvx"
args = [
  "--from",
  "albumentationsx-mcp",
  "albumentationsx-mcp",
  "--allowed-root",
  "/absolute/path/to/images-or-dataset",
  "--artifact-root",
  "/absolute/path/to/albumentationsx-mcp-artifacts",
]
```

## First smoke check

After connecting the server, ask the host to run a smoke check before rendering previews:

```text
Read albumentationsx://examples/client-smoke, then run run_host_smoke_check.
Continue only if preview_ready is true. If it is not ready, explain the remediation actions.
```

The smoke check verifies the local environment, safe roots, artifact output, recipe recommendation, pipeline validation,
and a bounded preview request template.

## Recommended preview workflow

Use this prompt for a first local preview:

```text
Use AlbumentationsX MCP to create a small augmentation preview for this folder:
/absolute/path/to/images-or-dataset

Do not render anything until validate_preview_request returns valid=true.
Start with a low-intensity pipeline. Render one variant per image. Create a contact sheet and explain what changed.
```

A good host should follow this sequence:

1. Call `plan_dataset_onboarding`.
2. Inspect the returned `preview_request_template`.
3. Call `validate_preview_request`.
4. Call `render_preview_batch` only after validation succeeds.
5. Inspect the generated contact sheet and manifest.
6. Ask for concrete feedback before increasing intensity or batch size.

## Feedback-driven tuning

After you inspect the first preview, give specific feedback:

```text
The geometric variation is useful, but examples 3 and 8 are too noisy. Reduce noise and keep the crop/flip behavior.
Compare the adjusted preview with the baseline before exporting anything.
```

The host can then call:

- `adjust_pipeline` to produce a candidate pipeline from feedback tags;
- `render_preview_batch` to render the candidate;
- `compare_preview_runs` to compare baseline and candidate manifests;
- `export_pipeline` when you accept the result.

For multi-step reviews, ask the host to start a tuning session:

```text
Start a tuning session. Record the baseline run, the candidate run, my feedback, and the final accepted pipeline.
Export the session as Markdown when we are done.
```

## Detection and segmentation datasets

`plan_dataset_onboarding` detects common local dataset layouts and annotation formats. For sampled images, it can build
annotation-aware preview templates.

Supported annotation inputs include:

- COCO bounding boxes;
- YOLO bounding boxes;
- COCO segmentation polygons;
- COCO compressed and uncompressed RLE masks;
- YOLO segmentation polygons.

For detection workflows, the generated preview request can include `pascal_voc` bounding boxes and labels so the preview
renderer can produce overlay contact sheets.

For segmentation workflows, use `["image", "mask"]` targets. The onboarding template keeps mask-only payloads for
segmentation-safe previews and avoids sending bounding boxes to pipelines that do not declare `bbox_params`.

Example prompt:

```text
Plan a segmentation-safe preview for this dataset. Use image and mask targets. Detect COCO or YOLO-seg annotations if
available. Render a small overlay contact sheet and summarize mask coverage before and after augmentation.
```

The onboarding response includes `annotation_summary`, which reports matched sample counts, missing annotations, mask
formats, and RLE counts. This helps the host explain whether it found the expected annotations before rendering.

## Exporting a final pipeline

When the preview looks acceptable, ask for a reproducible export:

```text
Export the accepted pipeline as Python and JSON. Include the seed, targets, bbox params or mask targets if used, and a
short note about the feedback that led to this version.
```

The exported Python can be copied into a training pipeline and reviewed like ordinary AlbumentationsX code.

## Safety and privacy

AlbumentationsX MCP is designed for local, bounded preview work:

- It does not train models.
- It does not overwrite datasets.
- It does not execute arbitrary user-provided Python.
- It rejects preview paths outside `--allowed-root`.
- It writes generated artifacts under `--artifact-root`.
- It validates preview requests before reading image bytes.

Avoid uploading private datasets, production images, unredacted local paths, secrets, or credentials into public issues
or chat transcripts. Use small synthetic samples or redacted contact sheets when reporting problems.

## Troubleshooting

If the host cannot see the server, first run the command manually:

```bash
uvx --from albumentationsx-mcp albumentationsx-mcp --help
```

If preview setup fails, ask the host:

```text
Read albumentationsx://diagnostics/guide and call diagnose_environment. Explain the failing checks and remediation
actions.
```

Common issues:

- Missing `uvx` in the host environment.
- Dataset path is outside `--allowed-root`.
- Artifact directory is not writable.
- Mask paths are outside the allowed root.
- Annotation count does not match the number of input images.
- A pipeline includes bounding boxes but does not declare compatible `bbox_params`.

## More links

- MCP server repository: <https://github.com/dKosarevsky/albu-mcp>
- PyPI package: <https://pypi.org/project/albumentationsx-mcp/>
- MCP Registry entry: <https://registry.modelcontextprotocol.io/v0.1/servers?search=io.github.dKosarevsky/albu-mcp>
- AlbumentationsX documentation: <https://albumentations.ai/docs/>
