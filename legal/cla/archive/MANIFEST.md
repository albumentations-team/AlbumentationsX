# AlbumentationsX CLA Archive Manifest

This directory preserves the exact operative texts needed to interpret CLA
acceptance records. The SHA-256 digest, not a filename alone, identifies the
accepted text.

| CLA text | Source or effective date | Archive file | Representation | SHA-256 of CLA bytes |
|---|---|---|---|---|
| Version 1 initial | commit `c1720fbab8209450328ef2e68f0ddc0c4806f7a8`, 2025-06-19 | `CLA-v1-c1720fb.md.base64` | Base64 of the original 7,360 bytes; the original had no final newline | `0318da3ff5c1d7b6e67ab0affa59e97b7e64902ae591e0c2d9e39a6299f835e9` |
| Version 1 formatting revision | commit `09b767a40d13a90872e23ae4fc2f047b5527199a`, 2025-06-19 | `CLA-v1-09b767a.md` | Exact 7,365-byte Markdown file | `d3ce911a802d2cea06f4deeb406d5667155305f01ba3ef0bd550d41d363b50ed` |
| Version 2.0 | effective 2026-07-14 | `CLA-v2.0-2026-07-14.md` | Exact Markdown file, identical to repository-root `CLA.md` at adoption | `e46696db80156d7f50dd017db61f39d903bcd841dcb92a2340dfc6ac909346d7` |

The two Version 1 files differ only in Markdown fence formatting and the final
newline, but both byte versions are retained because an acceptance record may
refer to either repository state.

CLA Assistant used the public Gist
`https://gist.github.com/ternaus/df31e11d8a3180ba5520f72b72d57198`, immutable
revision `3115a7364f5ab8a58a7e7ffa51dfdf1ec8a5b006`, created
`2025-06-19T21:58:29Z`. The hosted raw file was 7,360 bytes with SHA-256
`0318da3ff5c1d7b6e67ab0affa59e97b7e64902ae591e0c2d9e39a6299f835e9`,
byte-identical to `CLA-v1-c1720fb.md.base64`. Hosted Version 1 signer records
therefore map to the initial Version 1 digest, not the 7,365-byte formatting
revision.

To verify the base64-encoded initial file on macOS:

```bash
base64 -D -i legal/cla/archive/CLA-v1-c1720fb.md.base64 | shasum -a 256
```

On systems using GNU coreutils, use `base64 --decode` instead of `base64 -D`.

Acceptance records are not committed here because they may contain personal or
company information. The record system must retain, at minimum, the accepting
identity, timestamp, individual-versus-entity path, covered identities for an
Entity Acceptance, CLA version, and the SHA-256 identifier from this manifest.

Changing `CLA.md` requires a new version, a new immutable archive file, a new
manifest row, and explicit acceptance of that new version. Never overwrite an
archived CLA file or reinterpret an old acceptance as consent to a later text.
