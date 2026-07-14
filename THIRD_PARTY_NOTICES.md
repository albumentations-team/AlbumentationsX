# Third-Party Notices

AlbumentationsX contains material derived from the legacy `albumentations`
project. Legacy Albumentations release 2.0.8 was distributed under the MIT
License with the following copyright holders named in its notice:

> Copyright (c) 2017 Vladimir Iglovikov, Alexander Buslaev, Alexander Parinov,

The complete, byte-for-byte license text from the legacy `2.0.8` tag is
preserved in
[`LICENSES/MIT-Albumentations-2.0.8.txt`](LICENSES/MIT-Albumentations-2.0.8.txt).
Its SHA-256 digest is
`bea4dc8e93ae2784bccd45f1cdba53da97b99646bca390c7d725e17b72dc2180`.
That MIT notice applies to material inherited from that legacy release. The
current repository default for later AlbumentationsX work is stated in
[`LICENSE`](LICENSE); the history and boundary are described in
[`LICENSE_HISTORY.md`](LICENSE_HISTORY.md).

The current repository notice does not retroactively withdraw or narrow the
MIT permissions granted for inherited legacy material or the license metadata
that accompanied an earlier published AlbumentationsX release.

Files that carry their own license notice remain subject to that notice. This
document does not replace those file-level notices.

## Liberation Serif Bold Test Font

The source repository redistributes
`tests/files/LiberationSerif-Bold.ttf` (SHA-256
`d754ba427cfe0bca54ae052384baa8f842da5bd6550ad4da024ac441e7a7d5ce`) for
text-rendering tests. Its embedded metadata states:

> Digitized data copyright (c) 2010 Google Corporation, with Reserved Font
> Arimo, Tinos and Cousine.
> Copyright (c) 2012 Red Hat, Inc., with Reserved Font Name Liberation.

The font is licensed under the SIL Open Font License, Version 1.1. Its complete
notice and license are preserved at `LICENSES/OFL-1.1.txt`; the authoritative
license text is published by the
[OFL steward](https://openfontlicense.org/open-font-license-official-text/).

The font and `LICENSES/OFL-1.1.txt` are repository-only test assets. The Hatch
build configuration excludes both from wheel and sdist artifacts; package
license metadata therefore continues to list only the four outbound files
that accompany those artifacts. The repository copy of the font remains under
OFL-1.1 and is not relicensed under the repository default.

Runtime and optional dependencies declared in `pyproject.toml` are obtained as
separate packages and remain subject to their respective licenses. They are not
relicensed by AlbumentationsX.

Maintainers must update this document and the `LICENSES/` directory when
vendoring, copying, or substantially adapting material whose license or notice
must be preserved.
