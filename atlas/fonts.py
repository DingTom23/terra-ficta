"""The type the atlas is set in.

The faces travel with the engine, so a sheet drawn on Windows, Linux or a bare
CI runner is the same sheet: EB Garamond for the map and its prose, Gentium
Book Plus for the titling.  Both are under the SIL Open Font License, which
permits commercial use and redistribution; their licence texts sit beside the
font files.

``ATLAS_FONT_DIR`` selects another directory containing the same filenames --
for an edition set in a licensed face, or for trying another one.  The lookup
happens on every call, so a process can change the directory between sheets.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

from PIL import ImageFont

FONT_DIR = Path(__file__).resolve().parent / "fonts"

# role -> file.  The three serif faces carry the map and the dictionary; the
# three antiqua faces carry titles, keys and running heads.
_FACES = {
    "serif": "EBGaramond-Regular.ttf",
    "serif-bold": "EBGaramond-Bold.ttf",
    "serif-italic": "EBGaramond-Italic.ttf",
    "antiqua": "GentiumBookPlus-Regular.ttf",
    "antiqua-bold": "GentiumBookPlus-Bold.ttf",
    "antiqua-italic": "GentiumBookPlus-Italic.ttf",
}


def font_path(kind: str) -> Path:
    """Where the face for ``kind`` lives, or a clear error naming the role."""
    try:
        name = _FACES[kind]
    except KeyError:
        roles = ", ".join(sorted(_FACES))
        raise ValueError(f"unknown font role {kind!r}; expected one of {roles}") from None

    override = os.environ.get("ATLAS_FONT_DIR")
    directory = Path(override).expanduser() if override else FONT_DIR
    path = directory / name
    if not path.is_file():
        raise FileNotFoundError(
            f"font {name!r} for role {kind!r} is missing from {directory}. "
            "The faces are packaged with terra-ficta; reinstall it, or set "
            "ATLAS_FONT_DIR to a directory containing the same filenames."
        )
    return path


@functools.cache
def _load(path: str, size: int):
    return ImageFont.truetype(path, size)


def font(kind: str, size: int):
    """A cached, loaded face.  Sizes are integers, as FreeType wants them."""
    return _load(str(font_path(kind)), int(size))