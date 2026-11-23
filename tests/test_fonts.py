"""The type travels with the engine.

A sheet rendered on a CI runner must be the sheet rendered on a desk.  That is
only true if the faces are part of the package, not borrowed from a system font
directory, and if every role the renderer asks for resolves to one of them.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from pathlib import Path

from atlas.fonts import FONT_DIR, _FACES, font, font_path


@contextlib.contextmanager
def _shipped_fonts():
    """Ignore a developer's own ATLAS_FONT_DIR while checking the package."""
    old = os.environ.pop("ATLAS_FONT_DIR", None)
    try:
        yield
    finally:
        if old is not None:
            os.environ["ATLAS_FONT_DIR"] = old


def test_every_role_resolves_to_a_packaged_face():
    with _shipped_fonts():
        for kind, name in _FACES.items():
            path = font_path(kind)
            assert path.is_file(), f"{kind} resolves to a missing file: {path}"
            assert path.parent == FONT_DIR, f"{kind} borrowed a system face: {path}"
            assert path.name == name


def test_every_face_loads_and_has_the_expected_family():
    families = {
        "serif": "EB Garamond",
        "serif-bold": "EB Garamond",
        "serif-italic": "EB Garamond",
        "antiqua": "Gentium Book Plus",
        "antiqua-bold": "Gentium Book Plus",
        "antiqua-italic": "Gentium Book Plus",
    }
    with _shipped_fonts():
        for kind, family in families.items():
            face = font(kind, 24)
            assert face.getname()[0] == family, f"{kind} is set in {face.getname()[0]}"
            assert face.getlength("Aa 123") > 0


def test_the_heavy_face_is_heavier_than_the_text_face():
    with _shipped_fonts():
        plain = font("serif", 40)
        heavy = font("serif-bold", 40)
    assert heavy.getlength("Hamburgefonts") > plain.getlength("Hamburgefonts")


def test_a_face_is_loaded_once():
    with _shipped_fonts():
        assert font("serif", 24) is font("serif", 24)


def test_the_licences_travel_beside_the_faces():
    for name in ("OFL-EBGaramond.txt", "OFL-GentiumBookPlus.txt"):
        text = (FONT_DIR / name).read_text(encoding="utf-8")
        assert "SIL Open Font License" in text
        assert "Version 1.1" in text


def test_an_unknown_role_says_what_the_roles_are():
    try:
        font_path("sans")
    except ValueError as exc:
        assert "serif" in str(exc) and "antiqua" in str(exc)
    else:
        raise AssertionError("an unknown role was accepted")


def test_the_font_directory_can_be_overridden():
    old = os.environ.get("ATLAS_FONT_DIR")
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy2(FONT_DIR / _FACES["serif"], Path(tmp) / _FACES["serif"])
        os.environ["ATLAS_FONT_DIR"] = tmp
        try:
            path = font_path("serif")
            assert path.parent == Path(tmp)
            assert font("serif", 18).getname()[0] == "EB Garamond"
        finally:
            if old is None:
                os.environ.pop("ATLAS_FONT_DIR", None)
            else:
                os.environ["ATLAS_FONT_DIR"] = old