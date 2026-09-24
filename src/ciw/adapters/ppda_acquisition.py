"""Verify the complete PPDA checkout and its declared SCOUT gitlink.

Unlike the standalone telemetry projection, acquisition imports native DAF and
SCOUT code. Both source trees are checked before and after execution.
"""
from pathlib import Path

from .protocol import AdapterRefusal
from .subprocess import PinnedSubprocessAdapter
from ..pipelines import pin_map

# The acquired-dataset descriptor defines the PPDA pin; the SCOUT gitlink is fixed by that commit.
PPDA_REVISION = pin_map("acquired-dataset")["ppda"]["revision"]
VENDOR_REVISION = "5e146d5924675cd7b6e1d1ed44fb39f5da012610"
VENDOR_PATH = "vendor/scout-retrieval-agent"


class AcquisitionAdapter(PinnedSubprocessAdapter):
    def __init__(self, repository_root, **kwargs):
        root = Path(repository_root).resolve()
        vendor = root / VENDOR_PATH
        if vendor.is_symlink() or vendor.parent.is_symlink():
            raise ValueError("Acquisition vendor binding must not be a symlink")
        self.vendor = PinnedSubprocessAdapter(vendor, VENDOR_REVISION, "evidence.types", source_root=".", **kwargs)
        super().__init__(root, PPDA_REVISION, "daf.scheduling.runner", source_root=".", **kwargs)

    def _git(self, *arguments):
        data = super()._git(*arguments)
        if arguments == ("ls-tree", "-r", "-z", "HEAD"):
            expected = ("160000 commit " + VENDOR_REVISION + "\t" + VENDOR_PATH).encode()
            entries = data.rstrip(b"\0").split(b"\0")
            if entries.count(expected) != 1:
                raise AdapterRefusal("SOURCE_PIN_MISMATCH", "PPDA requires its exact declared SCOUT gitlink")
            # The generic verifier still checks every other tracked entry and
            # rejects any additional gitlink. The omitted tree is independently
            # checked by the ordinary strict adapter, never trusted by name.
            data = b"\0".join(entry for entry in entries if entry != expected) + b"\0"
        return data

    def _verify_source(self):
        self.vendor.runtime_identity()
        return super()._verify_source()

    def runtime_identity(self):
        value = super().runtime_identity()
        value["vendor"] = self.vendor.runtime_identity()
        return value
