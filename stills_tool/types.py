from dataclasses import dataclass
from pathlib import Path


class CancelledError(RuntimeError):
    pass


@dataclass
class Options:
    mode: str = "smart"
    count: int | None = None
    format: str = "png"
    bit_depth: int = 8
    color: str = "srgb"
    output: str | Path | None = None
    report: bool = False
    recursive: bool = False

    def validate(self):
        if self.mode not in {"smart", "legacy"}:
            raise ValueError("Mode must be smart or legacy.")
        if self.format not in {"png", "tiff", "jpeg"}:
            raise ValueError("Format must be png, tiff or jpeg.")
        if self.bit_depth not in {8, 16}:
            raise ValueError("Bit depth must be 8 or 16.")
        if self.format == "jpeg" and self.bit_depth != 8:
            raise ValueError("JPEG supports 8-bit output; choose PNG or TIFF for 16-bit.")
        if self.color not in {"srgb", "source"}:
            raise ValueError("Color mode must be srgb or source.")
        if self.color == "source" and (self.bit_depth != 16 or self.format == "jpeg"):
            raise ValueError("Source color requires 16-bit PNG or TIFF.")
        if self.count is not None and self.count < 1:
            raise ValueError("Count must be a positive integer.")
        if self.mode == "legacy" and (self.count is not None or self.format != "png"
                                     or self.bit_depth != 8 or self.color != "srgb"
                                     or self.output is not None or self.report):
            raise ValueError("Legacy mode keeps its original PNG8 output beside the source; smart options are unavailable.")
