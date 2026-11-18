"""Dataset definition: which sources exist and where their raw files live.

Run-time settings (output directory, parallelism) are CLI options, not config,
so this file stays the same on every machine. Relative source paths are resolved
against the config file's directory, so the CLI works from any directory.

    sources:
      haveasip:
        path: ../corpus/raw/haveasip
        glob: "*.m4a"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SourceConfig:
    name: str
    path: Path
    glob: str = "**/*"  # recursive by default; hidden files are skipped
    language: str = "vi"

    @property
    def root(self) -> Path:
        return self.path.resolve()

    def files(self) -> list[Path]:
        """Absolute paths of matching files, skipping hidden ones (.DS_Store, AppleDouble ._*)."""
        return sorted(
            p
            for p in self.root.glob(self.glob)
            if p.is_file() and not any(part.startswith(".") for part in p.relative_to(self.root).parts)
        )


@dataclass(frozen=True)
class Config:
    sources: dict[str, SourceConfig]

    @classmethod
    def load(cls, path: str | Path) -> Config:
        path = Path(path)
        raw = yaml.safe_load(path.read_text())
        return cls(
            sources={
                name: SourceConfig(name=name, **{**spec, "path": path.parent / spec["path"]})
                for name, spec in raw["sources"].items()
            }
        )
