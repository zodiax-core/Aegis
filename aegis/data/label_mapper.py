"""source label -> canonical AEGIS label. Nothing is merged implicitly: every mapping lives in YAML."""
from __future__ import annotations

from typing import Dict, List, Optional

import yaml


class UnmappedLabelError(KeyError):
    pass


class LabelMapper:
    def __init__(self, spec: dict):
        self.spec = spec
        self.canonical: List[str] = list(spec["canonical_labels"])
        self.normal_label: str = spec.get("normal_label", "NORMAL")
        self.on_unmapped: str = spec.get("on_unmapped", "error")
        self._maps: Dict[str, Dict[str, str]] = {}
        for ds, body in spec.get("datasets", {}).items():
            m = {k.lower(): v for k, v in body.get("folder_labels", {}).items()}
            for k, v in m.items():
                if v not in self.canonical:
                    raise ValueError(f"{ds}: '{k}' maps to '{v}' which is not a canonical label")
            self._maps[ds] = m
        if self.normal_label not in self.canonical:
            raise ValueError("normal_label must be one of canonical_labels")

    @classmethod
    def from_yaml(cls, path: str) -> "LabelMapper":
        with open(path) as f:
            return cls(yaml.safe_load(f))

    def datasets(self) -> List[str]:
        return list(self._maps)

    def has_dataset(self, ds: str) -> bool:
        return ds in self._maps

    def map(self, dataset: str, source_label: str) -> Optional[str]:
        """Canonical label, or None when policy is 'skip'. Raises when policy is 'error'."""
        if dataset not in self._maps:
            raise UnmappedLabelError(f"No mapping for dataset '{dataset}' in label_mapping.yaml")
        lab = self._maps[dataset].get(source_label.lower())
        if lab is None:
            if self.on_unmapped == "skip":
                return None
            raise UnmappedLabelError(f"{dataset}: label '{source_label}' has no explicit mapping")
        return lab

    def source_labels(self, dataset: str) -> List[str]:
        return list(self._maps.get(dataset, {}))

    def class_list(self, present: List[str]) -> List[str]:
        """Model classes = canonical labels that actually occur, NORMAL first, canonical order otherwise."""
        rest = [c for c in self.canonical if c in set(present) and c != self.normal_label]
        return [self.normal_label] + rest
