from typing import Dict, List, Tuple
import torch.nn as nn
from helpers import _count_from_tensor, _extract_first_tensor, TensorLike

class ExitCounter:
    def __init__(self, model: nn.Module, exit_snippets: List[str], conf_thr: float = 0.25):
        self.model = model
        self.exit_snippets = [s.lower() for s in exit_snippets]
        self.conf_thr = conf_thr
        self.per_image_counts: List[Dict[str, int]] = []
        self.totals: Dict[str, int] = {}
        self._handles = []
        self._last_counts: Dict[str, int] = {}

    def _is_exit_name(self, name: str) -> bool:
        n = name.lower()
        return any(snippet in n for snippet in self.exit_snippets)

    def _hook(self, name: str):
        def fwd_hook(mod: nn.Module, inputs: Tuple, output: TensorLike):
            t = _extract_first_tensor(output)
            cnt = _count_from_tensor(t, self.conf_thr) if t is not None else None
            if cnt is None:
                # couldn't parse; record zero so the key exists
                cnt = 0
            self._last_counts[name] = self._last_counts.get(name, 0) + int(cnt)
        return fwd_hook

    def attach(self):
        for name, m in self.model.named_modules():
            if name and self._is_exit_name(name):
                h = m.register_forward_hook(self._hook(name))
                self._handles.append(h)

    def begin_image(self):
        self._last_counts = {}

    def end_image(self):
        # snapshot for this image (ensure all exits listed, in provided order)
        ordered = {name: self._last_counts.get(name, 0) for name in self._ordered_exit_names()}
        self.per_image_counts.append(ordered)
        for k, v in ordered.items():
            self.totals[k] = self.totals.get(k, 0) + int(v)

    def _ordered_exit_names(self):
        # Order by EXIT_SNIPPETS appearance for stable reporting
        names = set()
        for name, _ in self.model.named_modules():
            if name and self._is_exit_name(name):
                names.add(name)
        # sort by first matching snippet index to keep your intended order
        def keyf(nm):
            nm_l = nm.lower()
            for i, sn in enumerate(self.exit_snippets):
                if sn in nm_l:
                    return (i, nm)
            return (9999, nm)
        return [n for n in sorted(names, key=keyf)]

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()