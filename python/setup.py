"""Wheel-tag control. Everything declarative lives in ``pyproject.toml``.

Two things cannot be said declaratively:

1. **This wheel is platform-specific iff it carries a cdylib.** A wheel with a staged
   ``_lib/onnxruntime_vulkan_ep.dll`` must not be tagged ``any`` -- pip would happily
   install a Windows build on Linux. A wheel built with no artifact staged (the pure-shim
   case, where the user supplies ``$ONNXRUNTIME_VULKAN_EP_LIB``) is genuinely portable and
   must not claim otherwise. So the tag is derived from what is actually on disk at build
   time, not from a flag someone remembered to pass.
2. **The wheel is not a CPython extension.** It carries a plain shared library loaded by
   ORT, not by CPython, so the tag is ``py3-none-<platform>`` and not ``cp312-cp312-...``.
   Tagging it for one interpreter would be a false constraint.
"""

from __future__ import annotations

from pathlib import Path

from setuptools import setup
from setuptools.dist import Distribution

try:  # setuptools >= 70 vendors it; older installs get it from `wheel`
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:  # pragma: no cover - depends on the build environment
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel  # type: ignore[no-redef]

_BUNDLE = Path(__file__).parent / "src" / "onnxruntime_ep_vulkan" / "_lib"
_HAS_ARTIFACT = _BUNDLE.is_dir() and any(
    p.suffix in {".dll", ".so", ".dylib"} or ".so." in p.name for p in _BUNDLE.iterdir()
)


class _Distribution(Distribution):
    def has_ext_modules(self) -> bool:  # noqa: D102
        return _HAS_ARTIFACT


class _BdistWheel(_bdist_wheel):
    def finalize_options(self) -> None:  # noqa: D102
        super().finalize_options()
        self.root_is_pure = not _HAS_ARTIFACT

    def get_tag(self) -> tuple[str, str, str]:  # noqa: D102
        _py, _abi, plat = super().get_tag()
        return "py3", "none", plat


setup(distclass=_Distribution, cmdclass={"bdist_wheel": _BdistWheel})
