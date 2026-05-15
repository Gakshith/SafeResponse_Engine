from pathlib import Path

from setuptools import find_namespace_packages, setup


def _requirements() -> list[str]:
    requirements_path = Path(__file__).with_name("requirements.txt")
    requirements = []
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("pytest"):
            continue
        requirements.append(line)
    return requirements


setup(
    name="saferesponse-engine",
    version="0.1.0",
    description="LLM hallucination-reduction middleware with retrieval, verification, and routing.",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    install_requires=_requirements(),
    extras_require={"dev": ["pytest"]},
    python_requires=">=3.11",
)
