"""SWE-Bench Verified — läuft via sweagent-Harness (Docker nötig).

Dauer: 6-10h. Orchestriert sweagent gegen unsere LM-Studio-URL.
"""
import os, subprocess, shutil
from pathlib import Path
from _common import BASE_URL, MODEL_ID, save

SWE_AGENT_REPO = "https://github.com/princeton-nlp/SWE-agent.git"
WORK = Path(os.environ.get("SWE_WORK", Path.home() / ".cache" / "swe-agent")).resolve()


def _ensure_sweagent() -> Path:
    if not WORK.exists():
        WORK.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", SWE_AGENT_REPO, str(WORK)], check=True)
        subprocess.run(["pip", "install", "-e", "."], cwd=WORK, check=True)
    return WORK


def main():
    if not shutil.which("docker"):
        raise RuntimeError("Docker is required for SWE-Bench Verified.")

    root = _ensure_sweagent()
    env = {
        **os.environ,
        "LITELLM_API_BASE": BASE_URL,
        "LITELLM_MODEL": f"openai/{MODEL_ID}",
        "OPENAI_API_KEY": "lm-studio",
    }
    cmd = [
        "sweagent", "run-batch",
        "--agent.model.name", f"openai/{MODEL_ID}",
        "--agent.model.api_base", BASE_URL,
        "--instances.type", "swe_bench",
        "--instances.subset", "verified",
        "--instances.slice", ":500",
        "--output_dir", str(root / "trajectories"),
    ]
    print("running sweagent …")
    subprocess.run(cmd, cwd=root, env=env, check=True)

    report = root / "trajectories" / "results.json"
    save("swebench_verified", {"model": MODEL_ID, "report_path": str(report)})
    print(f"Results: {report}")


if __name__ == "__main__":
    main()
