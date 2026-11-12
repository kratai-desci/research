"""Progress of the feedback sense's build, for the dashboard: one JSON file, also copied to the results bucket
(sense/status.json by default) at most every 30 seconds when SENSE_BUCKET is set (run_sense.sh sets it)."""

import json
import os
import subprocess
import time
from pathlib import Path

PATH = Path(__file__).parent.parent / "data" / "sense" / "status.json"


class Status:
    def __init__(self, path: Path = PATH, remote: str = "sense/status.json"):
        self.path, self.remote, self.last_upload = path, remote, 0.0
        self.state = json.loads(path.read_text()) if path.exists() else {"started": time.time()}

    def update(self, force: bool = False, **parts) -> None:
        """Merge parts into the status (dicts are merged one level deep) and save it."""
        for key, value in parts.items():
            if isinstance(value, dict) and isinstance(self.state.get(key), dict):
                self.state[key] = {**self.state[key], **value}
            else:
                self.state[key] = value
        self.state["updated"] = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state))
        tmp.replace(self.path)
        bucket = os.environ.get("SENSE_BUCKET")
        if bucket and (force or time.time() - self.last_upload > 30):
            subprocess.run(["gcloud", "storage", "cp", "--quiet", str(self.path), f"{bucket}/{self.remote}"],
                           capture_output=True)
            self.last_upload = time.time()
