"""RunPod serverless handler. Accepts a temporary HTTPS audio URL."""
import os
import tempfile
from urllib.parse import urlparse
import requests
import runpod
from inference import classify_file

MAX_BYTES = 50 * 1024 * 1024

def handler(job):
    url = job.get("input", {}).get("audio_url")
    if not isinstance(url, str):
        return {"error": "audio_url is required"}
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        return {"error": "A valid HTTPS audio_url is required"}
    # SECURITY: Before exposing this endpoint, add an allowlist of trusted
    # storage hostnames to prevent server-side request forgery (SSRF).
    allowed = [x.strip().lower() for x in os.getenv("AUDIO_ALLOWED_HOSTS", "").split(",") if x.strip()]
    if not allowed or parsed.hostname.lower() not in allowed:
        return {"error": "Audio URL host is not allowlisted"}
    try:
        with requests.get(url, stream=True, timeout=(10, 120), allow_redirects=False) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                count = 0
                for chunk in response.iter_content(chunk_size=65536):
                    count += len(chunk)
                    if count > MAX_BYTES:
                        raise ValueError("Audio file exceeds 50 MB")
                    tmp.write(chunk)
                tmp.flush()
                return classify_file(tmp.name)
    except Exception as exc:
        return {"error": str(exc)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
