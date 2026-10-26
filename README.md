# Bee audio RunPod worker — preparation package

This is an **un-deployed starter package** based on the user's working Kaggle M7 inference logic. The trained model is included. No external service was modified.

## Next checks BEFORE publishing
1. Compare feature extraction output against the training notebook on the same audio (including versions).
2. Run `python tests/test_inference.py /path/to/recording.wav` from the project root after `pip install -r runpod/requirements.txt`.
3. Compare clip-level outputs against Kaggle M7 on the exact same file.
4. Pin dependency versions to the tested versions, especially XGBoost and librosa; joblib/pickle compatibility matters.
5. Build locally: `docker build -f runpod/Dockerfile -t bee-audio-worker .`
6. For RunPod URL requests set `AUDIO_ALLOWED_HOSTS` to a comma-separated list of trusted file storage hostnames; use signed URLs, not public permanent links.
7. Deploy only after user approval. CPU vs GPU selection requires benchmarking.

The worker expects `{ "input": { "audio_url": "https://trusted-host/path/audio.wav" } }` and returns the four deployed class percentages plus a clip timeline. The two excluded classes are not returned as predictions.

**Known limitation:** Audio is decoded into memory, so bound recording length as well as file size before production. This prototype allows only trusted HTTPS hosts and disallows redirects; harden DNS/IP checks before any internet-facing deployment. The pinned dependency versions and exact preprocessing parity are not yet validated.
