# Run from project root after installing runpod/requirements.txt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runpod"))
from inference import classify_file

def test_audio(path):
    result = classify_file(path)
    assert result["clips"] >= 1
    assert set(result["percentages"]) == {"active_colony", "inactive_hive", "swarming", "quacking_queen_bee"}
    print(result)

if __name__ == "__main__":
    import sys
    test_audio(sys.argv[1])
