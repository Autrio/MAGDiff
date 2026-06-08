from pathlib import Path

ROOT = Path(__file__).resolve()

while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
    
DATASET_DIR = ROOT / "magdiff/dataset"
GRASP_DIR = DATASET_DIR / "grasps"
MESH_DIR = DATASET_DIR / "meshes"
GRIPPER_MESH_PATH = DATASET_DIR / "gripper.obj"
SCALES_JSON = DATASET_DIR / "scales.json"

MODELS_DIR = ROOT / "magdiff/models"
WORLD_BASE_FILE = MODELS_DIR / "world.xml"

RENDER_DIR = ROOT / "magdiff/renders"