import importlib.util
from pathlib import Path
import sys

# create dummy modules before loading utils to avoid dependency errors
sys.modules.setdefault('numpy', type('Dummy', (), {'ndarray': object}))
sys.modules.setdefault('tiktoken', type('Dummy', (), {}))

spec = importlib.util.spec_from_file_location(
    "utils", Path(__file__).resolve().parents[1] / "nanographrag_tmp" / "_utils.py"
)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


def test_check_and_fix_json():
    broken = '{"a":1,}'
    fixed = utils.check_and_fix_json(broken)
    assert fixed.endswith('}')
