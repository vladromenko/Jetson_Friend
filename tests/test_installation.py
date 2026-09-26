import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_example_paths_follow_checkout_root():
    text = (ROOT / 'config.env.example').read_text()
    assert '/home/vlad/Jetson_Friend' not in text
    assert 'APP_PYTHON="$ROOT/.venv/bin/python"' in text
    assert 'LLM_MODEL="$ROOT/models/' in text


def test_install_does_not_start_robot_or_enable_service():
    text = (ROOT / 'install.sh').read_text()
    assert 'systemctl --user start' not in text
    assert 'systemctl --user enable' not in text
    assert 'systemctl --user disable milo.service' in text


def test_every_model_download_has_sha256():
    text = (ROOT / 'scripts/bootstrap_assets.sh').read_text()
    calls = re.findall(r'^fetch\s+"https://', text, re.MULTILINE)
    hashes = re.findall(r'^\s+"[0-9a-f]{64}"$', text, re.MULTILINE)
    assert len(calls) == 6
    assert len(hashes) == len(calls)


def test_native_inference_uses_jetson_cuda():
    text = (ROOT / 'scripts/bootstrap_assets.sh').read_text()
    assert text.count('-DGGML_CUDA=ON') == 2
    assert 'CMAKE_CUDA_ARCHITECTURES=87' in text
    assert 'CUDA_COMPILER=/usr/local/cuda/bin/nvcc' in text


def test_user_service_targets_canonical_home_checkout():
    text = (ROOT / 'milo.service').read_text()
    assert 'WorkingDirectory=%h/Jetson_Friend' in text
    assert 'WantedBy=default.target' in text
