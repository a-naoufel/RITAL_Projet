mkdir -p /tempory/.home/
cd /tempory/.home/
git clone https://github.com/a-naoufel/RITAL_Projet.git
cd /tempory/.home/RITAL_Projet


python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install --upgrade pip setuptools wheel

mkdir -p /tempory/$USER/pip-cache /tempory/$USER/tmp
export PIP_CACHE_DIR=/tempory/$USER/pip-cache
export TMPDIR=/tempory/$USER/tmp

pip uninstall -y torch torchvision torchaudio

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129

pip install --no-cache-dir -r requirements.txt

python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
