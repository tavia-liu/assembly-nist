#!/bin/bash
cd /engrfs/project/chongjie/buzhao/autodeco/assembly-nist
/engrfs/project/chongjie/buzhao/miniconda3/bin/uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu124 "torch==2.6.0" "torchvision==0.21.0" 2>&1 | tail -n 3
.venv/bin/python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
.venv/bin/python -c "import torchvision; torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1); print('resnet18 weights cached')"
