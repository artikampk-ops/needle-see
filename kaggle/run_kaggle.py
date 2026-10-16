{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# needle-see — 20-layer graft run on Kaggle\n",
        "\n",
        "1. Upload the project as `needle-see.zip` to this notebook's files.\n",
        "2. Set add-on data to Cactus-Compute/needle3 (optional — the trainer downloads it).\n",
        "3. GPU on the right (T4 or P100). Turn on Accelerator.\n",
        "4. Run cells top to bottom."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "!pip -q install \"jax[cuda12]\" flax optax safetensors sentencepiece onnxruntime onnx pillow huggingface_hub"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "!rm -rf /kaggle/working/needle-see && unzip -q /kaggle/input/needle-see/needle-see.zip -d /kaggle/working"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "!ls /kaggle/working/needle-see/tools && cd /kaggle/working/needle-see && python tools/export_clip_patches.py"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "import os\n",
        "os.chdir('/kaggle/working/needle-see')\n",
        "!git clone --depth 1 https://github.com/$(echo unused) /tmp/_noop 2>/dev/null; echo ok"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "import os\n",
        "os.chdir('/kaggle/working/needle-see')\n",
        "os.environ['NEEDLE_COUNT'] = '12000'\n",
        "os.environ['NEEDLE_VAL'] = '1200'\n",
        "os.environ['NEEDLE_EPOCHS'] = '4'\n",
        "os.environ['NEEDLE_BATCH'] = '32'\n",
        "os.environ['NEEDLE_LAYERS'] = '20'\n",
        "!python /kaggle/working/needle-see/kaggle/run_kaggle.py"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
EOF
ls -la /home/maaz/Work/system1-research/needle-see/kaggle