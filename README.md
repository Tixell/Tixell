---
language:
 - "en"
tags:
 - video
 - genmo
license: apache-2.0
pipeline_tag: text-to-video
library_name: diffusers
---


# Mochi 1
[Blog](https://www.genmo.ai/blog) | [Hugging Face](https://huggingface.co/genmo/mochi-1-preview) | [Playground](https://www.genmo.ai/play) | [Careers](https://jobs.ashbyhq.com/genmo)

A state of the art video generation model by [Genmo](https://genmo.ai).

https://github.com/user-attachments/assets/4d268d02-906d-4cb0-87cc-f467f1497108

## Overview

Mochi 1 preview is an open state-of-the-art video generation model with high-fidelity motion and strong prompt adherence in preliminary evaluation. This model dramatically closes the gap between closed and open video generation systems. We’re releasing the model under a permissive Apache 2.0 license. Try this model for free on [our playground](https://genmo.ai/play).

## Installation

Install using [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/genmoai/models
cd models 
pip install uv
uv venv .venv
source .venv/bin/activate
uv pip install setuptools
uv pip install -e . --no-build-isolation
```

If you want to install flash attention, you can use:
```
uv pip install -e .[flash] --no-build-isolation
```

You will also need to install [FFMPEG](https://www.ffmpeg.org/) to turn your outputs into videos.

## Download Weights

Use [download_weights.py](scripts/download_weights.py) to download the model + decoder to a local directory. Use it like this:
```
python3 ./scripts/download_weights.py <path_to_downloaded_directory>
```

Or, directly download the weights from [Hugging Face](https://huggingface.co/genmo/mochi-1-preview/tree/main) or via `magnet:?xt=urn:btih:441da1af7a16bcaa4f556964f8028d7113d21cbb&dn=weights&tr=udp://tracker.opentrackr.org:1337/announce` to a folder on your computer.

## Running

Start the gradio UI with

```bash
python3 ./demos/gradio_ui.py --model_dir "<path_to_downloaded_directory>"
```

Or generate videos directly from the CLI with

```bash
python3 ./demos/cli.py --model_dir "<path_to_downloaded_directory>"
```

Replace `<path_to_downloaded_directory>` with the path to your model directory.

## API

This repository comes with a simple, composable API, so you can programmatically call the model. This API gives the highest quality results. You can find a full example [here](demos/api_example.py). But, roughly, it looks like this:

```python
from genmo.mochi_preview.pipelines import (
    DecoderModelFactory,
    DitModelFactory,
    MochiSingleGPUPipeline,
    T5ModelFactory,
    linear_quadratic_schedule,
)

pipeline = MochiSingleGPUPipeline(
    text_encoder_factory=T5ModelFactory(),
    dit_factory=DitModelFactory(
        model_path=f"{MOCHI_DIR}/dit.safetensors", model_dtype="bf16"
    ),
    decoder_factory=DecoderModelFactory(
        model_path=f"{MOCHI_DIR}/vae.safetensors",
    ),
    cpu_offload=True,
    decode_type="tiled_full",
)

video = pipeline(
    height=480,
    width=848,
    num_frames=31,
    num_inference_steps=64,
    sigma_schedule=linear_quadratic_schedule(64, 0.025),
    cfg_schedule=[4.5] * 64,
    batch_cfg=False,
    prompt="your favorite prompt here ...",
    negative_prompt="",
    seed=12345,
)
```


## Running with Diffusers

You can also use diffusers.

Install the latest version of Diffusers

```shell
pip install git+https://github.com/huggingface/diffusers.git
```

The following example requires 42GB VRAM but ensures the highest quality output.

```python
import torch
from diffusers import MochiPipeline
from diffusers.utils import export_to_video

pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview")

# Enable memory savings
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()

prompt = "Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k."

with torch.autocast("cuda", torch.bfloat16, cache_enabled=False):
      frames = pipe(prompt, num_frames=84).frames[0]

export_to_video(frames, "mochi.mp4", fps=30)
```

### Using a lower precision variant to save memory

The following example will use the `bfloat16` variant of the model and requires 22GB VRAM to run. There is a slight drop in the quality of the generated video as a result.

```python
import torch
from diffusers import MochiPipeline
from diffusers.utils import export_to_video

pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview", variant="bf16", torch_dtype=torch.bfloat16)

# Enable memory savings
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()

prompt = "Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k."
frames = pipe(prompt, num_frames=84).frames[0]

export_to_video(frames, "mochi.mp4", fps=30)
```

To learn more check out the [Diffusers](https://huggingface.co/docs/diffusers/main/en/api/pipelines/mochi) documentation


## Model Architecture

Mochi 1 represents a significant advancement in open-source video generation, featuring a 10 billion parameter diffusion model built on our novel Asymmetric Diffusion Transformer (AsymmDiT) architecture. Trained entirely from scratch, it is the largest video generative model ever openly released. And best of all, it’s a simple, hackable architecture. Additionally, we are releasing an inference harness that includes an efficient context parallel implementation. 

Alongside Mochi, we are open-sourcing our video AsymmVAE. We use an asymmetric encoder-decoder structure to build an efficient high quality compression model. Our AsymmVAE causally compresses videos to a 128x smaller size, with an 8x8 spatial and a 6x temporal compression to a 12-channel latent space. 

### AsymmVAE Model Specs
|Params <br> Count | Enc Base <br>  Channels | Dec Base <br> Channels |Latent <br> Dim | Spatial <br> Compression | Temporal <br> Compression | 
|:--:|:--:|:--:|:--:|:--:|:--:|
|362M   | 64  | 128  | 12   | 8x8   | 6x   | 

An AsymmDiT efficiently processes user prompts alongside compressed video tokens by streamlining text processing and focusing neural network capacity on visual reasoning. AsymmDiT jointly attends to text and visual tokens with multi-modal self-attention and learns separate MLP layers for each modality, similar to Stable Diffusion 3. However, our visual stream has nearly 4 times as many parameters as the text stream via a larger hidden dimension. To unify the modalities in self-attention, we use non-square QKV and output projection layers. This asymmetric design reduces inference memory requirements.
Many modern diffusion models use multiple pretrained language models to represent user prompts. In contrast, Mochi 1 simply encodes prompts with a single T5-XXL language model.

### AsymmDiT Model Specs
|Params <br> Count | Num <br> Layers | Num <br> Heads | Visual <br> Dim | Text <br> Dim | Visual <br> Tokens | Text <br> Tokens | 
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|10B   | 48   | 24   | 3072   | 1536   | 44520   |   256   |

## Hardware Requirements
The repository supports both multi-GPU operation (splitting the model across multiple graphics cards) and single-GPU operation, though it requires approximately 60GB VRAM when running on a single GPU. While ComfyUI can optimize Mochi to run on less than 20GB VRAM, this implementation prioritizes flexibility over memory efficiency. When using this repository, we recommend using at least 1 H100 GPU.

## Safety
Genmo video models are general text-to-video diffusion models that inherently reflect the biases and preconceptions found in their training data. While steps have been taken to limit NSFW content, organizations should implement additional safety protocols and careful consideration before deploying these model weights in any commercial services or products.

## Limitations
Under the research preview, Mochi 1 is a living and evolving checkpoint. There are a few known limitations. The initial release generates videos at 480p today. In some edge cases with extreme motion, minor warping and distortions can also occur. Mochi 1 is also optimized for photorealistic styles so does not perform well with animated content. We also anticipate that the community will fine-tune the model to suit various aesthetic preferences.

## Related Work
- [ComfyUI-MochiWrapper](https://github.com/kijai/ComfyUI-MochiWrapper) adds ComfyUI support for Mochi. The integration of Pytorch's SDPA attention was taken from their repository.
- [mochi-xdit](https://github.com/xdit-project/mochi-xdit) is a fork of this repository and improve the parallel inference speed with [xDiT](https://github.com/xdit-project/xdit).


## BibTeX
```
@misc{genmo2024mochi,
      title={Mochi 1},
      author={Genmo Team},
      year={2024},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished={\url{https://github.com/genmoai/models}}
}
```