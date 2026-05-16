# Mobile VisionLM
This repository covers the end-to-end pipeline for converting, quantizing, and deploying the Imp Vision Language Model on Android for fully on-device inference — no internet connection required at runtime.

<img src="https://github.com/NSTiwari/Mobile-VisionLM/blob/main/assets/mlc_mobile_vision_lm.png"/>

## What this project does

Most VLM demos run in the cloud. This one doesn't. The model runs entirely on the Android device using OpenCL for GPU acceleration, which means it works offline and doesn't send your images anywhere.

Getting there involves a non-trivial compilation pipeline:

- **TVM** is built from source with LLVM and CUDA support — this is the compiler backend that turns the model graph into optimized device code
- **MLC-LLM** is also built from source on top of TVM, configured for OpenCL (the GPU compute API available on Android)
- The model weights are converted to MLC's internal format and quantized to `q4f16` (4-bit weights, float16 compute) to fit within mobile memory constraints
- The whole thing is then cross-compiled for `aarch64-linux-android` using the Android NDK and packaged as a `.tar` that the Android library loads at runtime

The CMake configuration is intentionally specific — CUTLASS, CUBLAS, and FlashInfer are disabled because they're CUDA-only and don't work on Android's OpenCL backend. Getting this combination right took a fair amount of trial and error.

## Results

<img src="https://github.com/NSTiwari/Mobile-VisionLM/blob/main/assets/mobile_vision_lm.gif"/>

## How to run

### Requirements

- Linux (Ubuntu 22.04 recommended)
- CUDA-capable GPU with at least 16 GB VRAM (for the compilation step)
- ~50 GB free disk space (TVM build artifacts are large)
- JDK 17, Android NDK 26.1.10909125 (installed automatically by the script)
- Rust/Cargo (installed automatically)
- A Hugging Face account with an access token

### Install Python dependencies

```bash
pip install -r requirements.txt
```

### Set your Hugging Face token

```bash
export HF_TOKEN=hf_your_token_here
```

### Run the pipeline

```bash
python mobile_vision_lm.py --hf_username your_huggingface_username
```

This will handle everything: cloning repos, building TVM (~2 hours), building MLC-LLM (~15 minutes), downloading weights, converting and compiling the model, setting up the Android SDK/NDK, and packaging the app.

If TVM and MLC-LLM are already built, you can skip that step:

```bash
python mobile_vision_lm.py --hf_username your_huggingface_username --skip_build
```

The final output is a `mobile-vision-lm-android.zip` containing the Android project ready to open in Android Studio.

### Open in Android Studio

1. Unzip `mobile-vision-lm-android.zip`
2. Open the `Android_App` folder in Android Studio
3. Connect an Android device (API 26+ recommended)
4. Build and run

The app pulls the compiled model weights from Hugging Face on first launch, then caches them locally for offline use.

## Why building from source

Pre-built TVM and MLC-LLM wheels on PyPI don't include the OpenCL backend or the Android cross-compilation target. Building from source is the only way to get a binary that can actually compile models for Android. The `config.cmake` flags in the script are what make the difference — wrong flags and the compiled model either crashes or falls back to CPU.

## References
* MLC-Imp repository: [https://github.com/MILVLG/mlc-imp](https://github.com/MILVLG/mlc-imp)
* Imp v1.5 3B model: [https://huggingface.co/MILVLG/Imp-v1.5-3B-196](https://huggingface.co/MILVLG/Imp-v1.5-3B-196)
* Converted model weights: [https://huggingface.co/NSTiwari/imp-v1.5-3B-196-q4f16_1-MLC-android](https://huggingface.co/NSTiwari/imp-v1.5-3B-196-q4f16_1-MLC-android)
* Machine Learning Compiler (MLC.ai): [https://mlc.ai/](https://mlc.ai/)

# Acknowledgment
<img src="https://github.com/NSTiwari/Mobile-VisionLM/blob/main/assets/dev-logo.png">

This project was developed as part of Google's AI Developer Programs AI Sprint. Thanks to the AIDP Team for their generous support in providing GCP credits and Colab units to help facilitate this project.
