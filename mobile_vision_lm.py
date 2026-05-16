#!/usr/bin/env python3
# End-to-end pipeline to convert and deploy the Imp Vision Language Model on Android.
# Developed by Nitin Tiwari (github.com/NSTiwari)
#
# What this does:
#   1. Builds TVM and MLC-LLM from source
#   2. Downloads Imp-v1.5-3B-196 weights from Hugging Face
#   3. Converts and quantizes the model to MLC-LLM format
#   4. Compiles it for Android (OpenCL target)
#   5. Builds the Android library and packages the app
#
# Usage:
#   python mobile_vision_lm.py --hf_username your_username
#   python mobile_vision_lm.py --hf_username your_username --skip_build  (if TVM/MLC already built)
#
# Install dependencies first:
#   pip install -r requirements.txt
#
# System requirements:
#   - Linux (Ubuntu 22.04 recommended)
#   - CUDA-capable GPU with at least 16 GB VRAM
#   - ~50 GB free disk space (TVM build is large)
#   - JDK 17 (installed automatically if missing)

import os
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from huggingface_hub import snapshot_download, upload_folder, create_repo, whoami


# Paths — everything lives under a single workspace directory
WORKSPACE = "/content"
MLC_IMP_DIR = os.path.join(WORKSPACE, "mlc-imp")
MOBILE_REPO_DIR = os.path.join(WORKSPACE, "Mobile-VisionLM")
TVM_DIR = os.path.join(MLC_IMP_DIR, "3rdparty", "tvm")
DIST_DIR = os.path.join(MLC_IMP_DIR, "dist")
ANDROID_SDK_DIR = os.path.join(WORKSPACE, "android_sdk")

MODEL_NAME = "imp-v1.5-3B-196"
MODEL_TYPE = "imp"
QUANTIZATION = "q4f16_1"
HF_MODEL_REPO = "MILVLG/Imp-v1.5-3B-196"


def run(cmd, cwd=None, env=None):
    """Runs a shell command and raises if it fails."""
    full_env = {**os.environ, **(env or {})}
    subprocess.run(cmd, shell=True, check=True, cwd=cwd, env=full_env)


def step1_clone_repos():
    print("\n--- Cloning repositories ---")
    if not os.path.exists(MLC_IMP_DIR):
        run(f"git clone --recursive https://github.com/MILVLG/mlc-imp.git {MLC_IMP_DIR}")
    else:
        print("mlc-imp already cloned, skipping.")

    if not os.path.exists(MOBILE_REPO_DIR):
        run(f"git clone https://github.com/NSTiwari/Mobile-VisionLM.git {MOBILE_REPO_DIR}")
    else:
        print("Mobile-VisionLM already cloned, skipping.")


def step2_install_system_deps():
    print("\n--- Installing system dependencies ---")
    run("sudo apt install -y llvm")
    run("pip install timm")

    # Install Rust — needed for some MLC-LLM build steps
    run("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
    os.environ["PATH"] += ":/root/.cargo/bin"

    result = subprocess.run("cargo --version", shell=True, capture_output=True, text=True)
    print(f"Cargo installed: {result.stdout.strip()}")


def step3_build_tvm():
    print("\n--- Building TVM from source (this takes ~2 hours) ---")
    build_dir = os.path.join(TVM_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)

    # Copy the default cmake config and add our settings on top
    src_config = os.path.join(TVM_DIR, "cmake", "config.cmake")
    dst_config = os.path.join(build_dir, "config.cmake")
    shutil.copy(src_config, dst_config)

    extra_cmake_flags = [
        'set(CMAKE_BUILD_TYPE RelWithDebInfo)',
        'set(USE_LLVM "llvm-config --ignore-libllvm --link-static")',
        'set(HIDE_PRIVATE_SYMBOLS ON)',
        'set(USE_CUDA ON)',
    ]
    with open(dst_config, "a") as f:
        for flag in extra_cmake_flags:
            f.write(f"\n{flag}")

    run("cmake ..", cwd=build_dir)
    run("cmake --build . --parallel 4", cwd=build_dir)

    # Install TVM as an editable package so Python can import it
    tvm_python_dir = os.path.join(TVM_DIR, "python")
    run("pip install -e .", cwd=tvm_python_dir)

    result = subprocess.run(
        'python -c "import tvm; print(tvm.__file__)"',
        shell=True, capture_output=True, text=True
    )
    print(f"TVM installed at: {result.stdout.strip()}")


def step4_build_mlc_llm():
    print("\n--- Building MLC-LLM from source (~15 minutes) ---")
    build_dir = os.path.join(MLC_IMP_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)

    # Write config.cmake directly instead of using the interactive gen_cmake_config.py script
    config_content = f"""\
set(TVM_HOME {TVM_DIR})
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(USE_CUDA ON)
set(USE_CUTLASS OFF)
set(USE_CUBLAS OFF)
set(USE_ROCM OFF)
set(USE_VULKAN OFF)
set(USE_METAL OFF)
set(USE_OPENCL ON)
set(USE_THRUST ON)
set(USE_FLASHINFER OFF)
"""
    with open(os.path.join(build_dir, "config.cmake"), "w") as f:
        f.write(config_content)

    run("cmake ..", cwd=build_dir)
    run(f"cmake --build . --parallel $(nproc)", cwd=build_dir)

    mlc_python_dir = os.path.join(MLC_IMP_DIR, "python")
    run("pip install -e .", cwd=mlc_python_dir)

    result = subprocess.run(
        'python -c "import mlc_llm; print(mlc_llm)"',
        shell=True, capture_output=True, text=True
    )
    print(f"MLC-LLM installed: {result.stdout.strip()}")


def step5_download_model():
    print(f"\n--- Downloading {MODEL_NAME} weights ---")
    model_dir = os.path.join(DIST_DIR, "models", MODEL_NAME)
    os.makedirs(model_dir, exist_ok=True)

    snapshot_download(repo_id=HF_MODEL_REPO, local_dir=model_dir)
    print(f"Model saved to: {model_dir}")
    return model_dir


def step6_convert_weights(model_dir):
    print("\n--- Converting model weights to MLC format ---")
    libs_dir = os.path.join(DIST_DIR, "libs")

    run(
        f"mlc_llm convert_weight "
        f"--model-type {MODEL_TYPE} "
        f"{model_dir}/ "
        f"--quantization {QUANTIZATION} "
        f"-o {libs_dir}"
    )
    return libs_dir


def step7_generate_config(model_dir, libs_dir):
    print("\n--- Generating MLC config files ---")
    run(
        f"mlc_llm gen_config "
        f"{model_dir} "
        f"--quantization {QUANTIZATION} "
        f"--conv-template {MODEL_TYPE} "
        f"-o {libs_dir}"
    )


def step8_compile_for_android(libs_dir):
    print("\n--- Compiling model for Android (OpenCL target) ---")
    output_tar = os.path.join(libs_dir, f"{MODEL_NAME}-{QUANTIZATION}-android.tar")
    run(
        f"mlc_llm compile "
        f"{os.path.join(libs_dir, 'mlc-chat-config.json')} "
        f"--device android "
        f"-o {output_tar}"
    )
    return output_tar


def step9_upload_to_huggingface(libs_dir, hf_username):
    print("\n--- Uploading to Hugging Face ---")
    repo_name = f"{MODEL_NAME}-{QUANTIZATION}-MLC-android"
    repo_id = f"{hf_username}/{repo_name}"

    repo_id = create_repo(repo_id, exist_ok=True).repo_id
    upload_folder(
        repo_id=repo_id,
        folder_path=libs_dir,
        commit_message=f"{MODEL_NAME}-{QUANTIZATION}-MLC",
        ignore_patterns=["step_*", "epoch_*"],
    )
    hf_url = f"https://huggingface.co/{repo_id}"
    print(f"Uploaded to: {hf_url}")
    return hf_url


def step10_setup_android_sdk():
    print("\n--- Setting up Android SDK and NDK ---")

    # Download Android command-line tools
    tools_zip = os.path.join(WORKSPACE, "commandlinetools-linux-9123335_latest.zip")
    if not os.path.exists(tools_zip):
        run(
            f"curl -O https://dl.google.com/android/repository/commandlinetools-linux-9123335_latest.zip",
            cwd=WORKSPACE
        )

    run("sudo apt install -y unzip")
    run(f"unzip -o {tools_zip}", cwd=WORKSPACE)

    sdk_tools_dir = os.path.join(ANDROID_SDK_DIR, "cmdline-tools", "latest")
    os.makedirs(os.path.dirname(sdk_tools_dir), exist_ok=True)

    extracted = os.path.join(WORKSPACE, "cmdline-tools")
    if os.path.exists(extracted):
        shutil.move(extracted, sdk_tools_dir)

    # Install JDK 17
    run("sudo apt update && sudo apt install -y openjdk-17-jdk")

    # Set environment variables for Android toolchain
    ndk_version = "26.1.10909125"
    os.environ["ANDROID_HOME"] = ANDROID_SDK_DIR
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"
    os.environ["TVM_HOME"] = TVM_DIR
    os.environ["ANDROID_NDK"] = os.path.join(ANDROID_SDK_DIR, "ndk", ndk_version)
    os.environ["TVM_NDK_CC"] = os.path.join(
        os.environ["ANDROID_NDK"],
        "toolchains", "llvm", "prebuilt", "linux-x86_64",
        "bin", "aarch64-linux-android24-clang"
    )

    sdkmanager = os.path.join(sdk_tools_dir, "bin", "sdkmanager")
    run(f"chmod +x {sdkmanager}")

    # Install NDK — accept the license automatically
    run(f'yes | {sdkmanager} "ndk;{ndk_version}"')


def step11_build_android_library(android_tar_path, libs_dir):
    print("\n--- Building Android library ---")

    # The MLC build scripts expect a slightly different filename for the tar
    renamed_tar = os.path.join(libs_dir, f"imp-v1-3b_196-{QUANTIZATION}-android.tar")
    if os.path.exists(android_tar_path) and not os.path.exists(renamed_tar):
        os.rename(android_tar_path, renamed_tar)

    android_lib_dir = os.path.join(MLC_IMP_DIR, "android", "library")
    run("bash prepare_libs.sh", cwd=android_lib_dir)

    # Copy the built library into the Android app project
    build_src = os.path.join(android_lib_dir, "build")
    build_dst = os.path.join(MOBILE_REPO_DIR, "Android_App", "library", "build")
    if os.path.exists(build_dst):
        shutil.rmtree(build_dst)
    shutil.copytree(build_src, build_dst)
    print(f"Build files copied to: {build_dst}")


def step12_update_app_config(hf_model_url):
    print("\n--- Updating app-config.json ---")
    config_path = os.path.join(
        MOBILE_REPO_DIR, "Android_App", "library", "src", "main", "assets", "app-config.json"
    )

    with open(config_path, "r") as f:
        config = json.load(f)

    config["model_list"][0]["model_url"] = hf_model_url

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"model_url set to: {hf_model_url}")


def step13_package_app():
    print("\n--- Packaging the Android app ---")
    output_zip = os.path.join(WORKSPACE, "mobile-vision-lm-android.zip")
    run(f"zip -r {output_zip} {MOBILE_REPO_DIR}", cwd=WORKSPACE)
    print(f"\nDone. Android app package saved to: {output_zip}")
    return output_zip


def main():
    parser = argparse.ArgumentParser(
        description="Build and deploy the Imp Vision LM on Android via MLC-LLM"
    )
    parser.add_argument(
        "--hf_username",
        required=True,
        help="Your Hugging Face username (used to upload the compiled model)",
    )
    parser.add_argument(
        "--workspace",
        default="/content",
        help="Root directory for all cloned repos and build artifacts (default: /content)",
    )
    parser.add_argument(
        "--skip_build",
        action="store_true",
        help="Skip building TVM and MLC-LLM (use if they're already installed)",
    )
    parser.add_argument(
        "--skip_android_setup",
        action="store_true",
        help="Skip Android SDK/NDK setup if it's already done",
    )
    args = parser.parse_args()

    # Update global paths if workspace is different from default
    global WORKSPACE, MLC_IMP_DIR, MOBILE_REPO_DIR, TVM_DIR, DIST_DIR, ANDROID_SDK_DIR
    WORKSPACE = args.workspace
    MLC_IMP_DIR = os.path.join(WORKSPACE, "mlc-imp")
    MOBILE_REPO_DIR = os.path.join(WORKSPACE, "Mobile-VisionLM")
    TVM_DIR = os.path.join(MLC_IMP_DIR, "3rdparty", "tvm")
    DIST_DIR = os.path.join(MLC_IMP_DIR, "dist")
    ANDROID_SDK_DIR = os.path.join(WORKSPACE, "android_sdk")

    if not os.environ.get("HF_TOKEN"):
        raise EnvironmentError(
            "HF_TOKEN is not set. Export it before running:\n"
            "  export HF_TOKEN=hf_..."
        )

    step1_clone_repos()
    step2_install_system_deps()

    if not args.skip_build:
        step3_build_tvm()
        step4_build_mlc_llm()

    model_dir = step5_download_model()
    libs_dir = step6_convert_weights(model_dir)
    step7_generate_config(model_dir, libs_dir)
    android_tar = step8_compile_for_android(libs_dir)
    hf_model_url = step9_upload_to_huggingface(libs_dir, args.hf_username)

    if not args.skip_android_setup:
        step10_setup_android_sdk()

    step11_build_android_library(android_tar, libs_dir)
    step12_update_app_config(hf_model_url)
    step13_package_app()


if __name__ == "__main__":
    main()
