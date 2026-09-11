# AMD RX 7900 XTX 24GB Backend

Target:

- AMD Radeon RX 7900 XTX 24GB
- RDNA 3 / `gfx1100`
- HIP/ROCm direct compilation
- BF16 model storage
- FP32 accumulation
- Explicit device kernels for model and codec operators

Supported model families:

- 0.6B Base
- 0.6B CustomVoice
- 1.7B Base
- 1.7B CustomVoice
- 1.7B VoiceDesign

Resident execution uses 28 WGPs for generation and 20 WGPs for codec decoding.

Streaming contract:

- first audio packet: 8 codec frames / 640 ms of audio
- steady packet: 16 codec frames / 1280 ms of audio
- `--max-new-tokens` is required
- request range: 1..8192

## ROCm runtime lookup

The HIP executable embeds a `RUNPATH` to the selected ROCm installation's
runtime library directory. CMake looks for `libamdhip64.so` in `${ROCM_PATH}/lib`
first, then `${ROCM_PATH}/lib64`, and fails configuration if neither exists.
This ports the locally verified ROCm 10 runtime-lookup fix without changing
the device kernels or resident WGP partitioning. CUDA builds are unchanged.

For the ROCm 10 installation, configure and rebuild from the repository root:

```bash
cmake -S . -B build/rx7900xtx-24g-1.7b \
  -DQINGMING_DEVICE=rx7900xtx-24g \
  -DQINGMING_MODEL_FAMILY=1.7b \
  -DROCM_PATH=/opt/rocm-10.0.0 &&
cmake --build build/rx7900xtx-24g-1.7b --clean-first -j
```

`--clean-first` ensures an existing build executes the updated link command,
without deleting its CMake configuration. Check the resulting binary without
a shell-provided library path:

```bash
BIN=./build/rx7900xtx-24g-1.7b/qingming-qwen3-tts_rx7900xtx-24g_1.7b
env -u LD_LIBRARY_PATH readelf -d "$BIN" | grep -E 'NEEDED|RPATH|RUNPATH'
env -u LD_LIBRARY_PATH ldd "$BIN" | grep -E 'amdhip|not found'
```

The HIP runtime should resolve under the selected ROCm installation, with no
`not found` entries. This is a dynamic-linking check, not an inference test.

Build-configuration regression tests need Python, CMake and Ninja, but no GPU
or ROCm installation:

```bash
python3 -m unittest discover -s tests -p 'test_rocm_runtime_path.py' -v
```

The tests use temporary placeholder libraries and inspect generated commands;
they do not compile a HIP binary. If CMake or Ninja is not on `PATH`, set
`QINGMING_TEST_CMAKE` or `QINGMING_TEST_NINJA` to the corresponding executable.
