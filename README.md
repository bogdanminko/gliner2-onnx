# gliner2-onnx

GLiNER2 ONNX runtime for Python. Runs GLiNER2 models without PyTorch.

This library is experimental. The API may change between versions.

## Features

- Zero-shot NER and text classification
- Runs with ONNX Runtime (no PyTorch dependency)
- FP32, FP16, and INT8 precision support
- GPU acceleration via CUDA
- Batch inference for high-throughput processing
- Multi-task schema API — NER + classification in a single encoder pass

## Installation

```bash
pip install gliner2-onnx
```

## API overview

Every method has a single-text and a batch variant with identical semantics:

| Single text | Batch | Returns |
|---|---|---|
| `classify(text, labels)` | `classify_batch(texts, labels)` | `dict[str, float]` |
| `extract_entities(text, labels)` | `extract_entities_batch(texts, labels)` | `list[Entity]` |
| `extract(text, schema)` | `extract_batch(texts, schema)` | `ExtractionResult` |

Batch variants accept an optional `batch_size` argument (default `8`) controlling how many texts are encoded per forward pass.

## NER

```python
from gliner2_onnx import GLiNER2ONNXRuntime

runtime = GLiNER2ONNXRuntime.from_pretrained("lmo3/gliner2-multi-v1-onnx")

entities = runtime.extract_entities(
    "John works at Google in Seattle",
    ["person", "organization", "location"],
)
# [
#   Entity(text='John', label='person', start=0, end=4, score=0.98),
#   Entity(text='Google', label='organization', start=14, end=20, score=0.97),
#   Entity(text='Seattle', label='location', start=24, end=31, score=0.96),
# ]

# Batch
results = runtime.extract_entities_batch(
    ["John works at Google", "Paris is in France"],
    ["person", "organization", "location"],
)
```

## Classification

```python
# Single-label
result = runtime.classify("Buy milk from the store", ["shopping", "work", "entertainment"])
# {'shopping': 0.95}

# Multi-label
result = runtime.classify(
    "Buy milk and finish the report",
    ["shopping", "work", "entertainment"],
    threshold=0.3,
    multi_label=True,
)
# {'shopping': 0.85, 'work': 0.72}

# Batch
results = runtime.classify_batch(
    ["Buy milk", "Write the report", "Watch a movie"],
    ["shopping", "work", "entertainment"],
)
```

## Schema — multi-task extraction

Run NER and multiple classification tasks in a **single encoder forward pass**:

```python
from gliner2_onnx import GLiNER2ONNXRuntime, Schema

runtime = GLiNER2ONNXRuntime.from_pretrained("lmo3/gliner2-multi-v1-onnx")

schema = (
    Schema()
    .entities(["person", "organization", "location"], threshold=0.5)
    .classification("safety", ["safe", "unsafe"])
    .classification("intent", ["informational", "adversarial", "instructional"])
)

result = runtime.extract("John Smith works at Google in New York.", schema)

result.entities
# [Entity(text='John Smith', label='person', ...), ...]

result.classifications
# {
#   'safety': {'safe': 0.91},
#   'intent': {'informational': 0.87},
# }

# Batch
results = runtime.extract_batch(["text one", "text two"], schema)
```

`Schema` is immutable — each `.entities()` / `.classification()` call returns a new instance, so schemas can be reused and composed safely.

## CUDA

```python
runtime = GLiNER2ONNXRuntime.from_pretrained(
    "lmo3/gliner2-multi-v1-onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
```

## Precision

```python
# FP16
runtime = GLiNER2ONNXRuntime.from_pretrained("lmo3/gliner2-multi-v1-onnx", precision="fp16")

# INT8 (dynamic quantization, local export only — see Exporting Models)
runtime = GLiNER2ONNXRuntime("./model_out/gliner2-multi-v1", precision="int8")
```

## Session options

Fine-tune ONNX Runtime behaviour via `ONNXSessionOptions`:

```python
from gliner2_onnx import GLiNER2ONNXRuntime, ONNXSessionOptions

runtime = GLiNER2ONNXRuntime.from_pretrained(
    "lmo3/gliner2-multi-v1-onnx",
    session_options=ONNXSessionOptions(
        intra_op_num_threads=4,
        inter_op_num_threads=1,
    ),
)
```

## Models

Pre-exported ONNX models:

| Model | HuggingFace |
|-------|-------------|
| gliner2-large-v1 | [lmo3/gliner2-large-v1-onnx](https://huggingface.co/lmo3/gliner2-large-v1-onnx) |
| gliner2-multi-v1 | [lmo3/gliner2-multi-v1-onnx](https://huggingface.co/lmo3/gliner2-multi-v1-onnx) |

Note: `gliner2-base-v1` is not supported (uses a different architecture).

## Exporting Models

```bash
git clone https://github.com/lmoe/gliner2-onnx
cd gliner2-onnx

# FP32 only
make onnx-export MODEL=fastino/gliner2-large-v1

# FP32 + FP16
make onnx-export hivetrace/gliner-guard-omni QUANTIZE=fp16

# FP32 + INT8
make onnx-export MODEL=fastino/gliner2-large-v1 QUANTIZE=int8
```

Output is saved to `model_out/<model-name>/`.

## JavaScript/TypeScript

For Node.js, see [@lmoe/gliner-onnx.js](https://github.com/lmoe/gliner-onnx.js).

## Credits

- [fastino-ai/GLiNER2](https://github.com/fastino-ai/GLiNER2) - Original GLiNER2 implementation
- [fastino/gliner2-large-v1](https://huggingface.co/fastino/gliner2-large-v1) - Pre-trained models

## License

MIT