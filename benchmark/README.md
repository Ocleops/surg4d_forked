# Surgical Action Triplet Recognition Benchmark

Benchmarking system for evaluating vision-language models on surgical action triplet recognition from CholecT50 dataset.

## Features

- **Multi-frame temporal reasoning** using Qwen2.5-VL's video interface
- **3-condition ablation study**: Single frame, Multi-frame, Multi-frame + Graph
- **Triplet configuration analysis** from CholecT50 labels
- **CSV export** of triplet configurations for analysis
- **Exact string matching** for evaluation (no fuzzy matching)

---

## Repository Structure

```
benchmark/
├── README.md                      # This file
├── benchmark_config.py            # Configuration dataclass
├── cholect50_utils.py             # CholecT50 data loading utilities
├── sample_selector.py             # Single-frame sample selection
├── evaluator.py                   # Single-frame evaluator (baseline)
├── multiframe_selector.py         # Multi-frame sample selection
├── multiframe_evaluator.py        # Multi-frame evaluator with ablation
├── run_ablation.py                # Main script for ablation study
├── export_triplet_configs.py      # Export triplet configs to CSV
└── results/                       # Evaluation results (JSON)
```

---

## Quick Start

### 1. Export Triplet Configurations to CSV

Analyze CholecT50 labels and export consistent triplet configurations:

```bash
# Export all videos
python export_triplet_configs.py --all --output triplet_configs.csv

# Export specific videos
python export_triplet_configs.py --video_ids 1 2 3 --output configs.csv

# Options
python export_triplet_configs.py --help
```

**Output CSV columns:**
- `video_id`: Video number (1-50)
- `config_id`: Unique configuration ID
- `start_frame`: First frame of configuration
- `end_frame`: Last frame of configuration
- `num_frames`: Number of consecutive frames
- `triplet_ids`: Comma-separated triplet IDs
- `num_triplets`: Number of simultaneous actions
- `triplet_names`: Human-readable triplet names
- `instruments`, `verbs`, `targets`: Components
- `phase`: Surgical phase

### 2. Run Ablation Study

Run the 3-condition ablation study on multi-frame sequences:

```bash
# Default: 5 sequences, 5 frames each, all 3 conditions
python run_ablation.py

# Custom settings
python run_ablation.py --num_sequences 10 --frames_per_sequence 7

# Specific conditions only
python run_ablation.py --conditions single_frame multiframe

# With 4-bit quantization (for limited GPU memory)
python run_ablation.py --use_4bit

# Options
python run_ablation.py --help
```

**Three conditions:**
1. **Single Frame**: Uses only the middle frame (baseline)
2. **Multi-Frame**: Uses multiple frames for temporal reasoning
3. **Multi-Frame + Graph**: Uses multiple frames + precomputed 4D scene graph

**Output:** JSON file in `results/ablation_TIMESTAMP.json`

### 3. Run Single-Frame Baseline

For comparison with the original single-frame approach:

```bash
python evaluator.py
```

---

## Configuration

Edit `benchmark_config.py` to customize:

```python
@dataclass
class BenchmarkConfig:
    # Data paths
    cholect50_root: Path = Path("/path/to/CholecT50")
    preprocessed_root: Path = Path("/path/to/preprocessed_ssg")
    
    # Model settings
    model_name: Literal["qwen", "gpt4"] = "qwen"
    qwen_version: Literal["qwen2.5", "qwen3"] = "qwen2.5"
    use_4bit_quantization: bool = True
    
    # Evaluation settings
    num_test_frames: int = 5
    exact_match: bool = True  # Use exact string matching
    
    # Scene graph settings
    use_scene_graph: bool = True
```

---

## Ablation Study Results Format

Results are saved as JSON with this structure:

```json
{
  "conditions": {
    "single_frame": {
      "metrics": {
        "instrument_acc": 0.85,
        "verb_acc": 0.75,
        "target_acc": 0.70,
        "triplet_acc": 0.65,
        "num_samples": 5
      },
      "results": [...]
    },
    "multiframe": {
      "metrics": {...},
      "results": [...]
    },
    "multiframe_graph": {
      "metrics": {...},
      "results": [...]
    }
  }
}
```

**Metrics:**
- `instrument_acc`: Instrument recognition accuracy
- `verb_acc`: Verb recognition accuracy
- `target_acc`: Target recognition accuracy
- `triplet_acc`: Full triplet accuracy (all 3 components correct)

---

## Data Requirements

### CholecT50 Labels
Located at: `{cholect50_root}/labels/VID{XX}.json`

Each frame annotation contains:
- Triplet ID (0-93: valid actions, 94-99: instrument present, -1: no action)
- Instrument, verb, target IDs
- Surgical phase

### Preprocessed Data
Located at: `{preprocessed_root}/video{XX}/video{XX}_{FRAME}/`

Each clip contains:
- Image frames
- `qwen_instance_features/*.npy`: Precomputed Qwen features
- `adjacency_matrices.npy`: Spatial relationships
- `centers.npy`, `centroids.npy`, `extents.npy`: 3D positions

---

## Key Concepts

### Triplet Configuration
A **triplet configuration** is a set of frames where the exact same actions are performed. For example:
- Frames 698-745: `{grasper,grasp,gallbladder}` + `{clipper,clip,cystic_duct}`

This ensures consistent ground truth for temporal evaluation.

### Null Triplets (94-99)
These indicate an instrument is **visible** but not performing a recognized action:
- ID 94: grasper present
- ID 95: bipolar present
- ID 96: hook present
- etc.

**By default, these are excluded** from analysis to focus on actual actions.

### Multi-Frame Input
Qwen2.5-VL supports multiple images in a single prompt:

```python
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "file:///path/frame1.png"},
        {"type": "image", "image": "file:///path/frame2.png"},
        {"type": "image", "image": "file:///path/frame3.png"},
        {"type": "text", "text": "What actions are performed?"}
    ]
}]
```

### 4D Scene Graphs
The preprocessed clips contain **4D scene graphs**:
- **3D spatial relationships** (adjacency matrices, 3D positions)
- **Temporal dimension** (N timesteps sampled from the clip)

These are integrated with Qwen features for spatiotemporal reasoning.

---

## Evaluation Protocol

1. **Sample Selection**: 
   - Find continuous sequences with consistent triplet configurations
   - Sample N evenly-spaced frames from each sequence

2. **Prompting**:
   - Provide possible instruments, verbs, targets in prompt
   - Request JSON-formatted response

3. **Evaluation**:
   - Exact string matching (case-insensitive, normalized)
   - For multi-triplet frames: prediction matches ANY ground truth

4. **Metrics**:
   - Component-wise accuracy (I, V, T)
   - Full triplet accuracy (all 3 correct)

---

## Notes

- **Qwen2.5-VL** is used by default. Qwen3 support is planned.
- **4-bit quantization** reduces memory usage (8GB → 4GB VRAM).
- **Exact matching** only - no synonym fuzzy matching.
- Results are saved with timestamps for versioning.

---

## Troubleshooting

### Issue: "Labels not found for video X"
- Ensure CholecT50 labels are at `{cholect50_root}/labels/VID{XX}.json`

### Issue: "No samples selected"
- Check if preprocessed data exists at `{preprocessed_root}/video{XX}/`
- Verify graph files exist: `adjacency_matrices.npy`, etc.

### Issue: "Out of memory"
- Use `--use_4bit` flag for 4-bit quantization
- Reduce `--num_sequences` or `--frames_per_sequence`

### Issue: "KeyError: 'qwen2_5_vl'"
- Update transformers: `pip install git+https://github.com/huggingface/transformers accelerate`

---

## Citation

If you use this benchmark, please cite:

```bibtex
@article{Qwen2.5-VL,
  title={Qwen2.5-VL},
  author={Qwen Team},
  year={2025},
  url={https://qwenlm.github.io/blog/qwen2.5-vl/}
}
```

---

## Contact

For questions about the benchmark system, please open an issue.

