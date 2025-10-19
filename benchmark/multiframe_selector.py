#!/usr/bin/env python3
"""
Sample selector for multi-frame evaluation.

Selects continuous sequences of frames with consistent triplet configurations.
"""

import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))
from benchmark.benchmark_config import BenchmarkConfig
from benchmark.cholect50_utils import CholecT50Loader

# Import from local module
try:
    from benchmark.multiframe_evaluator import MultiFrameSample
except ImportError:
    # Fallback if running as script
    from multiframe_evaluator import MultiFrameSample


class MultiFrameSelector:
    """Select multi-frame samples for evaluation"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.loader = CholecT50Loader(config.cholect50_root)
        self.available_graphs = self._find_available_graphs()
    
    def _find_available_graphs(self) -> List[tuple]:
        """Find all preprocessed clips with graphs"""
        available = []
        
        for video_dir in self.config.preprocessed_root.glob("video*"):
            video_id = int(video_dir.name.replace("video", ""))
            
            for clip_dir in video_dir.glob("video*_*"):
                clip_name = clip_dir.name
                clip_start = int(clip_name.split("_")[1])
                
                # Check if essential graph files exist
                if (clip_dir / "adjacency_matrices.npy").exists():
                    available.append((video_id, clip_start, clip_dir))
        
        return available
    
    def select_sequences(
        self, 
        num_sequences: int = 5,
        frames_per_sequence: int = 5,
        min_config_length: int = 20
    ) -> List[MultiFrameSample]:
        """
        Select multi-frame sequences for evaluation.
        
        Args:
            num_sequences: Number of sequences to select
            frames_per_sequence: Number of frames in each sequence
            min_config_length: Minimum length of consistent triplet configuration
        """
        samples = []
        
        for video_id, clip_start, graph_path in self.available_graphs:
            if len(samples) >= num_sequences:
                break
            
            try:
                video_data = self.loader.load_video_annotations(video_id)
            except FileNotFoundError:
                continue
            
            # Find continuous sequences with consistent triplet config
            sequences = self.loader.find_continuous_triplet_sequences(
                video_id, 
                min_sequence_length=min_config_length
            )
            
            if not sequences:
                continue
            
            # Use sequences from this video
            for seq in sequences[:num_sequences - len(samples)]:
                # Select frames_per_sequence evenly spaced frames from this sequence
                seq_frames = list(range(seq['start_frame'], seq['end_frame'] + 1))
                
                if len(seq_frames) < frames_per_sequence:
                    continue
                
                # Sample evenly
                step = len(seq_frames) // frames_per_sequence
                selected_frames = seq_frames[::step][:frames_per_sequence]
                
                # Find image paths
                video_dir = self.config.preprocessed_root / f"video{video_id:02d}"
                clip_dirs = list(video_dir.glob(f"video{video_id:02d}_{clip_start:05d}*"))
                
                if not clip_dirs:
                    continue
                
                clip_dir = clip_dirs[0]
                
                # Get image paths for selected frames
                image_paths = []
                for frame_num in selected_frames:
                    # Try different image naming patterns
                    relative_frame = frame_num - clip_start
                    
                    # Pattern 1: direct frame number
                    img_path = clip_dir / f"{frame_num:06d}.png"
                    if not img_path.exists():
                        # Pattern 2: relative to clip start
                        img_path = clip_dir / f"{relative_frame:06d}.png"
                    if not img_path.exists():
                        # Pattern 3: with prefix
                        img_path = clip_dir / f"frame_{frame_num:06d}.png"
                    
                    if img_path.exists():
                        image_paths.append(img_path)
                
                if len(image_paths) != frames_per_sequence:
                    if self.config.verbose:
                        print(f"Warning: Could not find all images for {seq['start_frame']}-{seq['end_frame']}")
                    continue
                
                # Get ground truth triplets
                triplets = self.loader.get_frame_triplets(video_data, seq['start_frame'])
                
                if not triplets:
                    continue
                
                sample = MultiFrameSample(
                    video_id=video_id,
                    start_frame=selected_frames[0],
                    end_frame=selected_frames[-1],
                    clip_start=clip_start,
                    image_paths=image_paths,
                    graph_path=graph_path,
                    gt_triplets=triplets,
                    gt_phase=triplets[0]['phase'] if triplets else None
                )
                
                samples.append(sample)
                
                if self.config.verbose:
                    print(f"  Selected sequence: {sample.sample_id}")
                    print(f"    Frames: {selected_frames}")
                    print(f"    Triplets: {[t['triplet_name'] for t in triplets]}")
                
                if len(samples) >= num_sequences:
                    break
        
        return samples
    
    def print_summary(self, samples: List[MultiFrameSample]):
        """Print summary of selected samples"""
        print("\n" + "="*80)
        print("SELECTED MULTI-FRAME SAMPLES")
        print("="*80)
        print(f"\nTotal samples: {len(samples)}")
        print(f"Frames per sample: {samples[0].num_frames if samples else 0}")
        print()
        
        for i, sample in enumerate(samples, 1):
            print(f"{i}. {sample.sample_id}")
            print(f"   Video: {sample.video_id}, Frames: {sample.start_frame}-{sample.end_frame}")
            print(f"   Triplets: {[t['triplet_name'] for t in sample.gt_triplets]}")
            print(f"   Graph: {'Yes' if sample.graph_path else 'No'}")
        
        print("="*80)

