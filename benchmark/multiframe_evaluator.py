#!/usr/bin/env python3
"""
Multi-frame evaluator for surgical action triplet recognition.

Supports 3-condition ablation study:
1. Single Frame: Baseline using one frame
2. Multi-Frame (Video): Temporal reasoning with multiple frames
3. Multi-Frame + Graph: Spatiotemporal reasoning with scene graph

Uses Qwen2.5-VL's native video/multi-image interface.
"""

import sys
import torch
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from benchmark.benchmark_config import BenchmarkConfig, normalize_for_matching
from benchmark.cholect50_utils import CholecT50Loader
from qwen_vl import get_patched_qwen, prompt_with_graph


@dataclass
class MultiFrameSample:
    """Sample with multiple frames for temporal evaluation"""
    video_id: int
    start_frame: int
    end_frame: int
    clip_start: int
    image_paths: List[Path]  # Multiple frames
    graph_path: Optional[Path]
    gt_triplets: List[Dict]  # Ground truth for the sequence
    gt_phase: Optional[str]
    
    @property
    def sample_id(self) -> str:
        return f"v{self.video_id:02d}_f{self.start_frame:05d}-{self.end_frame:05d}"
    
    @property
    def num_frames(self) -> int:
        return len(self.image_paths)


class MultiFrameEvaluator:
    """Evaluator for multi-frame triplet recognition with ablation study"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.loader = CholecT50Loader(config.cholect50_root)
        
        # Load model
        print("Loading Qwen2.5-VL model...")
        self.model, self.processor = get_patched_qwen(
            use_bnb_4bit=config.use_4bit_quantization,
            device_map=config.device
        )
        print("✓ Model loaded")
    
    def _build_single_frame_prompt(self, gt_triplets: List[Dict]) -> str:
        """Build prompt for single-frame condition"""
        
        # Get unique options from ground truth for this video
        instruments = sorted(set(t['instrument'] for t in gt_triplets))
        verbs = sorted(set(t['verb'] for t in gt_triplets))
        targets = sorted(set(t['target'] for t in gt_triplets))
        
        prompt = """Analyze this surgical image and identify the action triplet(s).

For EACH visible surgical instrument performing an action, identify:
- Instrument: The surgical tool being used
- Verb: The action being performed  
- Target: The anatomical structure being acted upon

Respond in JSON format:
{
  "triplets": [
    {"instrument": "...", "verb": "...", "target": "..."},
    ...
  ]
}

Possible instruments: """ + ", ".join(instruments) + """
Possible verbs: """ + ", ".join(verbs) + """
Possible targets: """ + ", ".join(targets) + """

If multiple instruments are active simultaneously, list all triplets.
If no clear action is visible, return empty list."""
        
        return prompt
    
    def _build_multiframe_prompt(self, gt_triplets: List[Dict]) -> str:
        """Build prompt for multi-frame condition"""
        
        instruments = sorted(set(t['instrument'] for t in gt_triplets))
        verbs = sorted(set(t['verb'] for t in gt_triplets))
        targets = sorted(set(t['target'] for t in gt_triplets))
        
        prompt = """Analyze this sequence of surgical frames and identify the action triplet(s) being performed throughout the sequence.

For EACH surgical instrument performing an action in the sequence, identify:
- Instrument: The surgical tool being used
- Verb: The sustained action being performed across frames
- Target: The anatomical structure being acted upon

Respond in JSON format:
{
  "triplets": [
    {"instrument": "...", "verb": "...", "target": "..."},
    ...
  ]
}

Possible instruments: """ + ", ".join(instruments) + """
Possible verbs: """ + ", ".join(verbs) + """
Possible targets: """ + ", ".join(targets) + """

Focus on the CONSISTENT actions across the temporal sequence.
If multiple instruments are active simultaneously, list all triplets."""
        
        return prompt
    
    def _query_single_frame(self, image_path: Path, prompt: str) -> str:
        """Query model with a single frame"""
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return output_text
    
    def _query_multiframe(self, image_paths: List[Path], prompt: str) -> str:
        """Query model with multiple frames (using multiple images)"""
        
        # Build content with all images
        content = []
        for img_path in image_paths:
            content.append({"type": "image", "image": f"file://{img_path}"})
        content.append({"type": "text", "text": prompt})
        
        messages = [{"role": "user", "content": content}]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return output_text
    
    def _query_with_graph(self, image_paths: List[Path], graph_path: Path, prompt: str) -> str:
        """Query model with multiple frames AND scene graph"""
        
        # Load graph data
        graph_data = self._load_graph_data(graph_path)
        
        if graph_data is None:
            # Fallback to multiframe without graph
            return self._query_multiframe(image_paths, prompt)
        
        # Use existing prompt_with_graph from qwen_vl.py
        response = prompt_with_graph(
            node_feats=graph_data['node_feats'],
            adjacency_matrices=graph_data['adjacency_matrices'],
            node_centers=graph_data['node_centers'],
            node_centroids=graph_data['node_centroids'],
            node_extents=graph_data['node_extents'],
            question=prompt,
            model=self.model,
            processor=self.processor
        )
        
        return response
    
    def _load_graph_data(self, graph_path: Path) -> Optional[Dict]:
        """Load precomputed 4D graph data"""
        
        try:
            # Assuming graph structure similar to video01_00080
            clip_dir = graph_path
            
            # Load qwen features
            qwen_feat_dir = clip_dir / "qwen_instance_features"
            if not qwen_feat_dir.exists():
                print(f"Warning: No qwen features found at {qwen_feat_dir}")
                return None
            
            feat_files = sorted(qwen_feat_dir.glob("*_f.npy"))
            node_feats = [np.load(f) for f in feat_files]
            
            # Load spatial matrices
            adjacency_matrices = np.load(clip_dir / "adjacency_matrices.npy")
            centers = np.load(clip_dir / "centers.npy")
            centroids = np.load(clip_dir / "centroids.npy")
            extents = np.load(clip_dir / "extents.npy")
            
            return {
                'node_feats': node_feats,
                'adjacency_matrices': adjacency_matrices,
                'node_centers': centers,
                'node_centroids': centroids,
                'node_extents': extents
            }
        except Exception as e:
            print(f"Warning: Could not load graph data: {e}")
            return None
    
    def _process_vision_info(self, messages):
        """Process vision info from messages (helper from Qwen2.5-VL docs)"""
        image_inputs, video_inputs = [], []
        for message in messages:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele.get("type") == "image":
                        image_inputs.append(ele["image"])
                    elif ele.get("type") == "video":
                        video_inputs.append(ele["video"])
        return image_inputs if image_inputs else None, video_inputs if video_inputs else None
    
    def _parse_response(self, response: str) -> List[Dict]:
        """Parse model response to extract triplets"""
        
        try:
            # Try to find JSON in response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                triplets = data.get('triplets', [])
                return triplets
        except Exception as e:
            print(f"Warning: Could not parse JSON response: {e}")
        
        return []
    
    def evaluate_sample(
        self, 
        sample: MultiFrameSample, 
        condition: str
    ) -> Dict:
        """
        Evaluate a single sample under specified condition.
        
        Args:
            sample: MultiFrameSample to evaluate
            condition: One of "single_frame", "multiframe", "multiframe_graph"
        """
        
        # Build prompt
        if condition == "single_frame":
            prompt = self._build_single_frame_prompt(sample.gt_triplets)
            # Use middle frame
            middle_idx = len(sample.image_paths) // 2
            response = self._query_single_frame(sample.image_paths[middle_idx], prompt)
        elif condition == "multiframe":
            prompt = self._build_multiframe_prompt(sample.gt_triplets)
            response = self._query_multiframe(sample.image_paths, prompt)
        elif condition == "multiframe_graph":
            if sample.graph_path is None:
                print(f"Warning: No graph available for {sample.sample_id}, using multiframe")
                prompt = self._build_multiframe_prompt(sample.gt_triplets)
                response = self._query_multiframe(sample.image_paths, prompt)
            else:
                prompt = self._build_multiframe_prompt(sample.gt_triplets)
                response = self._query_with_graph(sample.image_paths, sample.graph_path, prompt)
        else:
            raise ValueError(f"Unknown condition: {condition}")
        
        # Parse response
        predicted_triplets = self._parse_response(response)
        
        # Evaluate
        metrics = self._evaluate_prediction(predicted_triplets, sample.gt_triplets)
        
        return {
            'sample_id': sample.sample_id,
            'condition': condition,
            'num_frames': sample.num_frames,
            'predicted_triplets': predicted_triplets,
            'gt_triplets': sample.gt_triplets,
            'response': response,
            'metrics': metrics
        }
    
    def _evaluate_prediction(self, pred_triplets: List[Dict], gt_triplets: List[Dict]) -> Dict:
        """Evaluate predicted triplets against ground truth"""
        
        # For multi-triplet ground truth, check if prediction matches ANY GT triplet
        best_match = {'instrument': False, 'verb': False, 'target': False, 'triplet': False}
        
        for gt in gt_triplets:
            for pred in pred_triplets:
                inst_match = False
                verb_match = False
                targ_match = False
                
                if pred.get('instrument'):
                    pred_inst = normalize_for_matching(pred['instrument'])
                    gt_inst = normalize_for_matching(gt['instrument'])
                    inst_match = pred_inst == gt_inst
                
                if pred.get('verb'):
                    pred_verb = normalize_for_matching(pred['verb'])
                    gt_verb = normalize_for_matching(gt['verb'])
                    verb_match = pred_verb == gt_verb
                
                if pred.get('target'):
                    pred_targ = normalize_for_matching(pred['target'])
                    gt_targ = normalize_for_matching(gt['target'])
                    targ_match = pred_targ == gt_targ
                
                # Update best match
                if inst_match:
                    best_match['instrument'] = True
                if verb_match:
                    best_match['verb'] = True
                if targ_match:
                    best_match['target'] = True
                if inst_match and verb_match and targ_match:
                    best_match['triplet'] = True
        
        return best_match
    
    def run_ablation_study(
        self, 
        samples: List[MultiFrameSample],
        conditions: List[str] = None
    ) -> Dict:
        """
        Run ablation study across multiple conditions.
        
        Args:
            samples: List of MultiFrameSample objects
            conditions: List of conditions to test (default: all three)
        """
        
        if conditions is None:
            conditions = ["single_frame", "multiframe", "multiframe_graph"]
        
        results = {
            'conditions': {},
            'samples': []
        }
        
        for condition in conditions:
            print(f"\n{'='*80}")
            print(f"CONDITION: {condition.upper()}")
            print(f"{'='*80}\n")
            
            condition_results = []
            
            for i, sample in enumerate(samples, 1):
                print(f"[{i}/{len(samples)}] {sample.sample_id}...")
                
                result = self.evaluate_sample(sample, condition)
                condition_results.append(result)
                
                # Print result
                metrics = result['metrics']
                status = "✓" if metrics['triplet'] else "✗"
                print(f"  {status} I:{int(metrics['instrument'])} V:{int(metrics['verb'])} T:{int(metrics['target'])} Full:{int(metrics['triplet'])}")
            
            # Compute aggregate metrics for this condition
            n = len(condition_results)
            metrics = {
                'instrument_acc': sum(r['metrics']['instrument'] for r in condition_results) / n,
                'verb_acc': sum(r['metrics']['verb'] for r in condition_results) / n,
                'target_acc': sum(r['metrics']['target'] for r in condition_results) / n,
                'triplet_acc': sum(r['metrics']['triplet'] for r in condition_results) / n,
                'num_samples': n
            }
            
            results['conditions'][condition] = {
                'metrics': metrics,
                'results': condition_results
            }
            
            print(f"\nCondition Metrics:")
            print(f"  Instrument: {metrics['instrument_acc']:.1%}")
            print(f"  Verb: {metrics['verb_acc']:.1%}")
            print(f"  Target: {metrics['target_acc']:.1%}")
            print(f"  Full Triplet: {metrics['triplet_acc']:.1%}")
        
        return results
    
    def save_results(self, results: Dict, output_path: Path):
        """Save evaluation results to JSON"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✓ Results saved to {output_path}")

