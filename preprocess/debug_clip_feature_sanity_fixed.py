#!/usr/bin/env python3
"""
Fixed debug_clip_feature_sanity.py - Uses ConsistentCLIPExtractor

This script fixes the dimension mismatch problem by using the consistent CLIP model
that matches the preprocessing pipeline.
"""

import sys
import numpy as np
import torch
import argparse

# Add consistent processor path
sys.path.append('/home/tumai/fabian/4DLangSplatSurgery/mllm_eval')
sys.path.append('/home/tumai/team1/Ken/4DLangSplatSurgery/preprocess')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True, help='Feature file .npy')
    parser.add_argument('--segmap', required=True, help='Segmentation map .npy')
    parser.add_argument('--layer_index', type=int, default=0, help='Layer index')
    parser.add_argument('--topk', type=int, default=5, help='Top-k results')
    args = parser.parse_args()
    
    # Load data
    features = np.load(args.features)
    segmap = np.load(args.segmap)
    
    print(f"Loaded features: {features.shape} {features.dtype}")
    print(f"Feature L2 norms: min={np.linalg.norm(features, axis=1).min():.4f} "
          f"median={np.median(np.linalg.norm(features, axis=1)):.4f} "
          f"mean={np.linalg.norm(features, axis=1).mean():.4f} "
          f"max={np.linalg.norm(features, axis=1).max():.4f}")
    
    print(f"Loaded segmap: {segmap.shape} {segmap.dtype}")
    unique_ids = np.unique(segmap)
    print(f"Segmap unique ids (incl -1)={len(unique_ids)}; min={unique_ids.min()} max={unique_ids.max()} valid_count={len(unique_ids)}")
    
    # Calculate areas per ID
    areas = []
    for uid in unique_ids:
        if uid >= 0:  # Skip -1 (background)
            area = np.sum(segmap == uid)
            areas.append(area)
    
    if areas:
        print(f"Areas per id (pixels): min={min(areas)} median={np.median(areas):.0f} mean={np.mean(areas):.0f} max={max(areas)}")
    
    # Medical/surgical prompts
    prompts = [
        'an abdominal wall', 'a liver', 'gastrointestinal tract', 'fat', 
        'a grasper', 'connective tissue', 'blood', 'cystic duct', 
        'l-hook electrocautery', 'a gallbladder', 'hepatic vein', 'liver ligament',
        'hand', 'hands', 'finger', 'fingers', 'surgical instrument', 'tool'
    ]
    
    print(f"Prompts ({len(prompts)}): {prompts}")
    
    # Try to use consistent CLIP extractor
    try:
        from consistent_feature_processor import ConsistentCLIPExtractor
        
        print("Creating consistent CLIP extractor...")
        clip_extractor = ConsistentCLIPExtractor()
        print("✓ Consistent CLIP extractor created")
        
        # Encode text prompts
        with torch.no_grad():
            text_embeddings = []
            for prompt in prompts:
                text_emb = clip_extractor.encode_text(prompt)
                text_embeddings.append(text_emb.cpu().numpy())
            
            text_embeddings = np.vstack(text_embeddings)  # (P, 512)
        
        print(f"Text embeddings shape: {text_embeddings.shape}")
        
        # Check dimension compatibility
        if features.shape[1] != text_embeddings.shape[1]:
            print(f"ERROR: Feature dimension mismatch!")
            print(f"  Features: {features.shape[1]} dimensions")
            print(f"  Text embeddings: {text_embeddings.shape[1]} dimensions")
            print(f"  This indicates inconsistent CLIP model usage!")
            
            # Try to handle the mismatch
            if features.shape[1] == 768 and text_embeddings.shape[1] == 512:
                print("Detected 768→512 dimension mismatch")
                print("This suggests features were extracted with a different CLIP model")
                print("Solution: Re-extract features with ConsistentCLIPExtractor")
                return
            else:
                print("Unknown dimension mismatch pattern")
                return
        
        # Compute similarities
        features_norm = features / np.linalg.norm(features, axis=1, keepdims=True)
        text_norm = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        
        similarities = features_norm @ text_norm.T  # (N, P)
        
        print(f"\nSimilarity results:")
        print(f"Similarities shape: {similarities.shape}")
        
        # Show top matches for each feature
        for i in range(len(features)):
            top_indices = np.argsort(similarities[i])[-args.topk:][::-1]
            print(f"\nFeature {i}:")
            for j, idx in enumerate(top_indices):
                print(f"  {j+1}. {prompts[idx]}: {similarities[i, idx]:.4f}")
        
        print("\n✓ Debug completed successfully with consistent CLIP!")
        
    except ImportError:
        print("ERROR: ConsistentCLIPExtractor not available")
        print("Please ensure you have the consistent_feature_processor.py file")
        print("Falling back to analysis without text similarity...")
        
        # Just analyze the features without text comparison
        print(f"\nFeature analysis only:")
        print(f"Features shape: {features.shape}")
        print(f"Features mean: {features.mean():.4f}")
        print(f"Features std: {features.std():.4f}")
        
    except Exception as e:
        print(f"Error with consistent CLIP: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
