import torch
from pathlib import Path


def load_mvpformer_partial(model, checkpoint_path, verbose=True):

    if checkpoint_path in (None, "", "None"):
        if verbose:
            print("No checkpoint provided → Using random initialization")
        return model, {'status': 'random_init'}
    
    try:
        ckpt_path = Path(checkpoint_path)
        if verbose:
            print(f"Loading weights from: {ckpt_path}")
        
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_dict = model.state_dict()
        
        # Filter: only keep keys with matching shapes
        filtered_dict = {}
        skipped_keys = []
        
        for k, v in state_dict.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    filtered_dict[k] = v
                else:
                    skipped_keys.append({
                        'key': k,
                        'ckpt_shape': v.shape,
                        'model_shape': model_dict[k].shape
                    })
        
        # Load filtered weights
        missing, unexpected = model.load_state_dict(filtered_dict, strict=False)
        
        stats = {
            'status': 'partial_load',
            'total_keys': len(state_dict),
            'loaded_keys': len(filtered_dict),
            'skipped_keys': len(skipped_keys),
            'missing_keys': len(missing),
            'unexpected_keys': len(unexpected),
            'skipped_details': skipped_keys,
        }
        
        if verbose:
            print(f"\n Partial Loading Complete:")
            print(f"    Loaded: {len(filtered_dict)}/{len(state_dict)} keys ({len(filtered_dict)/len(state_dict)*100:.1f}%)")
            
            if skipped_keys:
                print(f"\n  Skipped {len(skipped_keys)} shape-mismatched keys:")
                for item in skipped_keys[:5]:
                    print(f"    - {item['key']}")
                    print(f"      Checkpoint: {item['ckpt_shape']} → Model: {item['model_shape']}")
                if len(skipped_keys) > 5:
                    print(f"    ... and {len(skipped_keys)-5} more")
                
                # Analyze what was skipped
                encoder_skipped = [k for k in skipped_keys if 'encoder' in k['key']]
                transformer_skipped = [k for k in skipped_keys if 'transformer' in k['key'] or 'gpt' in k['key']]
                
                print(f"\n  Skipped breakdown:")
                print(f"    Encoder layers: {len(encoder_skipped)}")
                print(f"    Transformer layers: {len(transformer_skipped)}")
                print(f"    Other: {len(skipped_keys) - len(encoder_skipped) - len(transformer_skipped)}")
                
                if len(transformer_skipped) > 0:
                    print(f"\n WARNING: Some Transformer layers were skipped!")
                    print(f"    This is unusual and may affect performance.")
                else:
                    print(f"\n All Transformer layers loaded successfully!")
                    print(f"  → Only input-dependent layers are randomly initialized (expected)")
            
            if missing:
                print(f"\n  Missing keys: {len(missing)}")
                print(f"    (These will be randomly initialized)")
        
        return model, stats
        
    except Exception as e:
        if verbose:
            print(f"\n Failed to load checkpoint: {e}")
            import traceback
            traceback.print_exc()
        
        return model, {'status': 'failed', 'error': str(e)}


def analyze_checkpoint_compatibility(checkpoint_path, model, verbose=True):

    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_dict = model.state_dict()
        
        total = len(state_dict)
        compatible = 0
        incompatible = []
        
        for k, v in state_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                compatible += 1
            elif k in model_dict:
                incompatible.append({
                    'key': k,
                    'ckpt_shape': v.shape,
                    'model_shape': model_dict[k].shape
                })
        
        report = {
            'total_keys': total,
            'compatible_keys': compatible,
            'incompatible_keys': len(incompatible),
            'compatibility_rate': compatible / total * 100,
            'incompatible_details': incompatible,
        }
        
        if verbose:
            print(f"\n{'='*70}")
            print("Checkpoint Compatibility Analysis")
            print(f"{'='*70}")
            print(f"  Total keys in checkpoint: {total}")
            print(f"  Compatible keys: {compatible} ({report['compatibility_rate']:.1f}%)")
            print(f"  Incompatible keys: {len(incompatible)}")
            
            if incompatible:
                print(f"\n  Incompatible keys (shape mismatch):")
                for item in incompatible[:10]:
                    print(f"    - {item['key']}")
                    print(f"      Checkpoint: {item['ckpt_shape']} → Model: {item['model_shape']}")
                if len(incompatible) > 10:
                    print(f"    ... and {len(incompatible)-10} more")
            
            print(f"\n{'='*70}")
            print("Recommendation")
            print(f"{'='*70}")
            
            if report['compatibility_rate'] >= 90:
                print("High compatibility! Pretrained weights will be very useful.")
                print("  → Use partial loading (recommended)")
            elif report['compatibility_rate'] >= 70:
                print("Moderate compatibility. Most layers can be loaded.")
                print("  → Use partial loading with caution")
            else:
                print("Low compatibility. Consider:")
                print("    1. Training from scratch")
                print("    2. Using a different checkpoint")
                print("    3. Adjusting model architecture")
        
        return report
        
    except Exception as e:
        print(f"Failed to analyze: {e}")
        return {'status': 'failed', 'error': str(e)}


if __name__ == "__main__":
    """
    Example:
    
    from mvpformer_utils import load_mvpformer_partial
    
    # Build model
    mvpformer = build_mvpformer_from_yaml(size_input=2048)
    
    # Load with partial loading
    mvpformer, stats = load_mvpformer_partial(
        mvpformer, 
        "mvpformer/ckpts/genie-m-base.pt",
        verbose=True
    )
    
    # Freeze all parameters
    for param in mvpformer.parameters():
        param.requires_grad = False
    
    # Use in training
    ...
    """
    pass