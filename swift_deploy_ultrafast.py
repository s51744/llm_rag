#!/usr/bin/env python3
"""
Swift Deploy - Ultra Fast Configuration
Optimized for maximum speed with reasonable quality
"""

import sys
import os
import inspect

# Global variable to store the target max_new_tokens
TARGET_MAX_NEW_TOKENS = None

def extract_max_new_tokens():
    """Extract max_new_tokens from command line arguments"""
    global TARGET_MAX_NEW_TOKENS
    
    for i, arg in enumerate(sys.argv):
        if arg == '--max_new_tokens' and i + 1 < len(sys.argv):
            try:
                TARGET_MAX_NEW_TOKENS = int(sys.argv[i + 1])
                print(f"[INFO] Target max_new_tokens extracted: {TARGET_MAX_NEW_TOKENS}")
                return TARGET_MAX_NEW_TOKENS
            except ValueError:
                print(f"[WARNING] Invalid max_new_tokens value: {sys.argv[i + 1]}")
    
    # Default to a fast value if not specified
    TARGET_MAX_NEW_TOKENS = 256
    print(f"[INFO] Using default fast max_new_tokens: {TARGET_MAX_NEW_TOKENS}")
    return TARGET_MAX_NEW_TOKENS

def apply_vllm_compatibility_fix():
    """Apply comprehensive vLLM compatibility fixes"""
    if 'vllm._C' not in sys.modules:
        sys.modules['vllm._C'] = type('MockModule', (), {})()
    
    try:
        from vllm.engine.arg_utils import AsyncEngineArgs
        original_init = AsyncEngineArgs.__init__
        
        def patched_init(self, *args, **kwargs):
            sig = inspect.signature(original_init)
            param_names = set(sig.parameters.keys())
            
            unsupported_params = [
                'disable_log_requests',
                'enable_auto_tool_choice',
                'tool_call_parser'
            ]
            
            for param in unsupported_params:
                if param in kwargs:
                    print(f"[WARNING] Removing unsupported parameter: {param}")
                    del kwargs[param]
            
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in param_names}
            return original_init(self, *args, **filtered_kwargs)
        
        AsyncEngineArgs.__init__ = patched_init
        print("[INFO] Applied AsyncEngineArgs compatibility patch")
        
    except ImportError as e:
        print(f"[WARNING] Could not apply AsyncEngineArgs patch: {e}")
    except Exception as e:
        print(f"[WARNING] Error applying compatibility patch: {e}")

def patch_generation_config_ultra_fast():
    """Ultra-aggressive patching for maximum speed"""
    global TARGET_MAX_NEW_TOKENS
    
    if TARGET_MAX_NEW_TOKENS is None:
        TARGET_MAX_NEW_TOKENS = 256  # Fast default
    
    try:
        from transformers import GenerationConfig
        
        original_init = GenerationConfig.__init__
        original_update = GenerationConfig.update
        
        def ultra_fast_init(self, **kwargs):
            # Force ultra-fast generation parameters
            fast_kwargs = {
                'max_new_tokens': TARGET_MAX_NEW_TOKENS,
                'do_sample': True,
                'temperature': 0.8,
                'top_p': 0.9,
                'top_k': 50,
                'repetition_penalty': 1.05,
                'pad_token_id': kwargs.get('pad_token_id'),
                'eos_token_id': kwargs.get('eos_token_id'),
                'bos_token_id': kwargs.get('bos_token_id'),
                'return_dict_in_generate': True,
            }
            
            # Remove conflicting parameters
            conflicting_params = ['max_length', 'min_length', 'min_new_tokens']
            for param in conflicting_params:
                if param in kwargs:
                    del kwargs[param]
            
            # Override with fast parameters
            kwargs.update(fast_kwargs)
            
            print(f"[INFO] Ultra-fast GenerationConfig: max_new_tokens={TARGET_MAX_NEW_TOKENS}, temp=0.8")
            return original_init(self, **kwargs)
        
        def ultra_fast_update(self, **kwargs):
            # Always enforce fast parameters in updates
            fast_overrides = {
                'max_new_tokens': TARGET_MAX_NEW_TOKENS,
                'temperature': kwargs.get('temperature', 0.8),
                'top_p': kwargs.get('top_p', 0.9),
            }
            kwargs.update(fast_overrides)
            
            # Remove conflicting parameters
            for param in ['max_length', 'min_length', 'min_new_tokens']:
                if param in kwargs:
                    del kwargs[param]
            
            return original_update(self, **kwargs)
        
        GenerationConfig.__init__ = ultra_fast_init
        GenerationConfig.update = ultra_fast_update
        print(f"[INFO] Applied ultra-fast GenerationConfig patches")
        
    except Exception as e:
        print(f"[WARNING] Error applying ultra-fast GenerationConfig patch: {e}")

def optimize_command_line_args():
    """Optimize command line arguments for maximum speed"""
    global TARGET_MAX_NEW_TOKENS
    
    # Only add Swift-compatible optimizations
    swift_compatible_optimizations = {
        '--temperature': '0.8',
        '--top_p': '0.9',
        '--repetition_penalty': '1.05',
    }
    
    # Add quantization if not specified (Swift supports --quant_bits)
    if '--quant_bits' not in sys.argv and '--quantization_bit' not in sys.argv:
        sys.argv.extend(['--quant_bits', '8'])
        print("[INFO] Added 8-bit quantization for speed")
    
    # Add Swift-compatible optimizations
    for param, value in swift_compatible_optimizations.items():
        if param not in sys.argv:
            sys.argv.extend([param, value])
            print(f"[INFO] Added optimization: {param} {value}")
    
    # Add model_type if not specified (required by Swift)
    if '--model_type' not in sys.argv:
        # For Qwen2.5 models, use the correct model_type
        sys.argv.extend(['--model_type', 'qwen2_5'])
        print("[INFO] Added model_type 'qwen2_5' (correct type for Qwen2.5 models)")
    
    # Fix template_type to template for Swift compatibility
    if '--template_type' in sys.argv:
        template_index = sys.argv.index('--template_type')
        sys.argv[template_index] = '--template'
        print("[INFO] Fixed --template_type to --template for Swift compatibility")
    
    # Skip other problematic parameters that Swift doesn't recognize
    problematic_params = ['--max_model_len', '--use_flash_attn']
    print("[INFO] Skipping problematic parameters for Swift compatibility:", problematic_params)

    # Ensure LoRA weights are merged during deployment
    if '--merge_lora' not in sys.argv:
        sys.argv.extend(['--merge_lora', 'True'])
        print("[INFO] Added --merge_lora True for LoRA compatibility")

def check_bitsandbytes_dependency():
    """Checks if bitsandbytes is installed if 8-bit quantization is enabled."""
    quant_args = ['--quantization_bit', '--quant_bits']
    for arg in quant_args:
        if arg in sys.argv:
            try:
                quant_bit_index = sys.argv.index(arg)
                if quant_bit_index + 1 < len(sys.argv):
                    quant_bit_value = int(sys.argv[quant_bit_index + 1])
                    if quant_bit_value > 0:
                        try:
                            import bitsandbytes
                            print("[INFO] 'bitsandbytes' package found.")
                            return
                        except ImportError:
                            print("[ERROR] The 'bitsandbytes' package is required for 8-bit quantization.")
                            print("[ERROR] Please install it using: pip install bitsandbytes")
                            sys.exit(1)
            except ValueError:
                pass # Ignore if quantization arg is malformed or not an integer
            except Exception as e:
                print(f"[WARNING] Error checking 'bitsandbytes' dependency: {e}")

def main():
    print("[INFO] Starting Ultra-Fast Swift Deployment")
    print("[INFO] Optimizing for maximum speed...")
    
    # Extract and set target tokens
    extract_max_new_tokens()
    
    # Optimize command line arguments
    optimize_command_line_args()

    # Check for bitsandbytes if 8-bit quantization is enabled
    check_bitsandbytes_dependency()
    
    # Apply all patches
    apply_vllm_compatibility_fix()
    patch_generation_config_ultra_fast()
    
    print(f"[INFO] Final command: {' '.join(sys.argv)}")
    
    # Start deployment
    try:
        print("[INFO] Starting optimized Swift deployment...")
        from swift.cli.deploy import deploy_main
        deploy_main()
    except Exception as e:
        print(f"[ERROR] Deployment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()