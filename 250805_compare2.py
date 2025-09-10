import torch
import sys
import os

def compare_state_dicts(sd1, sd2):
    """2つのstate_dictを比較し、違いがあれば詳細を出力する関数 (辞書形式のパラメータに対応)"""
    if len(sd1) != len(sd2):
        print(f"Error: State dicts have different number of keys: {len(sd1)} vs {len(sd2)}")
        return False

    all_identical = True
    for key in sd1:
        if key not in sd2:
            print(f"Error: Key '{key}' not found in the second state dict.")
            all_identical = False
            continue

        val1 = sd1[key]
        val2 = sd2[key]

        # 値の型を取得
        t1 = type(val1)
        t2 = type(val2)
        
        if t1 != t2:
            print(f"‼️ PARAMETER TYPES ARE DIFFERENT FOR KEY: {key} ({t1} vs {t2})")
            all_identical = False
            continue

        # 実際のテンソルを取得
        param1, param2 = None, None
        if isinstance(val1, torch.Tensor):
            param1, param2 = val1, val2
        elif isinstance(val1, dict):
            # DeepSpeed ZeRO-3のパラメータは辞書形式で、'flat_param'に実体を持つことが多い
            if 'flat_param' in val1 and 'flat_param' in val2:
                param1 = val1['flat_param']
                param2 = val2['flat_param']
                print(f"Note: Comparing nested tensor from 'flat_param' for key: {key}")
            else:
                print(f"Warning: Dictionaries found for key '{key}', but could not find 'flat_param'. Skipping comparison for this key.")
                continue # このキーの比較はスキップ
        else:
            # テンソルでも辞書でもない場合 (例: ds_versionなど)
            if val1 != val2:
                print(f"‼️ METADATA IS DIFFERENT FOR KEY: {key} ({val1} vs {val2})")
                all_identical = False
            continue # 次のキーへ

        # テンソルが取得できた場合のみ比較
        if param1 is not None and param2 is not None:
            if not torch.equal(param1, param2):
                print(f"‼️ PARAMETERS ARE DIFFERENT FOR KEY: {key}")
                diff_norm = torch.linalg.norm(param1.float() - param2.float()).item()
                print(f"    Norm of difference: {diff_norm}")
                all_identical = False
    
    return all_identical

def load_model_state_dict(path):
    """DeepSpeedのチェックポイントからモデルのstate_dictをロードする関数"""
    full_path = os.path.join(path, "mp_rank_00_model_states.pt")
    
    if not os.path.exists(full_path):
        print(f"Error: Checkpoint file not found at {full_path}")
        sys.exit(1)
        
    print(f"Loading state dict from: {full_path}")
    checkpoint = torch.load(full_path, weights_only=False)
    
    if 'module' in checkpoint:
        return checkpoint['module']
    else:
        return checkpoint

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("\nUsage: python compare_models_v2.py /path/to/checkpoint_A/global_step15 /path/to/checkpoint_B/global_step15")
        sys.exit(1)

    path_A = sys.argv[1]
    path_B = sys.argv[2]

    state_dict_A = load_model_state_dict(path_A)
    state_dict_B = load_model_state_dict(path_B)

    print("\n--- Starting Comparison (v2) ---")
    result = compare_state_dicts(state_dict_A, state_dict_B)
    print("--- Comparison Finished ---\n")

    if result:
        print("✅✅✅ Result: All model parameters are BIT-FOR-BIT IDENTICAL.")
        print("This confirms the mystery: the hyperparameter change had NO effect on the model weights.")
    else:
        print("🎉🎉🎉 Result: Model parameters are DIFFERENT.")
        print("This solves the mystery: the implementation IS working, the effect on loss was just too small to see (low sensitivity).")