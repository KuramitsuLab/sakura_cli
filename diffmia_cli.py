#!/usr/bin/env python3
"""
diffmia_cli.py - Differential Membership Inference Attack CLI

テキストの変更前後の差分を使って、MIA評価を実行するツール
"""

import argparse
import json
import difflib
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def load_jsonl(file_path):
    """JSONLファイルを読み込む"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_jsonl(data, file_path):
    """JSONLファイルに保存"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def calculate_edit_similarity(text1, text2):
    """編集距離類似度を計算"""
    # 文字レベルでの編集距離類似度
    seq_matcher = difflib.SequenceMatcher(None, text1, text2)
    return seq_matcher.ratio()

def calculate_loss(model, tokenizer, text, device):
    """単一テキストのlossを計算"""
    try:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss.item()
        
        return loss
    except Exception as e:
        print(f"Error calculating loss: {e}")
        return float('inf')


def calculate_min_prob(model, tokenizer, text, device, k=20):
    """min k%probを計算"""
    try:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0]  # (seq_len, vocab_size)
            probs = torch.softmax(logits, dim=-1)
            
            # 各トークンの確率を取得
            token_probs = []
            for i in range(len(inputs['input_ids'][0]) - 1):
                next_token_id = inputs['input_ids'][0][i + 1]
                prob = probs[i, next_token_id].item()
                token_probs.append(prob)
            
            # 下位k%の確率の平均を計算
            if len(token_probs) > 0:
                k_percent = max(1, int(len(token_probs) * k / 100))
                min_k_probs = sorted(token_probs)[:k_percent]
                return np.mean(min_k_probs)
            else:
                return 0.0
    except Exception as e:
        print(f"Error calculating min prob: {e}")
        return 0.0


def evaluate_single_sample(model, tokenizer, before_text, after_text, device):
    """単一サンプルの評価"""
    # Before text evaluation
    before_loss = calculate_loss(model, tokenizer, before_text, device)
    before_min20 = calculate_min_prob(model, tokenizer, before_text, device, k=20)
    before_min40 = calculate_min_prob(model, tokenizer, before_text, device, k=40)
    
    # After text evaluation
    after_loss = calculate_loss(model, tokenizer, after_text, device)
    after_min20 = calculate_min_prob(model, tokenizer, after_text, device, k=20)
    after_min40 = calculate_min_prob(model, tokenizer, after_text, device, k=40)
    
    # Calculate differences
    loss_diff = after_loss - before_loss
    min20_diff = after_min20 - before_min20
    min40_diff = after_min40 - before_min40
    
    return {
        'before_loss': before_loss,
        'before_min20': before_min20,
        'before_min40': before_min40,
        'after_loss': after_loss,
        'after_min20': after_min20,
        'after_min40': after_min40,
        'loss_diff': loss_diff,
        'min20_diff': min20_diff,
        'min40_diff': min40_diff
    }


def calculate_optimal_threshold(y_true, y_scores):
    """ROC曲線から最適な閾値を計算"""
    try:
        # 無効な値を除去
        valid_indices = [i for i, score in enumerate(y_scores) if not np.isnan(score) and not np.isinf(score)]
        
        if len(valid_indices) < 2:
            return 0.0, 0.5
        
        y_true_valid = [y_true[i] for i in valid_indices]
        y_scores_valid = [y_scores[i] for i in valid_indices]
        
        # ROC曲線計算
        fpr, tpr, thresholds = roc_curve(y_true_valid, y_scores_valid)
        
        # Youden's J statistic (tpr - fpr)を最大化する閾値
        j_scores = tpr - fpr
        best_threshold_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_threshold_idx]
        
        # AUC計算
        auc = roc_auc_score(y_true_valid, y_scores_valid)
        
        return best_threshold, auc
    except Exception as e:
        print(f"Error calculating threshold: {e}")
        return 0.0, 0.5


def main():
    parser = argparse.ArgumentParser(description="Differential MIA CLI")
    parser.add_argument("input_file", help="入力JSONLファイル")
    parser.add_argument("--model_path", default="sbintuitions/sarashina2.2-0.5b", 
                       help="HuggingFace モデルパス")
    parser.add_argument("--output_csv", default="output.csv", 
                       help="出力CSVファイル")
    
    args = parser.parse_args()
    
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # モデルとトークナイザーのロード
    print(f"Loading model: {args.model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
            device_map="auto" if device.type == 'cuda' else None
        )
        model.eval()
        
        # パディングトークンの設定
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # データ読み込み
    print(f"Loading data from: {args.input_file}")
    try:
        data = load_jsonl(args.input_file)
        print(f"Loaded {len(data)} samples")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # 評価実行
    print("Evaluating samples...")
    results = []
    
    for i, sample in enumerate(tqdm(data, desc="Processing")):
        if 'before_text' not in sample or 'after_text' not in sample:
            print(f"Warning: Sample {i} missing required fields")
            continue
        
        try:
            editsim = calculate_edit_similarity(sample['before_text'], sample['after_text'])
            eval_result = evaluate_single_sample(
                model, tokenizer, 
                sample['before_text'], 
                sample['after_text'], 
                device
            )
            
            # 元のデータに評価結果を追加
            updated_sample = {**sample, **{"editsim": editsim}, **eval_result}
            results.append(updated_sample)
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    # 結果保存
    input_path = Path(args.input_file)
    output_path = input_path.parent / f"{input_path.stem}_diffmia.jsonl"
    
    print(f"Saving results to: {output_path}")
    save_jsonl(results, output_path)
    
    # 分類評価
    print("Calculating classification metrics...")
    
    # loss_diffが存在するサンプルのみ使用
    valid_samples = [s for s in results if 'loss_diff' in s and not np.isnan(s['loss_diff']) and not np.isinf(s['loss_diff'])]
    
    if len(valid_samples) == 0:
        print("No valid samples for classification")
        return
    
    # loss_diff > 0をクラス1、loss_diff <= 0をクラス0として扱う
    # y_true = [1 if s['loss_diff'] > 0 else 0 for s in valid_samples]
    y_true = [1 for s in valid_samples] + [0 for s in valid_samples]
    
    # 各メトリクスでの閾値とAUC計算
    loss_diffs0 = [s['loss_diff'] for s in valid_samples] + [0.0 for s in valid_samples]
    loss_diffs = [s['after_loss'] for s in valid_samples] + [s['before_loss'] for s in valid_samples]
    min20_diffs = [s['after_min20'] for s in valid_samples] + [s['before_min20'] for s in valid_samples]
    min40_diffs = [s['after_min40'] for s in valid_samples] + [s['before_min40'] for s in valid_samples]
    
    threshold_loss, auc_loss = calculate_optimal_threshold(y_true, loss_diffs)
    threshold_min20, auc_min20 = calculate_optimal_threshold(y_true, min20_diffs)
    threshold_min40, auc_min40 = calculate_optimal_threshold(y_true, min40_diffs)
    threshold_diff, auc_diff = calculate_optimal_threshold(y_true, loss_diffs0)
    
    auc_loss = auc_loss if auc_loss > 0.5 else 1.0-auc_loss
    auc_min20 = auc_min20 if auc_min20 > 0.5 else 1.0-auc_min20
    auc_min40 = auc_min40 if auc_min40 > 0.5 else 1.0-auc_min40
    auc_diff = auc_diff if auc_diff > 0.5 else 1.0-auc_diff

    # CSV結果作成
    csv_result = {
        'model_path': args.model_path,
        'jsonl_file': args.input_file,
        'auc_loss': auc_loss,
        'auc_min20': auc_min20,
        'auc_min40': auc_min40,
        'auc_diff': auc_diff,    
        'threshold_loss': threshold_loss,
        'threshold_min20': threshold_min20,
        'threshold_min40': threshold_min40,
        'threshold_diff': threshold_diff,
    }
    
    # CSVに保存
    df = pd.DataFrame([csv_result])
    
    # 既存ファイルがあれば追記
    if Path(args.output_csv).exists():
        existing_df = pd.read_csv(args.output_csv)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    df.to_csv(args.output_csv, index=False)
    
    print(f"\nCSV results saved to: {args.output_csv}")
    print(f"Loss - Threshold: {threshold_loss:.4f}, AUC: {auc_loss:.4f}")
    print(f"Min20% - Threshold: {threshold_min20:.4f}, AUC: {auc_min20:.4f}")
    print(f"Min40% - Threshold: {threshold_min40:.4f}, AUC: {auc_min40:.4f}")
    print(f"DiffLoss - Threshold: {threshold_diff:.4f}, AUC: {auc_diff:.4f}")
    print(f"Processed {len(valid_samples)} valid samples")


if __name__ == "__main__":
    main()