#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Loss Calculation CLI Tool
指定された年の各月の最初の日の文字列に対してLLMのloss値を計算するツール
"""

import argparse
import csv
import datetime
import os
import sys
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def get_weekday_name(date: datetime.date, lang: str) -> str:
    """指定された日付の曜日名を指定された言語で取得"""
    weekdays = {
        'en_us': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        'ja': ['月', '火', '水', '木', '金', '土', '日'],
        'zh': ['一', '二', '三', '四', '五', '六', '日'],
        'kr': ['월', '화', '수', '목', '금', '토', '일']
    }
    
    return weekdays[lang][date.weekday()]


def generate_date_string(year: int, month: int, lang: str) -> str:
    """指定された年月の最初の日の文字列を生成"""
    date = datetime.date(year, month, 1)
    weekday = get_weekday_name(date, lang)
    
    if lang == 'en_us':
        return f"Event Date: {date.strftime('%B %d, %Y')} ({weekday})"
    elif lang == 'ja':
        return f"開催日: {year}年{month}月{date.day}日（{weekday}）"
    elif lang == 'zh':
        return f"举办日期: {year}年{month}月{date.day}日（周{weekday}）"
    elif lang == 'kr':
        return f"개최일: {year}년 {month}월 {date.day}일 ({weekday}요일)"
    else:
        raise ValueError(f"Unsupported language: {lang}")


def calculate_loss(model, tokenizer, text: str, device: str) -> float:
    """指定されたテキストに対してモデルのloss値を計算"""
    # テキストをトークナイズ
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # モデルの出力を取得
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()
    
    return loss


def parse_year_range(year_str: str) -> List[int]:
    """年の範囲文字列をパースして年のリストを返す"""
    if '-' in year_str:
        # 範囲指定 (例: "2019-2022")
        start_year, end_year = map(int, year_str.split('-'))
        if start_year > end_year:
            raise ValueError(f"Invalid year range: {year_str}. Start year must be <= end year")
        return list(range(start_year, end_year + 1))
    else:
        # 単一年 (例: "2022")
        return [int(year_str)]


def load_model_and_tokenizer(model_path: str, device: str) -> Tuple:
    """モデルとトークナイザーをロード"""
    print(f"Loading model from: {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        # パッドトークンが設定されていない場合は設定
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.to(device)
        model.eval()
        
        print(f"Model loaded successfully on {device}")
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def detect_loss_spikes(results: List[dict], min_diff: float = 0.5) -> List[dict]:
    """前3ヶ月の平均値に対して最も大きな差分を示した月を検出する"""
    if len(results) < 4:  # 最低4データポイント必要（3ヶ月 + 現在月）
        return []
    
    spikes = []
    
    for i in range(3, len(results)):  # 前3ヶ月のデータが必要
        # 前3ヶ月のloss値の平均を計算
        prev_3_months = [results[j]['loss'] for j in range(i-3, i)]
        mean_prev_3 = sum(prev_3_months) / len(prev_3_months)
        
        current_loss = results[i]['loss']
        
        # 差分を計算
        diff = current_loss - mean_prev_3
        
        # 最小差分閾値を超えた場合のみスパイクとして記録
        if diff > min_diff:
            spike_info = {
                'year': results[i]['year'],
                'month': results[i]['month'],
                'current_loss': current_loss,
                'prev_3_mean': mean_prev_3,
                'diff': diff,
                'prev_3_months': prev_3_months,
                'month_index': i
            }
            spikes.append(spike_info)
    
    return spikes


def save_to_csv(results: List[dict], save_file: str):
    """結果をCSVファイルに追記保存"""
    fieldnames = ['model_path', 'lang', 'year', 'month', 'loss']
    
    # ファイルが存在するかチェック
    file_exists = os.path.exists(save_file)
    
    with open(save_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # ファイルが存在しない場合はヘッダーを書き込み
        if not file_exists:
            writer.writeheader()
        
        writer.writerows(results)
    
    if file_exists:
        print(f"Results appended to: {save_file}")
    else:
        print(f"New file created: {save_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate LLM loss values for monthly date strings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  python3 cutoff_cli.py --model_path gpt2 --year 2022 --lang ja --save_file results.csv
  python3 cutoff_cli.py --model_path gpt2 --year 2019-2022 --lang ja --save_file results.csv
  python3 cutoff_cli.py --model_path microsoft/DialoGPT-medium --year 2023 --lang en_us --spike_threshold 1.0
        """
    )
    
    parser.add_argument(
        '--model_path',
        required=True,
        help='Hugging Face model path or local model directory'
    )
    
    parser.add_argument(
        '--year',
        required=True,
        help='Target year or year range (e.g., "2022" or "2019-2022")'
    )
    
    parser.add_argument(
        '--lang',
        choices=['en_us', 'ja', 'zh', 'kr'],
        required=True,
        help='Language for date string generation'
    )
    
    parser.add_argument(
        '--save_file',
        default='loss_results.csv',
        help='CSV file path to save results (default: loss_results.csv)'
    )
    
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run the model on'
    )
    
    parser.add_argument(
        '--spike_threshold',
        type=float,
        default=0.5,
        help='Minimum difference from 3-month average to detect as spike (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    # 年の範囲をパース
    try:
        years = parse_year_range(args.year)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # 入力検証
    for year in years:
        if year < 1900 or year > 2100:
            print(f"Error: Year {year} must be between 1900 and 2100")
            sys.exit(1)
    
    # デバイス確認
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'
    
    print(f"Configuration:")
    print(f"  Model: {args.model_path}")
    print(f"  Years: {years[0] if len(years) == 1 else f'{years[0]}-{years[-1]}'} ({len(years)} years)")
    print(f"  Language: {args.lang}")
    print(f"  Device: {args.device}")
    print(f"  Save file: {args.save_file}")
    print(f"  Spike detection threshold: +{args.spike_threshold} from 3-month average")
    print()
    
    # モデルとトークナイザーをロード
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)
    
    # 全体の進捗を計算
    total_months = len(years) * 12
    results = []
    
    # tqdmでの進捗表示
    with tqdm(total=total_months, desc="Processing", unit="month") as pbar:
        for year in years:
            for month in range(1, 13):  # 1月から12月まで
                try:
                    # 日付文字列を生成
                    date_string = generate_date_string(year, month, args.lang)
                    
                    # loss値を計算
                    loss = calculate_loss(model, tokenizer, date_string, args.device)
                    
                    # 結果を保存
                    result = {
                        'model_path': args.model_path,
                        'lang': args.lang,
                        'year': year,
                        'month': month,
                        'loss': round(loss, 6)
                    }
                    results.append(result)
                    
                    # 進捗バーの説明を更新
                    pbar.set_postfix({
                        'Year': year,
                        'Month': f'{month:02d}',
                        'Loss': f'{loss:.4f}'
                    })
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"\nError processing {year}/{month:02d}: {e}")
                    pbar.update(1)
                    continue
    
    # 結果をCSVに保存
    if results:
        save_to_csv(results, args.save_file)
        print(f"\nCompleted! Processed {len(results)}/{total_months} months successfully.")
        
        # 統計情報を表示
        if len(results) > 0:
            losses = [r['loss'] for r in results]
            print(f"\nLoss statistics:")
            print(f"  Min: {min(losses):.6f}")
            print(f"  Max: {max(losses):.6f}")
            print(f"  Mean: {sum(losses)/len(losses):.6f}")
            
            # 前3ヶ月平均からの差分でlossスパイクを検出
            spikes = detect_loss_spikes(results, args.spike_threshold)
            if spikes:
                print(f"\n⚠️  Loss spikes detected ({len(spikes)} occurrences):")
                print("   Year/Month    Current    3-Month Avg    Difference    Previous 3 Months")
                print("   " + "="*85)
                for spike in spikes:
                    prev_months_str = " ".join([f"{x:.3f}" for x in spike['prev_3_months']])
                    print(f"   {spike['year']}/{spike['month']:02d}        "
                          f"{spike['current_loss']:.4f}      "
                          f"{spike['prev_3_mean']:.4f}       "
                          f"+{spike['diff']:.4f}       "
                          f"[{prev_months_str}]")
                
                # 最も大きな差分を特別に表示
                max_spike = max(spikes, key=lambda x: x['diff'])
                print(f"\n📈 Largest spike: {max_spike['year']}/{max_spike['month']:02d} "
                      f"(+{max_spike['diff']:.4f} above 3-month average)")
                
                # 追加の分析情報
                print(f"\n📊 Spike analysis:")
                differences = [s['diff'] for s in spikes]
                print(f"   Average difference: +{sum(differences)/len(differences):.4f}")
                print(f"   Range: +{min(differences):.4f} to +{max(differences):.4f}")
                
                # 最初のスパイクの時期を特定（知識カットオフの可能性）
                first_spike = min(spikes, key=lambda x: (x['year'], x['month']))
                print(f"   First spike detected: {first_spike['year']}/{first_spike['month']:02d} "
                      f"(potential knowledge cutoff)")
            else:
                print(f"\n✅ No significant spikes detected (threshold: +{args.spike_threshold} from 3-month average)")
    else:
        print("No results to save.")
        sys.exit(1)


if __name__ == "__main__":
    main()

