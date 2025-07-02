import os
import argparse
import json
from datasets import load_dataset
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(
        description="Download a subset of the C4 dataset and save it as a JSONL file."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="The directory where the downloaded JSONL file will be saved."
    )
    # 保存するレコード数を引数で指定できるようにする（オプション）
    parser.add_argument(
        "--num_records",
        type=int,
        default=10_000_000, # デフォルトを1000万件に設定
        help="Number of records to download from the dataset."
    )
    args = parser.parse_args()

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, "c4.jsonl")
    
    num_to_save = args.num_records

    print(f"C4データセットの先頭{num_to_save:,}件のダウンロードと変換を開始します...")
    print(f"保存先ファイル: {filepath}")

    try:
        dataset_stream = load_dataset("allenai/c4", "en", split="train", streaming=True)

        # .take() メソッドで、データセットの先頭から指定した件数だけを取り出す
        limited_stream = dataset_stream.take(num_to_save)

        with open(filepath, "w", encoding="utf-8") as f:
            # total を指定することで、プログレスバーの全体量がわかるようになる
            progress_bar = tqdm(limited_stream, total=num_to_save, desc=f"Writing to {filepath}")
            
            for example in progress_bar:
                json_line = json.dumps(example, ensure_ascii=False)
                f.write(json_line + '\n')

    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        if os.path.exists(filepath):
            print(f"不完全なファイル {filepath} を削除します。")
            os.remove(filepath)
        exit(1)

    print(f"\n'{filepath}' の保存が完了しました。 ({num_to_save:,}件)")
    print("\nすべての処理が完了しました。")


if __name__ == "__main__":
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterator, *args, **kwargs):
            print("tqdmライブラリがありません。'pip install tqdm' で進捗バーを表示できます。")
            return iterator

    main()