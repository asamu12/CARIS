import json
import os
from tqdm import tqdm
def merge_txt_datasets():
    input_base = "datasets/txt"
    output_dir = "datasets/txt/refcoco_mixed"
    datasets = ["refcoco", "refcoco+", "refcocog_u"]
    splits = ["train", "val"]

    os.makedirs(output_dir, exist_ok=True)

    for split in splits:
        output_path = os.path.join(output_dir, f"{split}.txt")
        total_count = 0

        with open(output_path, "w", encoding="utf-8") as f_out:
            for dataset in datasets:
                input_path = os.path.join(input_base, dataset, f"{split}.txt")

                if not os.path.exists(input_path):
                    print(f"Warning: {input_path} not found, skipping")
                    continue

                with open(input_path, "r", encoding="utf-8") as f_in:
                    lines = f_in.readlines()

                for line in tqdm(lines, desc=f"Processing {dataset}/{split}"):
                    try:
                        data = json.loads(line.strip())
                        data["seg_id"] = f"{dataset}_{data['seg_id']}"
                        old_mask_path = data["mask_path"]
                        new_mask_name = f"{dataset}_{os.path.basename(old_mask_path)}"
                        data["mask_path"] = os.path.join("masks/refcoco_mixed", new_mask_name)
                        f_out.write(json.dumps(data) + "\n")
                        total_count += 1
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in {input_path}, line: {line}")

        print(f"Merged {split}.txt with {total_count} samples")


if __name__ == "__main__":
    merge_txt_datasets()