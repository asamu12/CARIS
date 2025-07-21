import argparse
import os
import json


def folder2txt(json_data, img_dir, mask_dir, output_path):
    print(f"Writing data to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for item in json_data:
            img_path = os.path.join(img_dir, item['img_name'])
            mask_path = os.path.join(mask_dir, f"{item['segment_id']}.png")
            data = {
                'img_path': img_path,
                'mask_path': mask_path,
                'cat': item['cat'],
                'seg_id': item['segment_id'],
                'img_name': item['img_name'],
                'num_sents': item['sentences_num'],
                'sents': [i['sent'] for i in item['sentences']]
            }
            f.write(json.dumps(data) + '\n')


def parse_args():
    parser = argparse.ArgumentParser(description='Convert folder data to TXT.')
    parser.add_argument('-j', '--json-dir', type=str, required=True,
                        help='JSON file path（e.g., anns/refcoco/train.json）')
    parser.add_argument('-i', '--img-dir', type=str, required=True,
                        help='Image folder（e.g., images/train2014/）')
    parser.add_argument('-m', '--mask-dir', type=str, required=True,
                        help='Mask folder path（e.g., masks/refcoco）')
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                        help='Output directory（e.g., txt/refcoco）')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    split_name = os.path.splitext(os.path.basename(args.json_dir))[0]
    output_path = os.path.join(args.output_dir, f"{split_name}.txt")

    with open(args.json_dir, 'r') as f:
        json_data = json.load(f)

    folder2txt(json_data, args.img_dir, args.mask_dir, output_path)