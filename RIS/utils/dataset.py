import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from typing import List, Union
import json
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

info = {
    'refcoco': {
        'train': 42404,
        'val': 3811,
        'val-test': 3811,
        'testA': 1975,
        'testB': 1810
    },
    'refcoco+': {
        'train': 42278,
        'val': 3805,
        'val-test': 3805,
        'testA': 1975,
        'testB': 1798
    },
    'refcocog_u': {
        'train': 42226,
        'val': 2573,
        'val-test': 2573,
        'test': 5023
    },
    'refcocog_g': {
        'train': 44822,
        'val': 5000,
        'val-test': 5000
    },
    'refcoco_mixed': {
        'train': 126908, # 42404+42278+42226=126908
        'val': 10189, # 3811+3805+2573=10189
    }
}
_tokenizer = _Tokenizer()


def tokenize(texts: Union[str, List[str]],
             context_length: int = 77,
             truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token]
                  for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result
class RefDataset(Dataset):
    def __init__(self, txt_dir, mask_dir, mode, input_size, word_length,dataset,split):
        super(RefDataset, self).__init__()
        self.txt_dir = txt_dir
        self.mask_dir = mask_dir
        self.mode = mode
        self.input_size = (input_size, input_size)
        self.word_length = word_length
        self.dataset=dataset
        self.split=split
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
        self.data = self._load_data_from_txt()

    def _load_data_from_txt(self):
        data = []
        base_dir = "/RIS/datasets/"
        with open(self.txt_dir, 'r') as file:
            for line in file:
                json_data = json.loads(line.strip())
                img_path = os.path.join(base_dir,json_data["img_path"])
                mask_path = os.path.join(base_dir,json_data["mask_path"])
                category = json_data["cat"]
                seg_id = json_data["seg_id"]
                img_name = json_data["img_name"]
                num_sents = json_data["num_sents"]
                sentences = json_data["sents"]
                data.append({
                    'img_path': img_path,
                    'mask_path': mask_path,
                    'category': category,
                    'seg_id': seg_id,
                    'img_name': img_name,
                    'num_sents': num_sents,
                    'sents': sentences
                })
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ref = self.data[index]
        # img
        ori_img = cv2.imread(ref['img_path'])
        img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        img_size = img.shape[:2]
        # mask
        mask_dir = os.path.join(self.mask_dir, str(ref['seg_id']) + '.png')
        # sentences
        idx = np.random.choice(ref['num_sents'])
        sents = ref['sents']
        # transform
        mat, mat_inv = self.getTransformMat(img_size, True)
        img = cv2.warpAffine(
            img,
            mat,
            self.input_size,
            flags=cv2.INTER_CUBIC,
            borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255]
        )
        if self.mode == 'train':
            # mask transform
            mask = cv2.imread(ref['mask_path'], cv2.IMREAD_GRAYSCALE)
            mask = cv2.warpAffine(mask, mat, self.input_size, flags=cv2.INTER_LINEAR, borderValue=0.)
            mask = mask / 255.
            # sentence -> vector
            sent = sents[idx]
            word_vec = tokenize(sent, self.word_length, True).squeeze(0)
            img, mask = self.convert(img, mask)
            return img, word_vec, mask
        elif self.mode == 'val':
            sent = sents[0]
            word_vec = tokenize(sent, self.word_length, True).squeeze(0)
            img = self.convert(img)[0]
            params = {'mask_dir': mask_dir, 'inverse': mat_inv, 'ori_size': np.array(img_size)}
            return img, word_vec, params
        else:
            img = self.convert(img)[0]
            params = {
                'ori_img': ori_img,
                'seg_id': ref['seg_id'],
                'mask_dir': mask_dir,
                'inverse': mat_inv,
                'ori_size': np.array(img_size),
                'sents': sents
            }
            return img, params

    def getTransformMat(self, img_size, inverse=False):
        ori_h, ori_w = img_size
        inp_h, inp_w = self.input_size
        scale = min(inp_h / ori_h, inp_w / ori_w)
        new_h, new_w = ori_h * scale, ori_w * scale
        bias_x, bias_y = (inp_w - new_w) / 2., (inp_h - new_h) / 2.

        src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
        dst = np.array([[bias_x, bias_y], [new_w + bias_x, bias_y],
                        [bias_x, new_h + bias_y]], np.float32)

        mat = cv2.getAffineTransform(src, dst)
        if inverse:
            mat_inv = cv2.getAffineTransform(dst, src)
            return mat, mat_inv
        return mat, None

    def convert(self, img, mask=None):
        img = torch.from_numpy(img.transpose((2, 0, 1))).float().div_(255.).sub_(self.mean).div_(self.std)
        if mask is not None:
            mask = torch.from_numpy(mask).float()
        return img, mask

    def __repr__(self):
        return self.__class__.__name__ + "(" + \
            f"txt_dir={self.txt_dir}, " + \
            f"mask_dir={self.mask_dir}, " + \
            f"mode={self.mode}, " + \
            f"input_size={self.input_size}, " + \
            f"word_length={self.word_length})"