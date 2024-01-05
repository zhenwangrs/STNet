import os

from matplotlib import pyplot as plt
from tqdm import tqdm


def stat_img_len_with_text_len(stm_path='E:/Research/CSL/CorrNet/evaluation/slr_eval/phoenix2014-T-groundtruth-train.stm',
                               img_dir='E:/dataset/CSL/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/train'):
    max_len = 0
    max_img_len = 0
    max_img_len_ratio = 0
    img_text_pairs = []
    all_img_length = []
    all_text_length = []
    all_ratio = []
    with open(stm_path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            img_name = line.split()[0]
            sub_dir = line.split()[1]
            img_len = len(os.listdir(os.path.join(img_dir, img_name, sub_dir)))
            all_img_length.append(img_len)
            if img_len > max_img_len:
                max_img_len = img_len

            ss = line.split()[5:]
            text_len = len(ss)
            all_text_length.append(text_len)
            if text_len > max_len:
                max_len = text_len

            img_text_ratio = img_len / text_len
            all_ratio.append(img_text_ratio)
            if img_text_ratio > max_img_len_ratio:
                max_img_len_ratio = img_text_ratio
                ratio = (img_name, img_len, text_len)

            img_text_pairs.append((img_len, len(ss)))

    print(max_len)
    print(max_img_len)
    print(max_img_len_ratio)
    print(ratio)

    all_img_length = sorted(all_img_length)
    # 统计每种长度的图片数量
    img_len_dict = {}
    for img_len in all_img_length:
        if img_len in img_len_dict:
            img_len_dict[img_len] += 1
        else:
            img_len_dict[img_len] = 1
    print(img_len_dict)

    all_text_length = sorted(all_text_length)
    # 统计每种长度的文本数量
    text_len_dict = {}
    for text_len in all_text_length:
        if text_len in text_len_dict:
            text_len_dict[text_len] += 1
        else:
            text_len_dict[text_len] = 1
    print(text_len_dict)

    all_ratio = sorted(all_ratio)
    # 统计每种长度的图片数量
    ratio_dict = {}
    for ratio in all_ratio:
        ratio = int(ratio)
        if ratio in ratio_dict:
            ratio_dict[ratio] += 1
        else:
            ratio_dict[ratio] = 1
    print(ratio_dict)

    print(all_img_length[-200:])
    print(all_text_length[-200:])
    print(all_ratio[:200])
    print(img_text_pairs)


if __name__ == '__main__':
    split = 'dev'
    # stm_path = 'E:/Research/CSL/CorrNet/evaluation/slr_eval/phoenix2014-T-groundtruth-{}.stm'
    # img_dir = 'E:/dataset/CSL/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/{}'

    stm_path = 'F:/Research/CSL/CLGNN/CorrNet2/evaluation/slr_eval/phoenix2014-groundtruth-{}.stm'
    img_dir = 'D:/dataset/CSL/PHOENIX-2014/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/{}'
    stat_img_len_with_text_len(stm_path.format(split), img_dir.format(split))

    # img_list = [i for i in range(280)]
    # if len(img_list) > 200:
    #     # 随机采样200并按顺序排列
    #     img_list = sorted(random.sample(img_list, 200))
    # print(img_list)
    # print(len(img_list))