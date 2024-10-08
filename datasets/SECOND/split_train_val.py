import os
from shutil import copyfile


def main():
    src_dir = r'C:\Users\DINHPHAN\Desktop\SCanNet\DATA_ROOT/'
    train_dir = r'C:\Users\DINHPHAN\Desktop\SCanNet\DATA_ROOT\train/'
    val_dir = r'C:\Users\DINHPHAN\Desktop\SCanNet\DATA_ROOT\val/'

    train_info = open(r'C:\Users\DINHPHAN\Desktop\SCanNet\train_info.txt', 'r')
    train_list = train_info.readlines()
    val_info = open(r'C:\Users\DINHPHAN\Desktop\SCanNet\val_info.txt', 'r')
    val_list = val_info.readlines()
    dir_names = ['im1', 'im2', 'label1', 'label2', 'label1_rgb', 'label2_rgb']

    count = 0
    for it in train_list:
        _, it_name = os.path.split(it.strip())
        for dir_name in dir_names:
            dst_dir = os.path.join(train_dir, dir_name)
            if not os.path.exists(dst_dir): os.makedirs(dst_dir)
            src_path = os.path.join(src_dir, dir_name, it_name)
            dst_path = os.path.join(dst_dir, it_name)
            copyfile(src_path, dst_path)
        count += 1
        if not count % 100: print('%d/%d images saved.' % (count, len(train_list)))

    count = 0
    for it in val_list:
        _, it_name = os.path.split(it.strip())
        for dir_name in dir_names:
            dst_dir = os.path.join(val_dir, dir_name)
            if not os.path.exists(dst_dir): os.makedirs(dst_dir)
            src_path = os.path.join(src_dir, dir_name, it_name)
            dst_path = os.path.join(dst_dir, it_name)
            copyfile(src_path, dst_path)
        count += 1
        if not count % 100: print('%d/%d images saved.' % (count, len(train_list)))


if __name__ == '__main__':
    main()
