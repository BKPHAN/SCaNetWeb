from PIL import Image
import os

def CutImage(image, output_dir, filename):
    width, height = image.size
    step = 256  # Kích thước của mỗi phần cắt

    # Đảm bảo thư mục đầu ra tồn tại
    os.makedirs(output_dir, exist_ok=True)

    # Lưu kích thước ban đầu của ảnh vào tệp
    with open(os.path.join(output_dir, f'{filename}_size.txt'), 'w') as f:
        f.write(f'{width},{height}')  # Lưu chiều rộng và chiều cao

    for i in range(0, width, step):
        for j in range(0, height, step):
            # Điều chỉnh i và j để đảm bảo các phần cuối cùng vẫn có kích thước 256x256 nếu cần
            if i + step > width:
                i = width - step  # Điều chỉnh để phần cắt cuối cùng không bị thừa
            if j + step > height:
                j = height - step  # Điều chỉnh để phần cắt cuối cùng không bị thừa

            # Định nghĩa hộp để cắt
            box = (i, j, i + step, j + step)
            image_part = image.crop(box)

            # Lưu phần ảnh với tên tệp đã định dạng dựa trên vị trí của nó
            part_filename = f"{filename}_{i}_{j}.png"
            image_part.save(os.path.join(output_dir, part_filename))
            print(f"{filename}_{i}_{j}.png")


def CutImageToFolder(input_folder, output_folder):
    print('---------start cutting----------------------')
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('png', 'jpg', 'jpeg')):
            file_path = os.path.join(input_folder, filename)
            image = Image.open(file_path)  # Mở tệp ảnh
            CutImage(image, output_folder, os.path.splitext(filename)[0])  # Sử dụng tên tệp không có phần mở rộng
    print(f'Images successfully cut and saved to {output_folder}')


# # Ví dụ sử dụng
# input_folder = './Input/'
# output_folder = './Output/'
# CutImageToFolder(input_folder, output_folder)
