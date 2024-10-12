from PIL import Image
import os


def MergeImage(image_parts, original_size):
    """
    Gộp các phần ảnh đã cắt thành một ảnh duy nhất với kích thước gốc.

    Tham số:
    - image_parts: danh sách đường dẫn tới các phần ảnh.
    - original_size: tuple chứa kích thước gốc của ảnh (width, height).
    """
    original_width, original_height = original_size

    # Khởi tạo ảnh trống với kích thước gốc
    merged_image = Image.new('RGB', (original_width, original_height))

    # Duyệt qua danh sách các phần ảnh
    for part_path in image_parts:
        # Lấy tên tệp và tách lấy tọa độ i, j từ tên tệp (ví dụ: filename_0_0.png)
        part_filename = os.path.basename(part_path)
        parts = part_filename.split('_')

        if len(parts) >= 3:
            i = int(parts[-2])
            j = int(parts[-1].split('.')[0])

            # Mở phần ảnh
            image_part = Image.open(part_path)

            # Ghép phần ảnh vào đúng vị trí của nó trong ảnh gốc
            merged_image.paste(image_part, (i, j))

    # Trả về ảnh đã gộp
    return merged_image

# Ví dụ sử dụng:
# image_parts = ['path/to/image_0_0.png', 'path/to/image_0_256.png', 'path/to/image_256_0.png', ...]
# original_size = (1024, 768)  # Kích thước gốc của ảnh
# merged_image = MergeImage(image_parts, original_size)
# merged_image.save('path/to/output_image.png')
