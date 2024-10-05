from PIL import Image
import os


def CutImage(image, output_dir, filename):
    width, height = image.size
    step = 256  # Phần chồng lấp sẽ bằng kích thước của mỗi bước cắt

    for i in range(0, width, step):
        for j in range(0, height, step):
            # Kiểm tra xem có phải phần cuối của chiều rộng và chiều cao hay không
            if i + step > width:
                i = width - step  # Lùi lại để ảnh đủ 256px, nếu cần overlap
            if j + step > height:
                j = height - step  # Lùi lại để ảnh đủ 256px, nếu cần overlap

            # Định nghĩa khung cắt (box)
            box = (i, j, i + step, j + step)
            image_part = image.crop(box)

            # Chuyển đổi ảnh phần cắt sang RGB
            image_part = image_part.convert('RGB')

            # Lưu phần ảnh với tên tệp đã định dạng dựa trên vị trí của nó
            part_filename = f"{filename}_{i}_{j}.png"
            image_part.save(os.path.join(output_dir, part_filename))


# Hàm để hợp nhất các mảnh ảnh thành hình ảnh ban đầu
def merge_images(image_pieces, original_size, output_path):
    # Tạo một ảnh mới có kích thước gốc của ảnh ban đầu
    merged_image = Image.new('RGB', original_size)

    # Duyệt qua từng mảnh ảnh và dán vào vị trí tương ứng
    for (image_piece, (i, j)) in image_pieces:
        merged_image.paste(image_piece, (i, j))

    # Lưu ảnh đã hợp nhất
    merged_image.save(output_path)
