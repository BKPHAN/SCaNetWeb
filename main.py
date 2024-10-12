from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import glob
import os
import shutil
from pred_SCD import *
from cut_image import *
from merge_image import *

app = Flask(__name__)

# Thư mục lưu trữ file tải lên
UPLOAD_FOLDER = 'static/test_dir/'
INPUT_FOLDER = 'static/input_dir/'
PRED_DIR = 'static/pred_dir/'
OUTPUT = 'static/output/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def clear_directory(directory):
    """Clear all contents of the specified directory."""
    if os.path.exists(directory):
        shutil.rmtree(directory)  # Remove the directory and all its contents
        os.makedirs(directory)  # Recreate the directory
    else:
        os.makedirs(directory)  # Create the directory


# Hàm xử lý trang chủ và upload file
@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        # Clear directories before processing files
        clear_directory(UPLOAD_FOLDER)
        clear_directory(PRED_DIR)
        clear_directory(INPUT_FOLDER)
        clear_directory(OUTPUT)

        if request.method == 'POST':
            files = request.files.getlist('files[]')
            input_paths = []
            output_path = []
            original_image_sizes = {}

            # Tạo thư mục im1 và im2 trong UPLOAD_FOLDER
            im1_folder, im2_folder = create_folders(app.config['UPLOAD_FOLDER'])
            # Tạo thư mục im1 và im2 trong UPLOAD_FOLDER
            input1_folder, input2_folder = create_folders(INPUT_FOLDER)

            for i, file in enumerate(files):
                if file:
                    # Đặt tên file
                    filename = secure_filename(file.filename)

                    # Lưu file tạm thời để kiểm tra kích thước
                    if i == 0:
                        temp_path = os.path.join(input1_folder, filename)
                        input_paths.append(temp_path.replace('static/', '').replace('\\', '/'))
                        file.save(temp_path)
                    else:
                        temp_path = os.path.join(input2_folder, filename)
                        input_paths.append(temp_path.replace('static/', '').replace('\\', '/'))
                        file.save(temp_path)


                    # Mở ảnh để kiểm tra kích thước
                    with Image.open(temp_path) as img:
                        width, height = img.size
                        original_image_sizes[filename] = (width, height)  # Lưu kích thước ảnh ban đầu
                        print(f"Kích thước ảnh: {width}x{height}")

                        # Nếu kích thước > 256x256, cắt ảnh
                        if width > 256 or height > 256:
                            if i == 0:
                                file_path = os.path.join(im1_folder, filename)
                            else:
                                file_path = os.path.join(im2_folder, filename)

                            # Gọi hàm cắt ảnh và lưu các phần ảnh cắt vào file_path
                            CutImage(img, os.path.dirname(file_path), os.path.splitext(filename)[0])
                        else:
                            # Lưu file vào thư mục im1 hoặc im2 nếu kích thước không cần cắt
                            if i == 0:
                                file_path = os.path.join(im1_folder, filename)
                                # file_test_paths.append(os.path.join('test_dir/im1/', filename))
                            else:
                                file_path = os.path.join(im2_folder, filename)
                                # file_test_paths.append(os.path.join('test_dir/im2/', filename))

                            file.save(file_path)

            # Call prediction function
            pred_batch_size = 1
            test_dir = UPLOAD_FOLDER
            pred_dir = PRED_DIR
            chkpt_path = './checkpoints/SCanNet_psd_47e_mIoU84.91_Sek48.89_Fscd84.38_OA94.85_Loss0.22_adam.pth'
            predict_main(pred_batch_size, test_dir, pred_dir, chkpt_path)

            # Lấy toàn bộ thư mục con trong PRED_DIR
            subdirs = [os.path.join(PRED_DIR, d) for d in os.listdir(PRED_DIR) if
                       os.path.isdir(os.path.join(PRED_DIR, d))]

            # Duyệt qua từng thư mục con
            for i, subdir in enumerate(subdirs):
                # Lấy đường dẫn của tất cả các tệp ảnh dự đoán trong thư mục con
                file_pred_paths = glob.glob(subdir + '/**/*.[pjb]g', recursive=True) + \
                                  glob.glob(subdir + '/**/*.jpeg', recursive=True) + \
                                  glob.glob(subdir + '/**/*.png', recursive=True)

                # Làm sạch đường dẫn tệp
                # file_pred_paths = [path.replace('static/', '').replace('\\', '/') for path in file_pred_paths]

                # tim kích thước ảnh
                original_size = original_image_sizes[secure_filename(files[i].filename)]

                # Gộp các phần ảnh lại thành kích thước gốc
                merged_image = MergeImage(file_pred_paths, original_size)

                # Lưu ảnh đã gộp lại với tên tệp gốc
                merged_filename = secure_filename(file.filename)
                output_path.append(f"{subdir.replace('static/', '')}/{merged_filename}")
                merged_image.save(os.path.join(subdir, merged_filename))

            # Render template with file paths
            return render_template('index.html', file_test_paths=input_paths, file_pred_paths=output_path)

        return render_template('index.html')
    except Exception as e:
        return f"An error occurred: {str(e)}"


# Hàm tạo thư mục nếu chưa tồn tại
def create_folders(upload_folder):
    im1_path = os.path.join(upload_folder, 'im1')
    im2_path = os.path.join(upload_folder, 'im2')

    if not os.path.exists(im1_path):
        os.makedirs(im1_path)
    if not os.path.exists(im2_path):
        os.makedirs(im2_path)

    return im1_path, im2_path


# Hàm để lấy tất cả các file ảnh từ thư mục và các thư mục con
def get_all_image_files(directory):
    image_files = []
    # Duyệt qua tất cả các thư mục và file trong cây thư mục
    for root, dirs, files in os.walk(directory):
        # Lấy tất cả các file có đuôi .jpg, .jpeg, .png
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            image_files.extend(glob.glob(os.path.join(root, ext)))
    return image_files


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
