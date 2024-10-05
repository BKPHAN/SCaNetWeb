from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import glob
import os
import shutil
from pred_SCD import *
from cut_image import *

app = Flask(__name__)

# Thư mục lưu trữ file tải lên
UPLOAD_FOLDER = 'static/test_dir'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
PRED_DIR = 'static/pred_dir/'


def clear_directory(directory):
    """Clear all contents of the specified directory."""
    if os.path.exists(directory):
        shutil.rmtree(directory)  # Remove the directory and all its contents
        os.makedirs(directory)  # Recreate the directory


# Hàm xử lý trang chủ và upload file
@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        # Clear directories before processing files
        clear_directory(UPLOAD_FOLDER)
        clear_directory(PRED_DIR)

        if request.method == 'POST':
            files = request.files.getlist('files[]')
            file_test_paths = []

            # Tạo thư mục im1 và im2 trong UPLOAD_FOLDER
            im1_folder, im2_folder = create_folders(app.config['UPLOAD_FOLDER'])

            for i, file in enumerate(files):
                if file:
                    # Đặt tên file
                    filename = secure_filename(file.filename)
                    filename = f"{filename.split('.')[0]}{i + 1}.{filename.split('.')[1]}"

                    # Lưu file gốc vào thư mục UPLOAD_FOLDER
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    # file_test_paths.append(os.path.join('test_dir/im' + str(i + 1), filename))
                    file_test_paths.append(os.path.join('test_dir/', filename))

                    # # Mở ảnh từ file
                    # image = Image.open(file_path)
                    #
                    # # Chọn thư mục lưu ảnh cắt (im1 hoặc im2)
                    # if i % 2 == 0:
                    #     CutImage(image, im1_folder, filename)  # Lưu vào im1
                    # else:
                    #     CutImage(image, im2_folder, filename)  # Lưu vào im2
                    # image.close()

            # Call prediction function
            pred_batch_size = 1
            test_dir = UPLOAD_FOLDER
            pred_dir = PRED_DIR
            chkpt_path = './checkpoints/SCanNet_psd_50e_mIoU70.47_Sek18.38_Fscd58.14_OA85.93_Loss0.59.pth'
            predict_main(pred_batch_size, test_dir, pred_dir, chkpt_path)

            # Get paths of predicted files
            # file_pred_paths = get_all_image_files(PRED_DIR)
            file_pred_paths = glob.glob(PRED_DIR + '/*.[pjb]g') + glob.glob(PRED_DIR + '/*.jpeg') + glob.glob(
                PRED_DIR + '/*.png')
            file_pred_paths = ['pred_dir/' + os.path.basename(path) for path in file_pred_paths]

            # Render template with file paths
            return render_template('index.html', file_test_paths=file_test_paths, file_pred_paths=file_pred_paths)

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
