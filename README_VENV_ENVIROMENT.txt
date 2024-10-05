B1: cd đến thư mục của project hoặc dùng terminal trong pychamr
run: python -m venv venv
B2: active venv
WinDows run: venv\Scripts\activate
linus run: source venv/bin/activate
B3: cài thư viện vào venv:
run: pip install <tên-thư-viện>
 *  neu da co file requirements.txt 
run: pip install -r requirements.txt

B4: (nếu có pycharm và chạy pycharm với môi trường ảo)

# Đi đến File -> Settings (hoặc PyCharm -> Preferences trên macOS).
# Trong phần Project: <Tên-Dự-Án> -> Python Interpreter, bạn sẽ thấy danh sách các interpreter hiện có.
# Nhấn vào biểu tượng bánh răng (cogwheel) và chọn Add hoặc chữ Add Interpreter chọn Add Local Interpreter .
# Chọn Existing environment và nhấp vào ... để tìm đến file python trong thư mục venv mà bạn đã tạo:
	- Trên Windows: venv\Scripts\python.exe
	-Trên macOS/Linux: venv/bin/python
#Chọn interpreter này và nhấn OK.

B5: chạy file
# nếu chạy bằng cmd hay terminal: python ten_file.py (ex: python main.py)
# nếu chạy bằng pycharm thì ấn vào dấu mũi tên hoặc shift+f10