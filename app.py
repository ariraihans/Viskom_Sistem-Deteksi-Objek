from flask import Flask, render_template, request, jsonify, send_from_directory  # Mengimpor modul Flask untuk membuat aplikasi web
from flask_cors import CORS  # Mengimpor CORS untuk menangani masalah lintas domain
import cv2  # Mengimpor OpenCV untuk pengolahan citra
import numpy as np  # Mengimpor NumPy untuk manipulasi array
from ultralytics import YOLO  # Mengimpor YOLO dari ultralytics untuk deteksi objek
import base64  # Mengimpor modul base64 untuk mengkodekan gambar menjadi string
import os  # Mengimpor os untuk bekerja dengan file dan direktori
from werkzeug.utils import secure_filename  # Mengimpor fungsi secure_filename untuk mengamankan nama file

# Inisialisasi Flask
app = Flask(__name__, template_folder="tampilanweb")  # Membuat instance aplikasi Flask dengan folder template
CORS(app)  # Mengatasi masalah CORS (Cross-Origin Resource Sharing)

UPLOAD_FOLDER = 'static/uploads'  # Menentukan folder tempat menyimpan file yang diupload
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # Menyimpan konfigurasi folder upload
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Membatasi ukuran file yang diupload maksimal 16 MB

# Memuat model YOLO dengan file model yang sudah dilatih
model = YOLO('model_yolov8n.pt')  

@app.route('/')  # Menentukan route untuk halaman utama
def index():
    return render_template('index.html')  # Menampilkan halaman index.html

@app.route('/realtime')  # Menentukan route untuk halaman realtime
def realtime():
    return render_template('realtime.html')  # Menampilkan halaman realtime.html

@app.route('/upload')  # Menentukan route untuk halaman upload
def upload():
    return render_template('upload.html')  # Menampilkan halaman upload.html

@app.route('/detect', methods=['POST'])  # Menentukan route untuk endpoint deteksi kamera
def detect_camera():
    if 'image' not in request.files:  # Memeriksa apakah ada file gambar yang diupload
        return jsonify({'error': 'No image uploaded'}), 400  # Mengembalikan error jika tidak ada gambar

    file = request.files['image']  # Mengambil file gambar dari request
 
    npimg = np.frombuffer(file.read(), np.uint8)  # Mengubah file yang diupload menjadi array NumPy
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)  # Meng-decode gambar menjadi format OpenCV

    # Melakukan deteksi objek dengan model YOLO dan threshold kepercayaan 0.25
    results = model.predict(source=img, conf=0.25) 

    # Menandai objek pada gambar dengan kotak pembatas
    annotated_img = results[0].plot()  

    _, buffer = cv2.imencode('.jpg', annotated_img)  # Mengkodekan gambar yang sudah dianotasi ke format JPG
    encoded_image = base64.b64encode(buffer).decode('utf-8')  # Mengkodekan buffer gambar menjadi string Base64

    return jsonify({'image': encoded_image})  # Mengembalikan gambar yang sudah dianotasi dalam format Base64

@app.route('/detect_upload', methods=['POST'])  # Menentukan route untuk endpoint deteksi gambar yang diupload
def detect_upload():
    if 'image' not in request.files:  # Memeriksa apakah ada file gambar yang diupload
        return jsonify({'error': 'No image uploaded'}), 400  # Mengembalikan error jika tidak ada gambar

    file = request.files['image']  # Mengambil file gambar dari request
    filename = secure_filename(file.filename)  # Mengamankan nama file gambar yang diupload
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # Menentukan path untuk menyimpan file
    file.save(file_path)  # Menyimpan file gambar ke server

    # Membaca gambar dari file yang disimpan
    img = cv2.imread(file_path) 

    # Melakukan deteksi objek dengan model YOLO dan threshold kepercayaan 0.25
    results = model.predict(source=img, conf=0.25)  

    # Menandai objek pada gambar dengan kotak pembatas
    annotated_img = results[0].plot()  

    result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + filename)  # Menentukan path untuk menyimpan gambar hasil deteksi
    cv2.imwrite(result_path, annotated_img)  # Menyimpan gambar yang sudah dianotasi ke server

    _, buffer = cv2.imencode('.jpg', annotated_img)  # Mengkodekan gambar hasil deteksi ke format JPG
    encoded_image = base64.b64encode(buffer).decode('utf-8')  # Mengkodekan buffer gambar menjadi string Base64

    return jsonify({'image': encoded_image, 'result_path': result_path})  # Mengembalikan gambar hasil deteksi dan path gambar

@app.route('/uploads/<filename>')  # Menentukan route untuk mengakses file yang diupload
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)  # Mengirimkan file yang diminta dari folder upload

if __name__ == '__main__':  # Memastikan aplikasi berjalan jika file ini dieksekusi langsung
    app.run(debug=True)  # Menjalankan aplikasi Flask dalam mode debug
