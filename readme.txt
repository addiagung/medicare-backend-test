=== Petunjuk Penggunaan ===

1. Membuat lingkungan virtual
    python -m venv venv
    .\venv\Scripts\activate

2. Instalasi Dependensi
    pip install Flask
    pip install numpy
    pip install Pillow
    pip install pandas
    pip install tensorflow

3. Menjalankan Kode
    Flask run

4. Endpoint API
    Method: POST
    Request URL: http://localhost:5000/detection
    Body:
        Key: Image
        Value: Select Files -> Untuk upload gambar deteksi

5. Catatan
    Pastikan Python dan pip telah terinstal sebelum memulai.
    Saat ini dari model ML hanya mendeteksi 2 macam obat (untuk percobaan) yaitu bodrex dan tolak angin.



