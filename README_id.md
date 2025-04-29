# Pemeriksaan Fakta Berita Hoax

Alat untuk mendeteksi berita hoax/palsu di Indonesia menggunakan RAG (Retrieval-Augmented Generation).

## Fitur

- Antarmuka CLI untuk pemeriksaan fakta
- Antarmuka web menggunakan Gradio
- Endpoint API melalui FastAPI
- Database vektor (Milvus) untuk menyimpan embedding dokumen
- Integrasi pencarian web (Tavily)

## Instalasi

1. Clone repositori
2. Instal dependensi:
   ```bash
   pip install -r requirements.txt
   ```
3. Siapkan Milvus:
   - Pastikan Anda telah menginstal dan menjalankan Milvus. Lihat dokumentasi Milvus untuk instruksi instalasi.
   - Aplikasi terhubung ke Milvus menggunakan pengaturan default. Jika instance Milvus Anda menggunakan pengaturan yang berbeda, sesuaikan parameter koneksi di `hoax_detect/services/vector_store.py`.

## Penggunaan

### CLI

1.  **Inisialisasi database vektor (Opsional):**
    Jika Anda ingin menggunakan database vektor untuk pemeriksaan fakta, Anda perlu menginisialisasinya terlebih dahulu. Ini melibatkan pemuatan dataset dan pembuatan embedding. Langkah ini hanya diperlukan jika Anda belum menginisialisasi database atau jika Anda ingin menyegarkan data.

    ```bash
    python -m hoax_detect.data.loader --init_db
    ```

2.  **Jalankan antarmuka baris perintah:**

    ```bash
    python -m hoax_detect.cli --query "pertanyaan Anda di sini" --use_vector_db True --use_tavily True
    ```

    -   `--query`: Cuplikan berita atau pernyataan yang ingin Anda periksa faktanya. **Wajib diisi**.
    -   `--use_vector_db`: Nilai boolean yang menunjukkan apakah akan menggunakan database vektor untuk mengambil konteks. Defaultnya adalah `True`.
    -   `--use_tavily`: Nilai boolean yang menunjukkan apakah akan menggunakan Tavily untuk pencarian web. Defaultnya adalah `True`.

    Contoh:

    ```bash
    python -m hoax_detect.cli --query "Jokowi mengundurkan diri" --use_vector_db True --use_tavily True
    ```

### Aplikasi Gradio

1.  **Jalankan aplikasi Gradio:**

    ```bash
    python gradio_app.py
    ```

2.  **Akses aplikasi di browser Anda:**

    Aplikasi akan menyediakan URL lokal (biasanya `http://localhost:7860`) yang dapat Anda gunakan untuk mengakses antarmuka Gradio di browser web Anda.

3.  **Gunakan antarmuka:**

    -   Masukkan cuplikan berita atau pernyataan yang ingin Anda periksa faktanya di bidang input.
    -   Pilih apakah akan menggunakan database vektor dan/atau Tavily untuk pengambilan konteks menggunakan kotak centang yang disediakan.
    -   Klik tombol "Submit" untuk memulai proses pemeriksaan fakta.
    -   Hasilnya, termasuk pernyataan yang telah diperiksa faktanya dan bukti pendukung, akan ditampilkan di area output.

## API

Aplikasi ini juga menyediakan API FastAPI. Lihat `hoax_detect/api.py` untuk detail tentang endpoint yang tersedia. Anda dapat mengakses dokumentasi API di `/docs` setelah menjalankan API.

## Konfigurasi

Pengaturan konfigurasi, seperti jalur dataset, didefinisikan dalam `hoax_detect/config.py`. Anda dapat mengubah pengaturan ini dengan membuat file `.env` di direktori root proyek. Lihat `.env.example` untuk opsi yang tersedia.

## Kontribusi

Kontribusi dipersilakan! Silakan kirim pull request dengan perubahan Anda.
