# LOG INTERAKSI LLM — Mohamed Ali Ibrahim

**Mata Kuliah:** CAK4KBB3 — Vibe Coding Generative AI from Scratch  
**Arsitektur:** Class-Conditional GAN (cGAN) on Fashion-MNIST  

## Data Anggota
| Nama Anggota | NIM | Peran / Job Desk |
| :--- | :--- | :--- |
| Mohamed Ali Ibrahim | 103012350550 | Lead Documentarian & Error Handler |

---

## SESI 4: Evaluasi, Unit Testing & Error Handling
- **Tanggal Sesi:** 2026-10-26  
- **Target Modul:** `src/evaluate.py`, `tests/test_sample_evaluate.py`  
- **Tautan Sesi Chat:** [PASTE_YOUR_CHAT_LINK_HERE]

### KRONOLOGI PROMPT & RESPONS:

1. **TAHAP INISIASI (Meminta Kode Awal)**
   - **Prompt Kelompok:** "Rancang skrip evaluasi src/evaluate.py dan test suite tests/test_sample_evaluate.py untuk cGAN Fashion-MNIST. Latih 2-layer CNN reference classifier pada data real untuk mengukur conditional accuracy, prediction entropy, dan feature-space diversity ratio dari sampel yang dihasilkan Generator."
   - **Alasan Khusus:** Menghindari penggunaan FID yang terlalu berat untuk CPU training, sekaligus menyediakan metrik kuantitatif alternatif untuk memverifikasi bahwa Generator mematuhi class label tanpa mengalami mode collapse.
   - **Output Model:** Fungsi evaluasi kuantitatif (`predict`, `embed`, `confusion_matrix`, `normalized_entropy`, `diversity_ratios`) dan class `Classifier` berbasis `nn.Sequential`.

2. **TAHAP PENANGANAN EROR / DEBUGGING**
   - **Pesan Eror dari Notebook/Terminal:** `AttributeError: 'Classifier' object has no attribute 'embed'` saat menjalankan unit test `test_diversity_detects_collapse` pada `tests/test_sample_evaluate.py`.
   - **Prompt Perbaikan ke LLM:** "Unit test gagal karena Classifier di evaluate.py hanya memiliki method forward() ke 10-class logits, sehingga diversity_ratios() tidak bisa mengambil 128-dim feature representation. Tambahkan method embed() ke Classifier."
   - **Analisis & Solusi Kelompok:** Menambahkan method `embed(self, x)` pada `Classifier` yang mengembalikan output layer `self.features(x)` (128-dim tensor) tanpa melewati classification head (`self.head`), memungkinkan kalkulasi *mean pairwise feature distance*.

3. **TAHAP OPTIMASI**
   - **Prompt Kelompok:** "Bagaimana cara mencegah CNN classifier melatih ulang bobotnya (6 epoch) setiap kali src.evaluate atau unit test dijalankan?"
   - **Analisis & Solusi Kelompok:** Mengimplementasikan `load_or_train_classifier()` dengan caching bobot ke `outputs/checkpoints/classifier.pt`. Jika file checkpoint ada, script langsung meload `state_dict`, memotong waktu evaluasi dari ~90 detik menjadi <2 detik.