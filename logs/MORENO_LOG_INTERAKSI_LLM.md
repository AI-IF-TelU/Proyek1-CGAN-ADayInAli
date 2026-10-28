# LOG INTERAKSI LLM — KELOMPOK A Day in Ali (Moreno Cello Rhyrhm)

- **Anggota:** Moreno Cello Rhyrhm
- **NIM:** 103012340256
- **Peran / Job Desk:** Prompt Engineer & LLM Navigator (Ketua)
- **Tanggal Sesi:** 25/10/2026 – 28/10/2026
- **Target Modul:** Riset metode, setup repository, review implementasi cGAN, verifikasi API PyTorch/torchvision, report figures, README, references, dan dokumentasi log.
- **Model AI / Environment:** Reasonix coding agent (LLM); Linux, Python 3.14.8, PyTorch 2.14.1 (CPU), uv 0.13.6. Notebook dijalankan manual di kernel Jupyter **"GenAI +cuda"** dengan PyTorch 2.14.1+cu130. `torchvision` belum terpasang di environment sandbox.
- **Tautan Sesi Chat:** Tidak ada share link publik. Log ini adalah ekspor Markdown dari sesi Reasonix; sesuai panduan, ekspor Markdown diperbolehkan.

> **Catatan ekspor:** Log ini merangkum seluruh sesi perancangan, debugging,
> optimasi, verifikasi dokumentasi, dan penyusunan deliverables. Prompt ditulis
> ulang secara ringkas dan profesional; perintah, error, keputusan, commit, dan
> metrik dipertahankan.

## BAGIAN 1: RINGKASAN TERSTRUKTUR SESUAI RUBRIK

### Sesi 1 — Riset metode, template, dan kelayakan (25/10/2026)

#### 1. TAHAP INISIASI
- **Prompt Kelompok:** "Riset apakah Conditional GAN tepat untuk proyek Fashion-MNIST ini. Apakah saya perlu mencari template repo di Google? Jika ya, template apa yang boleh dipakai sebagai base?"
- **Alasan Khusus:** Panduan proyek mewajibkan metode *from scratch* dan melarang copy-paste tutorial. Kami perlu memastikan cGAN memang metode yang disetujui, sekaligus mencari batas antara "referensi konsep" dan "menyalin kode".
- **Output Model:** cGAN adalah pilihan yang tepat karena panduan secara eksplisit mencantumkan **Conditional GAN Klasik** dengan Fashion-MNIST. Tidak ada starter template cGAN yang aman untuk dipakai sebagai base; repo cGAN di GitHub dan tutorial LearnOpenCV/Medium/YouTube adalah tutorial, bukan referensi API. Base yang dipakai:
  - Paper cGAN: Mirza & Osindero, *Conditional Generative Adversarial Nets*, arXiv:1411.1784.
  - Paper GAN: Goodfellow et al., arXiv:1406.2661.
  - Dataset: Fashion-MNIST, arXiv:1708.07747 dan repo Zalando.
  - Dokumentasi resmi PyTorch/torchvision untuk fungsi dasar.

#### 2. TAHAP PENANGANAN EROR / DEBUGGING
- **Temuan:** Arsitektur awal di README memakai `concat(128) -> Linear -> 7x7x256`, yang berarti satu layer Linear saja sekitar 1,6 juta parameter; total model sekitar **2,87 juta** parameter. Ini terlalu besar untuk target ~1 juta dan akan memperlambat training CPU.
- **Prompt Perbaikan:** "Cek apakah arsitektur generator/discriminator di README realistis untuk tiga hari dan target ~1 juta parameter. Jika tidak, kecilkan tanpa mengubah ide cGAN."
- **Analisis & Solusi:** Arsitektur diperkecil menjadi:
  - Generator: `z=64` + label embedding `32` -> concat `96` -> `Linear 7x7x128` -> `ConvTranspose 14x14x64` -> `ConvTranspose 28x28x1` -> `Tanh`.
  - Discriminator: `1x28x28` + label map -> concat `2x28x28` -> Conv `32` -> `64` -> `128` -> `Linear -> logit`.
  - Total terverifikasi: **741.249 + 192.881 = 934.130 parameter**; output G `(N,1,28,28)` dan D `(N,1)`.
- **Pelajaran:** Mengecilkan model sejak awal lebih baik daripada mengejar kapasitas yang tidak muat di budget CPU.

#### 3. TAHAP OPTIMASI
- **Prompt:** "Buat scope tiga hari yang realistis; jangan menambah arsitektur atau output baru."
- **Analisis & Solusi:** Scope dikunci ke satu arsitektur cGAN, satu dataset, dan maksimal dua varian hyperparameter. FID dihapus karena Inception terlalu berat; evaluasi memakai conditional accuracy, prediction entropy, dan diversity ratio. Target hardware diturunkan menjadi 2-core/4 GB (minimum) dan 4-core/8 GB (rekomendasi), dengan batch 128 dan fallback 64/32.

### Sesi 2 — Scaffold repository, README, dan struktur folder (25/10/2026)

#### 1. TAHAP INISIASI
- **Prompt Kelompok:** "Setup folder project dulu, lalu buat README dari template BLANK_README, masukkan struktur folder, model cGAN, dataset, team, dan workflow. Jangan commit README sebelum saya review."
- **Alasan Khusus:** Repository perlu struktur yang jelas sebelum kode model ditulis, dan README adalah deliverable yang dibaca dosen.
- **Output Model:** Dibuat struktur:
  - `data/raw/`, `notebooks/`, `src/`, `tests/`, `logs/`, `references/`, `docs/`, `reports/figures/`, `outputs/checkpoints|samples|logs/`.
  - `.gitkeep` untuk folder kosong yang perlu dilacak.
  - README berisi model, dataset, setup, usage, minimum hardware, team, workflow, roadmap, dan later references.
  - `.gitignore` untuk `data/`, `outputs/`, `*.pt`, `*.pth`, `.venv`, cache, dan file lokal.

#### 2. TAHAP PENANGANAN EROR / DEBUGGING
- **Temuan:** README menampilkan `AGENTS.md` dan `GIT_COMMIT_GUIDE.md` sebagai bagian struktur, padahal keduanya file lokal yang tidak boleh ikut repo. README juga sempat menyebut target 24 thread/32 GB dan rencana tiga hari.
- **Prompt Perbaikan:** "Hapus file lokal dari README; README untuk dosen/audiens, bukan catatan developer. Jangan sebut 24 thread atau rencana tiga hari di README."
- **Analisis & Solusi:** File lokal dihapus dari tree README. `AGENTS.md`, `GIT_COMMIT_GUIDE.md`, template README, dan panduan proyek dimasukkan ke `.gitignore`. README diubah ke bahasa audiens dengan minimum 2-core/4 GB, rekomendasi 4-core/8 GB, dan tanpa detail mesin developer. Catatan tiga hari tetap hanya di `AGENTS.md` lokal.

#### 3. TAHAP OPTIMASI
- **Prompt:** "Pastikan README tidak menjanjikan fitur yang belum ada; tandai notebook 01 sebagai planned dan notebook 02 sebagai aktif."
- **Analisis & Solusi:** README memakai status "implemented and tested" hanya untuk komponen yang memang ada; notebook 01 tetap `planned; not implemented`, notebook 02 menjadi notebook aktif. Instalasi memakai CPU wheel yang lebih kecil, dengan catatan opsional untuk CUDA. README juga menjelaskan `outputs/` diabaikan Git.

### Sesi 3 — Git initialization, commit pertama, dan debugging push (25/10/2026)

#### 1. TAHAP INISIASI
- **Prompt Kelompok:** "Commit scaffold dan siapkan push ke GitHub. Remote-nya SSH."
- **Alasan Khusus:** Commit per anggota dinilai; repository harus memiliki commit sebelum branch bisa di-push.
- **Output Model:** Commit awal `ab994d3 Initialize project scaffold` berisi `.gitignore`, `README.md`, dan `.gitkeep`. Remote `origin` adalah `git@github.com:AI-IF-TelU/Proyek1-CGAN-ADayInAli.git`.

#### 2. TAHAP PENANGANAN EROR / DEBUGGING
- **Pesan Eror:** `error: src refspec refs/heads/main does not match any` saat `git push`.
- **Prompt Perbaikan:** "Kenapa push gagal? Saya kira ini masalah SSH/akun."
- **Analisis & Solusi:** Error bukan karena SSH dan bukan karena README. Repository belum punya commit, sehingga branch `main` belum ada. Solusinya membuat commit awal lebih dulu. Remote sudah SSH dan tidak diubah. README clone URL tetap HTTPS agar pembaca bisa clone; URL README tidak memengaruhi `git push`.
- **Pesan Eror Tambahan:** Di sandbox, `git ls-remote` menghasilkan `Bad owner or permissions on /etc/ssh/ssh_config.d/20-systemd-ssh-proxy.conf`. Ini masalah konfigurasi SSH di sandbox, bukan repository. Sandbox tidak memiliki akses ke SSH agent/key pengguna, jadi push harus dijalankan dari terminal pengguna.
- **Pelajaran:** `git push` membutuhkan commit; jangan menyalahkan remote/SSH sebelum memeriksa `git log` dan `git status`.

#### 3. TAHAP OPTIMASI
- **Prompt:** "Pastikan commit mengikuti GIT_COMMIT_GUIDE dan file lokal tidak ikut ter-commit."
- **Analisis & Solusi:** Commit memakai pesan imperatif, subject singkat, dan body yang menjelaskan alasan. File lokal (`AGENTS.md`, panduan, template, commit guide) tetap di `.gitignore`. Push dilakukan dari terminal pengguna yang memiliki SSH.

### Sesi 4 — Pull implementasi tim, review, dan baseline (28/10/2026)

#### 1. TAHAP INISIASI
- **Prompt Kelompok:** "Saya akan pull perubahan tim. Setelah pull, jangan asumsikan rencana lama masih benar; baca repo dan update AGENTS.md."
- **Alasan Khusus:** Tim lain sudah mengimplementasikan model, tests, notebook, dan logs; rencana scaffold bisa berubah total.
- **Output Model:** Pull menambahkan:
  - `src/config.py`, `data.py`, `models.py`, `train.py`, `sample.py`, `evaluate.py`, `plots.py`, `utils.py`.
  - `tests/test_data.py`, `test_models.py`, `test_train.py`, `test_sample_evaluate.py`.
  - `notebooks/02-ad-cgan-experiments.ipynb`.
  - Logs Ali, Davu, Maliki.
  - README dengan Results.
- **Review:** Model terverifikasi 741.249 + 192.881 = 934.130 parameter. Baseline 30 epoch; metrics awal dari CPU run: conditional 0.912, entropy 0.998, diversity 0.95.

#### 2. TAHAP PENANGANAN EROR / DEBUGGING
- **Pesan Eror:** `ModuleNotFoundError: No module named 'torchvision'` saat `python -m unittest discover -s tests`.
- **Prompt Perbaikan:** "Verifikasi apa yang bisa dijalankan tanpa torchvision dan apa yang perlu setup."
- **Analisis & Solusi:** Model tests dan `compileall` lulus; test data/train/sample/evaluate memerlukan torchvision. Setup dinyatakan belum lengkap di sandbox. Notebook dijalankan manual di kernel pengguna yang memiliki torchvision/CUDA.
- **Temuan Dokumentasi:** README masih menunjuk `logs/LOG_INTERAKSI_LLM.md` yang tidak ada; log yang ada adalah per anggota. Log Moreno belum ada. README masih memakai metrik CPU lama. Semua dicatat untuk perbaikan.
- **Pelajaran:** Jangan mengklaim full suite lulus jika dependency environment belum tersedia; bedakan environment sandbox dan environment pengguna.

#### 3. TAHAP OPTIMASI
- **Prompt:** "Update AGENTS.md dengan kondisi pasca-pull; jangan menulis kode baru sebelum rencana direkonsiliasi."
- **Analisis & Solusi:** `AGENTS.md` diperbarui: current HEAD, model 0.93M, tests, logs per anggota, torchvision missing, dan catatan re-baseline setelah pull. Repo di-review tanpa membuat perubahan yang tidak diminta.

### Sesi 5 — Notebook run, report figures, dan checkpoint resume (28/10/2026)

#### 1. TAHAP INISIASI
- **Prompt Kelompok:** "Saya sudah run notebook 02. Output-nya mengikuti checkpoint epoch 30. Apakah ini aman? Jika saya ingin epoch baru, apakah cukup hapus? Pastikan perubahan kode tetap bekerja dan output bisa dilihat dengan checkpoint."
- **Alasan Khusus:** Pengguna tidak ingin retrain jika checkpoint sudah ada; output dan report figures tetap harus dihasilkan.
- **Output Model:** Notebook `RESUME=True` memuat `outputs/checkpoints/ckpt.pt` di epoch 30, menampilkan `Resumed from ... at epoch 30`, `Epochs recorded: 30`, dan **0 error**. Evaluasi menghasilkan:
  - Classifier accuracy on real: **0.9129**
  - Conditional accuracy: **0.9170**
  - Prediction entropy: **0.9986**
  - Mean diversity ratio: **0.9448**

#### 2. TAHAP PENANGANAN EROR / DEBUGGING
- **Temuan:** Karena checkpoint sudah di epoch 30 dan `EPOCHS=30`, loop training tidak berjalan. Ini bukan error; `train()` memuat history dan melewati training. Namun output figure tetap harus diperbarui.
- **Prompt Perbaikan:** "Pastikan copy report figures dan evaluasi tetap berjalan meski training di-skip; jelaskan cara mulai fresh."
- **Analisis & Solusi:**
  - `src/train.py` menyalin loss curve dan final sample grid ke `reports/figures/` setelah loop; ctime file 22:38 membuktikan copy berjalan saat resume.
  - `src/evaluate.py` menyalin confusion matrix final; file `confusion_matrix_final.png` juga diperbarui pada 22:38.
  - Notebook menyimpan `real_samples.png`, `sample_grid_progress.png`, `fresh_samples_per_class.png`, dan `evaluation_metrics.png` lewat `savefig`.
  - Untuk fresh run, `outputs/` harus dihapus. Untuk melanjutkan, naikkan `EPOCHS` (mis. 40) dan pertahankan `ckpt.pt`.

#### 3. TAHAP OPTIMASI
- **Prompt:** "Dokumentasikan fresh run vs resume; jangan menambah output baru, gunakan figure yang sudah ada."
- **Analisis & Solusi:** Report figures memakai tujuh figure yang sudah ada: loss curve, final grid, confusion matrix, real samples, grid progress, fresh per-class samples, dan evaluation metrics. Tidak ada plot type baru. README dan AGENTS mencatat perilaku resume dan cara membersihkan `outputs/`.

### Sesi 6 — Verifikasi dokumentasi Context7 dan hardening kode (28/10/2026)

#### 1. TAHAP INISIASI
- **Prompt Kelompok:** "Gunakan Context7 MCP untuk memastikan semua kode Python dan notebook efisien serta mengikuti dokumentasi Python dan library-nya. Verifikasi semuanya dan commit sepanjang proses."
- **Alasan Khusus:** Review manual saja tidak cukup; API seperti `torch.load`, `DataLoader`, `save_image`, dan Matplotlib harus dicek terhadap dokumentasi resmi.
- **Output Model:** Context7 dipakai untuk:
  - `/pytorch/pytorch/v2.14.1` — `torch.load(weights_only=...)`, reproducibility, DataLoader.
  - `/pytorch/vision` — transform dan `save_image`.
  - `/python/cpython` — referensi Python stdlib.
  - Dokumentasi resmi Matplotlib untuk `Axes.set_xticks`.
  - Dokumentasi resmi Python untuk `Path.write_text(..., encoding=...)`.

#### 2. TAHAP PENANGANAN EROR / DEBUGGING
- **Temuan 1:** `src/utils.py` memakai `torch.load(..., weights_only=False)`. Dokumentasi PyTorch menyarankan `weights_only=True` untuk checkpoint yang berisi state dict dan primitive.
  - **Prompt Perbaikan:** "Cek apakah checkpoint nyata bisa dimuat dengan weights_only=True; jika ya, ubah dan uji."
  - **Analisis & Solusi:** Checkpoint nyata `outputs/checkpoints/ckpt.pt` berhasil dimuat dengan `weights_only=True` (epoch 30, 30 history entries, 741.249 generator params). Kode dan test diubah; commit `cdd5e5f`.
- **Temuan 2:** `float(out.min())` pada tensor `requires_grad` menghasilkan warning.
  - **Prompt Perbaikan:** "Hilangkan warning dengan cara yang benar."
  - **Analisis & Solusi:** Diganti `out.detach().min().item()`; model tests lulus tanpa warning.
- **Temuan 3:** DataLoader workers belum menyeed RNG Python/NumPy.
  - **Prompt Perbaikan:** "Ikuti panduan reproducibility PyTorch untuk multi-worker loading."
  - **Analisis & Solusi:** Ditambahkan `_seed_worker` dan `worker_init_fn`; commit `23e48f9`.
- **Temuan 4:** JSON logs belum memakai encoding eksplisit.
  - **Prompt Perbaikan:** "Buat I/O JSON konsisten lintas platform."
  - **Analisis & Solusi:** Ditambahkan `encoding="utf-8"` pada write/read; commit `4f05113`.
- **Temuan 5:** CLI menerima nilai nol yang bisa menyebabkan error (mis. `--sample-every 0`).
  - **Prompt Perbaikan:** "Validasi argumen numerik sebelum masuk loop."
  - **Analisis & Solusi:** Ditambahkan validasi batch size, sample interval, max batches, per-class, dan classifier epochs; commit `032d4af`.
- **Temuan 6:** `git diff --cached --check` menemukan trailing whitespace pada dua baris header `references/README.md`.
  - **Prompt Perbaikan:** "Perbaiki whitespace tanpa menghilangkan hard break Markdown."
  - **Analisis & Solusi:** Dua spasi di akhir baris diganti `<br>`; commit referensi berhasil.

#### 3. TAHAP OPTIMASI
- **Prompt:** "Catat hasil verifikasi dokumentasi dan pastikan tidak ada perubahan output type baru."
- **Analisis & Solusi:**
  - `torch.manual_seed` sudah menyeed CPU dan CUDA; tidak perlu `cuda.manual_seed_all` tambahan.
  - `save_image(..., normalize=True, value_range=(-1, 1))` sesuai dokumentasi `make_grid`.
  - `set_xticks(ticks, labels, **kwargs)` sesuai Matplotlib.
  - `Path.write_text(..., encoding=...)` sesuai Python docs.
  - Model tests dan `compileall` lulus; full suite tetap tertahan torchvision di sandbox.
  - Commit: `cdd5e5f`, `23e48f9`, `4f05113`, `032d4af`.

### Sesi 7 — References, README latest run, dan log export (28/10/2026)

#### 1. TAHAP INISIASI
- **Prompt Kelompok:** "Buat references yang rapi; update README berdasarkan run terbaru; jelaskan bahwa fresh run harus membersihkan output; notebook 01 tetap planned, notebook 02 sudah memuaskan."
- **Alasan Khusus:** README harus sesuai output nyata, bukan rencana lama; referensi harus bisa dipakai dosen tanpa menyalin kode tutorial.
- **Output Model:**
  - `references/README.md` berisi paper cGAN/GAN/Fashion-MNIST, API docs, tabel penggunaan, aturan download, dan BibTeX.
  - README memakai metrik CUDA terbaru: 0.9170 conditional, 0.9129 classifier, 0.9986 entropy, 0.9448 diversity.
  - README menambahkan bagian **Fresh run vs. resume**, reproducibility, references, dan status notebook 01/02.
  - Log ini (`logs/MORENO_LOG_INTERAKSI_LLM.md`) dibuat sebagai ekspor Markdown lengkap.

#### 2. TAHAP PENANGANAN EROR / DEBUGGING
- **Temuan 1:** README masih memakai metrik CPU lama dan menyebut `logs/LOG_INTERAKSI_LLM.md` yang tidak ada.
  - **Prompt Perbaikan:** "Ganti dengan metrik run terbaru dan log per anggota yang benar."
  - **Analisis & Solusi:** Results, per-epoch table, per-class notes, dan D accuracy diperbarui dari `outputs/logs/eval_ckpt.json`, `eval_ckpt_epoch_*.json`, dan `history.json`. Tree logs menampilkan ALI/DAVU/MALIKI; unified log dicatat belum ada.
- **Temuan 2:** Notebook dapat resume dari checkpoint dan melewati training, tetapi pengguna tidak yakin output tetap dihasilkan.
  - **Prompt Perbaikan:** "Jelaskan resume vs fresh run dan buktikan output."
  - **Analisis & Solusi:** Notebook 0 error; metrics 0.9170; report figures diperbarui; `outputs/` tetap checkpoint epoch 30 (mtime 22:09); fresh run memerlukan `rm -rf outputs`.
- **Temuan 3:** `references/README.md` memiliki trailing whitespace pada header.
  - **Prompt Perbaikan:** "Perbaiki agar `git diff --check` bersih."
  - **Analisis & Solusi:** `<br>` dipakai; commit referensi `4f7dea3` bersih.
- **Temuan 4:** Notebook 02 bagian Observations masih menyebut CPU/0.912, sedangkan README memakai CUDA/0.917.
  - **Status:** Dicatat sebagai pekerjaan berikutnya; notebook 02 tidak diubah karena pengguna meminta README lebih dulu dan menyatakan sudah puas dengan notebook 02.

#### 3. TAHAP OPTIMASI
- **Prompt:** "Verifikasi link references, fokus README, dan jangan commit notebook/README sebelum review."
- **Analisis & Solusi:**
  - Semua 26 link references mengembalikan HTTP 200.
  - `references/README.md` di-commit sebagai `4f7dea3`.
  - README, notebook, dan empat report figures yang diperbarui tetap uncommitted untuk review.
  - Branch `main` ahead 8 sebelum log ini (9 setelah commit log); commit kode/referensi selesai, push dilakukan dari terminal pengguna.

## BAGIAN 2: KRONOLOGI COMMIT & ARTEFAK

| Tanggal | Commit / Artefak | Isi |
| --- | --- | --- |
| 25/10/2026 | `ab994d3` | Initial scaffold: `.gitignore`, README, `.gitkeep` |
| 28/10/2026 | `96ed552` | README title update dari tim |
| 28/10/2026 | `3e8ccda` | Hapus duplikat `LOG_INTERAKSI_ALI.md` |
| 28/10/2026 | `d945f0c` | Copy loss curve, final grid, confusion matrix ke `reports/figures/` |
| 28/10/2026 | `173e68e` | Empat figure notebook ke `reports/figures/` |
| 28/10/2026 | `cdd5e5f` | `weights_only=True` + warning test |
| 28/10/2026 | `23e48f9` | Seed DataLoader workers |
| 28/10/2026 | `4f05113` | UTF-8 JSON logs |
| 28/10/2026 | `032d4af` | Validasi CLI counts |
| 28/10/2026 | `4f7dea3` | `references/README.md` + BibTeX |

**Status repository:** 26 commit sebelum log ini; setelah log ini di-commit menjadi 27. `main` ahead 8 sebelum, 9 setelah log ini. README, notebook 02, dan empat report figures masih uncommitted untuk review; final report dan unified log belum dibuat.

**Metrik baseline terbaru:** CUDA, 30 epoch, batch 128, lr 2e-4; classifier real 0.9129; conditional 0.9170; entropy 0.9986; diversity 0.9448.

## BAGIAN 3: PELAJARAN & STATUS AKHIR

### Pelajaran utama
1. **Metode:** cGAN adalah pilihan yang disetujui panduan; paper dan API docs boleh dibaca, tetapi kode tutorial tidak boleh disalin.
2. **Ukuran model:** 0.93M parameter cukup; arsitektur 2.87M terlalu berat dan tidak perlu.
3. **Portabilitas:** minimum 2-core/4 GB, rekomendasi 4-core/8 GB; detail mesin developer tidak masuk README.
4. **Git:** commit dulu sebelum push; error `refspec` berarti belum ada commit, bukan masalah SSH. Sandbox tidak memiliki SSH agent; push dari terminal pengguna.
5. **Checkpoint:** `RESUME=True` memuat `ckpt.pt`; `EPOCHS=30` dengan checkpoint epoch 30 tidak retrain, tetapi evaluasi dan report figures tetap berjalan. Fresh run membutuhkan `rm -rf outputs`; melanjutkan training cukup menaikkan `EPOCHS`.
6. **Reproducibility:** seed Python/NumPy/PyTorch, worker seeding, fixed grid, `weights_only=True`, UTF-8 JSON.
7. **Dokumentasi:** `references/README.md` link-only dan BibTeX; log per anggota diperlukan; unified log dan final report masih kurang.
8. **Verifikasi:** Context7 dan dokumentasi resmi menangkap `weights_only`, `value_range`, `set_xticks`, dan `encoding`; full unittest tetap membutuhkan `torchvision`.

### Status akhir
- Implementasi, tests, notebook 02, references, dan report figures selesai.
- Baseline CUDA 30 epoch terekam; metrik terbaru 0.9170 conditional accuracy.
- `references/README.md` committed; kode hardening committed.
- README diperbarui ke run terbaru dan memuat instruksi fresh run.
- Notebook 01 tetap planned; notebook 02 aktif.
- Log Moreno selesai (file ini); yang masih kurang: unified `LOG_INTERAKSI_LLM.md` dan final report.
- README, notebook 02, dan empat report figures perlu di-commit bersama setelah review.
