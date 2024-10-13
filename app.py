import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def add_custom_css():
    st.markdown("""
    <style>
        /* Background color */
        .stApp {
            background-color: #165136;
        }

        /* Text color */
        .stTextInput, .stDataFrame {
            color: #FFFFFF;
        }

        /* Title styling */
        h1, h2, h3 {
            font-family: 'ITC Avant Garde Gothic';
            color: #FFF8E8;
        }

        /* Sidebar styling */
        .css-1d391kg {
            background-color: #9DBDFF;
        }

        /* Table styling */
        table {
            border-collapse: collapse;
            width: 100%;
            background-color: #FFF8E8; /* Background table */
        }

        th {
            background-color: #FF6B6B; /* Background header tabel */
            color: white; /* Warna teks header */
            padding: 10px;
        }

        td {
            background-color: #FFD93D; /* Background body tabel */
            color: #000000; /* Warna teks body */
            padding: 10px;
        }

        tr:nth-child(even) {
            background-color: #F8D5BB; /* Baris tabel genap */
        }
        
        tr:nth-child(odd) {
            background-color: #FFE5D9; /* Baris tabel ganjil */
        }
    </style>
    """, unsafe_allow_html=True)

# Fungsi untuk metode SAW (Simple Additive Weighting)
def saw_method(decision_matrix, weights, criteria_type):
    normalized = np.zeros_like(decision_matrix, dtype=float)
    for j in range(decision_matrix.shape[1]):
        if criteria_type[j] == 'Benefit':
            normalized[:, j] = decision_matrix[:, j] / decision_matrix[:, j].max()
        else:  # Cost
            normalized[:, j] = decision_matrix[:, j].min() / decision_matrix[:, j]
    weighted = normalized * weights
    scores = weighted.sum(axis=1)
    return normalized, weighted, scores

# Fungsi untuk metode WP (Weighted Product)
def wp_method(decision_matrix, weights, criteria_type):
    # Convert decision_matrix to float
    decision_matrix = decision_matrix.astype(float)
    
    # Calculate S
    S = np.ones(decision_matrix.shape[0])
    for j in range(decision_matrix.shape[1]):
        if criteria_type[j] == 'Benefit':
            S = decision_matrix[:, j] * weights[j]
        else:  # Cost
            S = decision_matrix[:, j] * (-weights[j])
    
    # Calculate V
    V = S / S.sum()
    
    return S, V

# Fungsi untuk metode TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)
def topsis_method(decision_matrix, weights, criteria_type):
    # Normalisasi matriks keputusan
    normalized = decision_matrix / np.sqrt((decision_matrix ** 2).sum(axis=0))
    
    # Menghitung matriks keputusan ternormalisasi terbobot
    weighted = normalized * weights
    
    # Menentukan solusi ideal positif dan negatif
    ideal_solution = np.zeros(weighted.shape[1])
    nadir_solution = np.zeros(weighted.shape[1])
    
    for j in range(weighted.shape[1]):
        if criteria_type[j] == 'Benefit':
            ideal_solution[j] = weighted[:, j].max()
            nadir_solution[j] = weighted[:, j].min()
        else:  # Cost
            ideal_solution[j] = weighted[:, j].min()
            nadir_solution[j] = weighted[:, j].max()
    
    # Menghitung jarak ke solusi ideal dan nadir
    distance_to_ideal = np.sqrt(((weighted - ideal_solution) ** 2).sum(axis=1))
    distance_to_nadir = np.sqrt(((weighted - nadir_solution) ** 2).sum(axis=1))
    
    # Menghitung nilai preferensi
    scores = distance_to_nadir / (distance_to_ideal + distance_to_nadir)
    
    return normalized, weighted, ideal_solution, nadir_solution, distance_to_ideal, distance_to_nadir, scores

# Fungsi untuk metode AHP (Analytical Hierarchy Process)
def ahp_method(criteria, pairwise_matrix):
    eigvals, eigvecs = np.linalg.eig(pairwise_matrix)
    max_index = np.argmax(eigvals)
    principal_eigvec = eigvecs[:, max_index].real
    weights = principal_eigvec / principal_eigvec.sum()
    
    # Hitung consistency index (CI)
    n = pairwise_matrix.shape[0]
    lambda_max = eigvals[max_index].real
    ci = (lambda_max - n) / (n - 1)
    
    # Hitung consistency ratio (CR)
    random_index = [0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45]  # Nilai RI untuk n = 1 sampai 9
    cr = ci / random_index[n - 1] if n - 1 < len(random_index) else None
    
    return weights, ci, cr

# Fungsi utama
def main():
    add_custom_css()
    st.title("Aplikasi Pengambilan Keputusan Rekrutmen")
    st.write("""
    Aplikasi ini membantu tim HR dalam memilih kandidat terbaik berdasarkan berbagai kriteria yang telah ditentukan.
    """)

    st.sidebar.header("Input Data Rekrutmen")
    
    # Default kriteria rekrutmen
    default_criteria = ["Pengalaman Kerja (tahun)", "Pendidikan", "Keterampilan Teknis", "Keterampilan Soft", "Penilaian Wawancara"]
    default_weights = [0.3, 0.2, 0.2, 0.2, 0.1]
    default_criteria_type = ["Benefit", "Benefit", "Benefit", "Benefit", "Benefit"]
    
    # Pengguna dapat memilih untuk menggunakan kriteria default atau menentukan sendiri
    use_default = st.sidebar.checkbox("Gunakan Kriteria Default Rekrutmen", value=True)

    # Input jumlah kandidat
    n_alternatives = st.sidebar.number_input("Jumlah Kandidat", min_value=2, max_value=20, value=3) 
    
    if use_default:
        n_criteria = len(default_criteria)
        criteria = default_criteria.copy()
        weights = default_weights.copy()
        criteria_type = default_criteria_type.copy()
    else:
        n_criteria = st.sidebar.number_input("Jumlah Kriteria", min_value=2, max_value=10, value=3)
        criteria = []
        weights = []
        criteria_type = []
        for i in range(n_criteria):
            st.sidebar.markdown(f"### Kriteria {i+1}")
            criteria.append(st.sidebar.text_input(f"Nama Kriteria {i+1}", f"Kriteria {i+1}", key=f"name_{i}"))
            weights.append(st.sidebar.number_input(f"Bobot Kriteria {i+1}", min_value=0.0, max_value=1.0, value=0.5, key=f"weight_{i}"))
            criteria_type.append(st.sidebar.selectbox(f"Tipe Kriteria {i+1}", ['Benefit', 'Cost'], key=f"type_{i}"))
    
    # Pastikan total bobot adalah 1
    if use_default:
        weights = np.array(weights)
    else:
        weights = np.array(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones_like(weights) / len(weights)
    
    st.sidebar.markdown("---")
    
    # Membuat tabel input untuk kandidat dan kriteria
    st.subheader("Masukkan Nilai Kandidat untuk Setiap Kriteria")
    candidate_names = [f"Kandidat {i+1}" for i in range(n_alternatives)]
    decision_matrix = np.zeros((n_alternatives, n_criteria))
    df_input = pd.DataFrame(decision_matrix, index=candidate_names, columns=criteria)
    
    # Tampilkan tabel untuk input nilai menggunakan st.data_editor
    st.write("Silakan masukkan nilai pada tabel di bawah:")
    # Gunakan st.data_editor dengan dataframe yang benar-benar berbentuk DataFrame
    try:
        edited_df = st.data_editor(df_input, use_container_width=True)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menampilkan tabel input: {e}")

    # User memilih metode keputusan
    st.sidebar.header("Pilih Metode Pengambilan Keputusan")
    method = st.sidebar.selectbox("Metode", ["SAW", "WP", "TOPSIS", "AHP"])
    
    if st.sidebar.button("Hitung"):
        if method == "SAW":
            normalized, weighted, scores = saw_method(decision_matrix, weights, criteria_type)
            st.subheader("Hasil Perhitungan dengan SAW")
            st.write("Matriks Ternormalisasi:")
            st.dataframe(pd.DataFrame(normalized, index=candidate_names, columns=criteria))
            st.write("Matriks Terbobot:")
            st.dataframe(pd.DataFrame(weighted, index=candidate_names, columns=criteria))
            st.write("Nilai Akhir:")
            st.dataframe(pd.DataFrame(scores, index=candidate_names, columns=["Skor SAW"]))
        
        elif method == "WP":
            S, V = wp_method(decision_matrix, weights, criteria_type)
            st.subheader("Hasil Perhitungan dengan WP")
            st.write("Nilai S untuk masing-masing kandidat:")
            st.dataframe(pd.DataFrame(S, index=candidate_names, columns=["Nilai S"]))
            st.write("Nilai Akhir:")
            st.dataframe(pd.DataFrame(V, index=candidate_names, columns=["Skor WP"]))
        
        elif method == "TOPSIS":
            normalized, weighted, ideal_solution, nadir_solution, distance_to_ideal, distance_to_nadir, scores = topsis_method(decision_matrix, weights, criteria_type)
            st.subheader("Hasil Perhitungan dengan TOPSIS")
            st.write("Matriks Ternormalisasi:")
            st.dataframe(pd.DataFrame(normalized, index=candidate_names, columns=criteria))
            st.write("Matriks Terbobot:")
            st.dataframe(pd.DataFrame(weighted, index=candidate_names, columns=criteria))
            st.write("Solusi Ideal (Positif):")
            st.dataframe(pd.DataFrame([ideal_solution], columns=criteria))
            st.write("Solusi Ideal Negatif:")
            st.dataframe(pd.DataFrame([nadir_solution], columns=criteria))
            st.write("Jarak ke Solusi Ideal Positif:")
            st.dataframe(pd.DataFrame(distance_to_ideal, index=candidate_names, columns=["Jarak Positif"]))
            st.write("Jarak ke Solusi Ideal Negatif:")
            st.dataframe(pd.DataFrame(distance_to_nadir, index=candidate_names, columns=["Jarak Negatif"]))
            st.write("Nilai Akhir:")
            st.dataframe(pd.DataFrame(scores, index=candidate_names, columns=["Skor TOPSIS"]))
        
        elif method == "AHP":
            st.subheader("Hasil Perhitungan dengan AHP")
            pairwise_matrix = np.zeros((n_criteria, n_criteria))
            st.write("Masukkan matriks perbandingan berpasangan (pairwise matrix):")
            for i in range(n_criteria):
                for j in range(i+1, n_criteria):
                    pairwise_matrix[i, j] = st.number_input(f"Perbandingan {criteria[i]} terhadap {criteria[j]}", min_value=0.1, max_value=9.0, value=1.0, step=0.1, key=f"ahp_{i}_{j}")
                    pairwise_matrix[j, i] = 1 / pairwise_matrix[i, j]
            np.fill_diagonal(pairwise_matrix, 1)
            st.write("Matriks Perbandingan Berpasangan:")
            st.dataframe(pd.DataFrame(pairwise_matrix, index=criteria, columns=criteria))
            weights, ci, cr = ahp_method(criteria, pairwise_matrix)
            st.write("Bobot Kriteria:")
            st.dataframe(pd.DataFrame(weights, index=criteria, columns=["Bobot"]))
            st.write(f"Consistency Index (CI): {ci}")
            st.write(f"Consistency Ratio (CR): {cr}")
            if cr < 0.1:
                st.success("Matriks konsisten.")
            else:
                st.error("Matriks tidak konsisten. Perlu penyesuaian pada perbandingan.")
        
if __name__ == "_main_":
    main()