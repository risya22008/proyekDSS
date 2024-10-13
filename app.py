import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def add_custom_css():
    st.markdown("""
    <style>
        /* Background color */
        .stApp {
            background-color: #FFE3E3;
        }

        /* Text color */
        .stTextInput, .stDataFrame {
            color: #000000; /* Ubah teks menjadi warna hitam */
        }

        /* Title styling */
        h1, h2, h3 {
            font-family: 'Arial', sans-serif;
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
            S *= decision_matrix[:, j] ** weights[j]
        else:  # Cost
            S *= decision_matrix[:, j] ** (-weights[j])
    
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
    df_input = pd.DataFrame(decision_matrix, columns=criteria, index=candidate_names)

    # Tampilkan tabel untuk input nilai menggunakan st.data_editor
    st.write("Silakan masukkan nilai pada tabel di bawah:")
    edited_df = st.data_editor(df_input, use_container_width=True)
    
    if st.sidebar.button("Hitung"):
        st.header("Hasil Perhitungan")

        # Metode SAW
        with st.expander("Metode SAW - Langkah demi Langkah"):
            saw_normalized, saw_weighted, saw_scores = saw_method(edited_df.values, weights, criteria_type)
            df_saw_normalized = pd.DataFrame(saw_normalized, columns=criteria, index=edited_df.index)
            df_saw_weighted = pd.DataFrame(saw_weighted, columns=criteria, index=edited_df.index)
            df_saw_scores = pd.DataFrame({'Skor SAW': saw_scores}, index=edited_df.index)

            # Menambahkan perankingan untuk SAW
            df_saw_scores['Peringkat SAW'] = df_saw_scores['Skor SAW'].rank(ascending=False)

            st.subheader("Normalisasi Matriks Keputusan")
            st.dataframe(df_saw_normalized)

            st.subheader("Matriks Ternormalisasi Terbobot")
            st.dataframe(df_saw_weighted)

            st.subheader("Skor SAW")
            st.dataframe(df_saw_scores)

        # Metode WP
        with st.expander("Metode WP - Langkah demi Langkah"):
            wp_S, wp_V = wp_method(edited_df.values, weights, criteria_type)
            
            df_wp_S = pd.DataFrame({'Nilai S': wp_S}, index=edited_df.index)
            st.subheader("Menghitung S untuk Setiap Kandidat")
            st.dataframe(df_wp_S)

            df_wp_V = pd.DataFrame({'Nilai V (Skor WP)': wp_V}, index=edited_df.index)
            st.subheader("Menghitung V dan Perangkingan (Skor WP)")
            st.dataframe(df_wp_V)

            df_wp_V['Peringkat'] = df_wp_V['Nilai V (Skor WP)'].rank(ascending=False)
            st.subheader("Peringkat Berdasarkan WP")
            st.dataframe(df_wp_V[['Nilai V (Skor WP)', 'Peringkat']])

        # Metode TOPSIS
        with st.expander("Metode TOPSIS - Langkah demi Langkah"):
            topsis_normalized, topsis_weighted, ideal_solution, nadir_solution, distance_to_ideal, distance_to_nadir, topsis_scores = topsis_method(edited_df.values, weights, criteria_type)
            df_topsis_normalized = pd.DataFrame(topsis_normalized, columns=criteria, index=edited_df.index)
            df_topsis_weighted = pd.DataFrame(topsis_weighted, columns=criteria, index=edited_df.index)
            df_topsis_scores = pd.DataFrame({'Skor TOPSIS': topsis_scores}, index=edited_df.index)

            # Menambahkan perankingan untuk TOPSIS
            df_topsis_scores['Peringkat TOPSIS'] = df_topsis_scores['Skor TOPSIS'].rank(ascending=False)

            st.subheader("Normalisasi Matriks Keputusan")
            st.dataframe(df_topsis_normalized)

            st.subheader("Matriks Ternormalisasi Terbobot")
            st.dataframe(df_topsis_weighted)

            st.subheader("Skor TOPSIS")
            st.dataframe(df_topsis_scores)

        # Menampilkan hasil akhir
        results = pd.DataFrame({
            'Kandidat': edited_df.index,
            'SAW': saw_scores,
            'Peringkat SAW': df_saw_scores['Peringkat SAW'].values,
            'WP': wp_V,
            'Peringkat WP': df_wp_V['Peringkat'].values,
            'TOPSIS': topsis_scores,
            'Peringkat TOPSIS': df_topsis_scores['Peringkat TOPSIS'].values
        })

        st.subheader("Skor untuk Setiap Metode")
        st.dataframe(results.set_index('Kandidat'))

        # Menampilkan rekomendasi
        st.subheader("Rekomendasi Kandidat Terbaik")
        best_saw = results.loc[results['SAW'].idxmax(), 'Kandidat']
        best_wp = results.loc[results['WP'].idxmax(), 'Kandidat']
        best_topsis = results.loc[results['TOPSIS'].idxmax(), 'Kandidat']

        st.write(f"**Kandidat Terbaik (SAW):** {best_saw}")
        st.write(f"**Kandidat Terbaik (WP):** {best_wp}")
        st.write(f"**Kandidat Terbaik (TOPSIS):** {best_topsis}")

    # Visualisasi
        st.subheader("Visualisasi Hasil")
        fig, ax = plt.subplots(figsize=(10, 6))
        results.set_index('Kandidat').plot(kind='bar', ax=ax)
        plt.title("Perbandingan Skor Metode untuk Setiap Kandidat")
        plt.xlabel("Kandidat")
        plt.ylabel("Skor")
        plt.legend(title="Metode")
        plt.tight_layout()
        st.pyplot(fig)
        
if __name__ == "__main__":
    main()