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
            background-color: #117B6C;
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
        .stSidebar {
                background-color: #1D3A45;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            background-color: #FFF8E8; /* Background table */
            background-color: #FFF8E8;
        }
        th {
            background-color: #FF6B6B; /* Background header tabel */
            color: white; /* Warna teks header */
            background-color: #FF6B6B;
            color: white;
            padding: 10px;
        }
        td {
            background-color: #FFD93D; /* Background body tabel */
            color: #000000; /* Warna teks body */
            background-color: #FFD93D;
            color: #000000;
            padding: 10px;
        }
        tr:nth-child(even) {
            background-color: #F8D5BB; /* Baris tabel genap */
            background-color: #F8D5BB;
        }
        
        tr:nth-child(odd) {
            background-color: #FFE5D9; /* Baris tabel ganjil */
            background-color: #FFE5D9;
        }
    </style>
    """, unsafe_allow_html=True)

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

def wp_method(decision_matrix, weights, criteria_type):
    decision_matrix = decision_matrix.astype(float)
    S = np.ones(decision_matrix.shape[0])
    for j in range(decision_matrix.shape[1]):
        if criteria_type[j] == 'Benefit':
            S = decision_matrix[:, j] * weights[j]
        else:  # Cost
            S = decision_matrix[:, j] * (-weights[j])
    V = S / S.sum()
    return S, V

def topsis_method(decision_matrix, weights, criteria_type):
    normalized = decision_matrix / np.sqrt((decision_matrix ** 2).sum(axis=0))
    weighted = normalized * weights
    ideal_solution = np.zeros(weighted.shape[1])
    nadir_solution = np.zeros(weighted.shape[1])
    
    for j in range(weighted.shape[1]):
        if criteria_type[j] == 'Benefit':
            ideal_solution[j] = weighted[:, j].max()
            nadir_solution[j] = weighted[:, j].min()
        else:  # Cost
            ideal_solution[j] = weighted[:, j].min()
            nadir_solution[j] = weighted[:, j].max()
    
    distance_to_ideal = np.sqrt(((weighted - ideal_solution) ** 2).sum(axis=1))
    distance_to_nadir = np.sqrt(((weighted - nadir_solution) ** 2).sum(axis=1))
    scores = distance_to_nadir / (distance_to_ideal + distance_to_nadir)
    
    return normalized, weighted, ideal_solution, nadir_solution, distance_to_ideal, distance_to_nadir, scores

def ahp_method(criteria, pairwise_matrix):
    n = len(criteria)
    eigenvalues, eigenvectors = np.linalg.eig(pairwise_matrix)
    max_index = np.argmax(eigenvalues)
    eigenvector = eigenvectors[:, max_index].real
    priority_vector = eigenvector / np.sum(eigenvector)
    
    lambda_max = eigenvalues[max_index].real
    ci = (lambda_max - n) / (n - 1)
    
    ri_values = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    ri = ri_values.get(n, 1.51)  # Use 1.51 for n > 10
    cr = ci / ri if ri != 0 else 0
    
    return priority_vector, ci, cr

def main():
    add_custom_css()
    st.title("🧑🏻‍💻Aplikasi Pengambilan Keputusan Rekrutmen👩🏻‍💻")
    st.write("""
    Aplikasi ini membantu tim HR dalam memilih kandidat terbaik berdasarkan berbagai kriteria yang telah ditentukan.
    """)

    st.sidebar.header("Input Data Rekrutmen")
    
    default_criteria = ["Pengalaman Kerja (tahun)", "Pendidikan", "Keterampilan Teknis", "Keterampilan Soft", "Penilaian Wawancara"]
    default_weights = [0.3, 0.2, 0.2, 0.2, 0.1]
    default_criteria_type = ["Benefit", "Benefit", "Benefit", "Benefit", "Benefit"]
    
    use_default = st.sidebar.checkbox("Gunakan Kriteria Default Rekrutmen", value=True)

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
    
    weights = np.array(weights)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones_like(weights) / len(weights)
    
    st.sidebar.markdown("---")
    
    st.subheader("Masukkan Nilai Kandidat untuk Setiap Kriteria")
    candidate_names = [f"Kandidat {i+1}" for i in range(n_alternatives)]
    decision_matrix = np.zeros((n_alternatives, n_criteria))
    df_input = pd.DataFrame(decision_matrix, index=candidate_names, columns=criteria)
    
    edited_df = st.data_editor(df_input, use_container_width=True)
    decision_matrix = edited_df.values

    st.sidebar.header("Pilih Metode Pengambilan Keputusan")
    method = st.sidebar.selectbox("Metode", ["SAW", "WP", "TOPSIS", "AHP"])
    
    if method != "AHP":
        if st.sidebar.button("Hitung"):
            if method == "SAW":
                normalized, weighted, scores = saw_method(decision_matrix, weights, criteria_type)
                st.subheader("Hasil Perhitungan dengan SAW")
                st.write("Matriks Ternormalisasi:")
                st.dataframe(pd.DataFrame(normalized, index=candidate_names, columns=criteria))
                st.write("Matriks Terbobot:")
                st.dataframe(pd.DataFrame(weighted, index=candidate_names, columns=criteria))
                st.write("Nilai Akhir:")
                results_df = pd.DataFrame(scores, index=candidate_names, columns=["Skor SAW"])
                results_df = results_df.sort_values("Skor SAW", ascending=False).reset_index()
                st.dataframe(results_df)
                
                fig, ax = plt.subplots()
                ax.bar(results_df["index"], results_df["Skor SAW"])
                ax.set_xlabel("Kandidat")
                ax.set_ylabel("Skor SAW")
                ax.set_title("Peringkat Kandidat berdasarkan SAW")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            elif method == "WP":
                S, V = wp_method(decision_matrix, weights, criteria_type)
                st.subheader("Hasil Perhitungan dengan WP")
                st.write("Nilai S untuk masing-masing kandidat:")
                st.dataframe(pd.DataFrame(S, index=candidate_names, columns=["Nilai S"]))
                st.write("Nilai Akhir:")
                results_df = pd.DataFrame(V, index=candidate_names, columns=["Skor WP"])
                results_df = results_df.sort_values("Skor WP", ascending=False).reset_index()
                st.dataframe(results_df)
                
                fig, ax = plt.subplots()
                ax.bar(results_df["index"], results_df["Skor WP"])
                ax.set_xlabel("Kandidat")
                ax.set_ylabel("Skor WP")
                ax.set_title("Peringkat Kandidat berdasarkan WP")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
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
                results_df = pd.DataFrame(scores, index=candidate_names, columns=["Skor TOPSIS"])
                results_df = results_df.sort_values("Skor TOPSIS", ascending=False).reset_index()
                st.dataframe(results_df)
                
                fig, ax = plt.subplots()
                ax.bar(results_df["index"], results_df["Skor TOPSIS"])
                ax.set_xlabel("Kandidat")
                ax.set_ylabel("Skor TOPSIS")
                ax.set_title("Peringkat Kandidat berdasarkan TOPSIS")
                plt.xticks(rotation=45)
                st.pyplot(fig)
    
    else:  # AHP method
        st.subheader("Hasil Perhitungan dengan AHP")
        
        # Initialize the pairwise matrix
        if 'pairwise_matrix' not in st.session_state:
            st.session_state.pairwise_matrix = np.ones((n_criteria, n_criteria))
        
        # Input pairwise comparisons
        for i in range(n_criteria):
            for j in range(i+1, n_criteria):
                key = f"pairwise_{i}_{j}"
                if key not in st.session_state:
                    st.session_state[key] = 1.0
                
                value = st.number_input(f"{criteria[i]} dibandingkan dengan {criteria[j]}", 
                                        min_value=1/9, max_value=9.0, value=st.session_state[key], 
                                        step=0.1, format="%.2f", key=key)
                
                st.session_state.pairwise_matrix[i, j] = value
                st.session_state.pairwise_matrix[j, i] = 1 / value
        
        # Display the pairwise comparison matrix
        st.write("Matriks Perbandingan Berpasangan:")
        st.dataframe(pd.DataFrame(st.session_state.pairwise_matrix, index=criteria, columns=criteria))
        
        # Calculate AHP results
        weights, ci, cr = ahp_method(criteria, st.session_state.pairwise_matrix)
        
        st.write("Bobot Kriteria:")
        weights_df = pd.DataFrame({"Kriteria": criteria, "Bobot": weights})
        st.dataframe(weights_df.set_index("Kriteria"))
        
        st.write(f"Consistency Index (CI): {ci:.4f}")
        st.write(f"Consistency Ratio (CR): {cr:.4f}")
        
        if cr < 0.1:
            st.success("Matriks konsisten (CR < 0.1)")
        else:
            st.warning("Matriks tidak konsisten (CR >= 0.1). Pertimbangkan untuk merevisi penilaian Anda.")
        
        final_scores = np.dot(decision_matrix, weights)
        results_df = pd.DataFrame({
            "Kandidat": candidate_names,
            "Skor AHP": final_scores
        })
        results_df = results_df.sort_values("Skor AHP", ascending=False).reset_index(drop=True)
        
        st.write("Hasil Peringkat Kandidat:")
        st.dataframe(results_df)
        
        fig, ax = plt.subplots()
        ax.bar(results_df["Kandidat"], results_df["Skor AHP"])
        ax.set_xlabel("Kandidat")
        ax.set_ylabel("Skor AHP")
        ax.set_title("Peringkat Kandidat berdasarkan AHP")
        plt.xticks(rotation=45)
        st.pyplot(fig)

if __name__ == "__main__":
    main()