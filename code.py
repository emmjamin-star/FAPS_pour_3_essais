import streamlit as st
import ezc3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
import tempfile

st.set_page_config(page_title="Score FAPS", layout="centered")
st.title("🦿 Score FAPS - Interface interactive")

# 1. Upload des fichiers .c3d
st.header("1. Importer un ou plusieurs fichiers .c3d dont au moins un fichier d'essai statique et un d'essai dynamique")
uploaded_files = st.file_uploader("Choisissez un ou plusieurs fichiers .c3d", type="c3d", accept_multiple_files=True)
st.header("2. Indiquer le score allant de 0 (aucune aide à la marche) à 5 (participant totalement dépendant) pour les aides ambulatoire et les dispositifs d'assistances")
df = pd.DataFrame({'Score' : [0,1,2,3,4,5]})

AmbulatoryAids = st.selectbox(
    "Pour l'aide ambulatoire :",
    df['Score'])
    
AssistiveDevice = st.selectbox(
    "Pour le dispositif d'assistance :",
    df['Score'])

if uploaded_files:
    selected_file_statique = st.selectbox("Choisissez un fichier statique pour l'analyse", uploaded_files, format_func=lambda x: x.name)
    selected_file_dynamique1 = st.selectbox("Choisissez un fichier dynamique 1 pour l'analyse", uploaded_files, format_func=lambda x: x.name)
    selected_file_dynamique2 = st.selectbox("Choisissez un fichier dynamique 2 pour l'analyse", uploaded_files, format_func=lambda x: x.name)
    selected_file_dynamique3 = st.selectbox("Choisissez un fichier dynamique 3 pour l'analyse", uploaded_files, format_func=lambda x: x.name)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".c3d") as tmp:
        tmp.write(selected_file_statique.read())
        tmp_path = tmp.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".c3d") as tmp:
        tmp.write(selected_file_dynamique1.read())
        tmp1_path = tmp.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".c3d") as tmp:
        tmp.write(selected_file_dynamique2.read())
        tmp2_path = tmp.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".c3d") as tmp:
        tmp.write(selected_file_dynamique3.read())
        tmp3_path = tmp.name

if st.button("Lancer le calcul du score FAPS"):
    try:
         def calculate_faps_fixed():
            # --- CONFIGURATION ---
            trials = [tmp1_path, tmp2_path, tmp3_path]
            static_file = tmp_path
            
            # 1. PARAMÈTRES ANTHROPOMÉTRIQUES
            try:
                statique = ezc3d.c3d(static_file)
                labelsStat = statique['parameters']['POINT']['LABELS']['value']
                
                def get_static_point(label):
                    if label not in labelsStat: return None
                    idx = labelsStat.index(label)
                    pt = statique['data']['points'][:3, idx, :]
                    mask = ~np.isnan(pt[0, :])
                    return pt[:, mask][:, 0] if np.any(mask) else None
        
                # Calcul de la longueur de jambe (Leg Length)
                p1 = get_static_point('LPSI')
                p2 = get_static_point('LANK')
                if p1 is None or p2 is None:
                    print("Erreur : Marqueurs LPSI ou LANK introuvables.")
                    return
                    
                leg_length = np.linalg.norm(p1 - p2) / 1000.0 # en METRES
            except Exception as e:
                print(f"Erreur statique : {e}")
                return
        
            results = {'v': [], 'sl_r': [], 'sl_l': [], 'st_r': [], 'st_l': [], 'dbs': []}
        
            # 2. TRAITEMENT DES ESSAIS
            for trial_path in trials:
                try:
                    acq = ezc3d.c3d(trial_path)
                    labels = acq['parameters']['POINT']['LABELS']['value']
                    freq = acq['header']['points']['frame_rate']
                    data = acq['data']['points']
                    
                    def get_clean_marker(label):
                        if label not in labels: return None
                        idx = labels.index(label)
                        m = data[:3, idx, :].copy()
                        for i in range(3):
                            mask = np.isnan(m[i, :])
                            if np.any(mask) and not np.all(mask):
                                m[i, mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), m[i, ~mask])
                        return m
        
                    r_he = get_clean_marker('RHEE')
                    l_he = get_clean_marker('LHEE')
                    strn = get_clean_marker('STRN')
                    
                    if r_he is None or l_he is None or strn is None: continue
        
                    # Détection des Heel Strikes (Axe Z)
                    hs_r, _ = find_peaks(-r_he[2, :], distance=int(freq*0.5), prominence=2)
                    hs_l, _ = find_peaks(-l_he[2, :], distance=int(freq*0.5), prominence=2)
        
                    # Vitesse (m/s) : Distance parcourue par le tronc / temps
                    dist_tronc = np.abs(strn[0, -1] - strn[0, 0]) / 1000.0
                    time_total = strn.shape[1] / freq
                    results['v'].append(dist_tronc / time_total)
        
                    # Longueur de Pas (Step Length en METRES)
                    # Axe de progression : on utilise la différence absolue sur X
                    for t0 in hs_l:
                        next_r = hs_r[hs_r > t0]
                        if len(next_r) > 0:
                            results['sl_r'].append(np.abs(r_he[0, next_r[0]] - l_he[0, t0]) / 1000.0)
                    
                    for t0 in hs_r:
                        next_l = hs_l[hs_l > t0]
                        if len(next_l) > 0:
                            results['sl_l'].append(np.abs(l_he[0, next_l[0]] - r_he[0, t0]) / 1000.0)
        
                    # Temps de pas (s)
                    if len(hs_r) > 1: results['st_r'].append(np.mean(np.diff(hs_r)) / freq)
                    if len(hs_l) > 1: results['st_l'].append(np.mean(np.diff(hs_l)) / freq)
        
                    # Base de soutien (cm) - Ecart moyen Y entre pieds
                    results['dbs'].append(np.abs(np.mean(r_he[1, :]) - np.mean(l_he[1, :])) / 10.0)
        
                except Exception as e:
                    print(f"Erreur essai {trial_path}: {e}")
        
            # 3. CALCULS FINAUX ET SCORING
            if not results['sl_r'] or not results['sl_l']:
                print("Calcul impossible : Données de pas manquantes.")
                return
        
            # Moyennes Globales
            avg_v = np.mean(results['v'])
            avg_sl_r = np.mean(results['sl_r'])
            avg_sl_l = np.mean(results['sl_l'])
            avg_st_r = np.mean(results['st_r'])
            avg_st_l = np.mean(results['st_l'])
            avg_dbs = np.mean(results['dbs'])
        
            # --- NORMALISATION (C'est ici que l'erreur se trouvait) ---
            # GV (Vitesse normalisée) = Vitesse (m/s) / LegLength (m)
            # GSL (Longueur pas normalisée) = StepLength (m) / LegLength (m)
            gv = avg_v / leg_length
            gsl_r = avg_sl_r / leg_length
            gsl_l = avg_sl_l / leg_length
        
            # FORMULE DE GOUELLE (2014)
            def get_step_function_penalty(gv_val, gsl_val, st_val):
                # On calcule l'écart à la norme pour chaque paramètre
                p_v = np.abs(gv_val - 1.49) / 0.082
                p_sl = np.abs(gsl_val - 0.77) / 0.046
                p_st = np.abs(st_val - 1.12) / 0.028
                
                total_penalty = p_v + p_sl + p_st
                return min(total_penalty, 22)
        
            sf_r = get_step_function_penalty(gv, gsl_r, avg_st_r)
            sf_l = get_step_function_penalty(gv, gsl_l, avg_st_l)
            
            # Asymétrie et Base de soutien
            sl_asy = min((np.abs(avg_sl_r / avg_sl_l - 1) / 0.2) * 8, 8) # Normalisation de l'écart à 1
            dbs_penality = min(np.abs(avg_dbs - 10) / 5, 8) # Pénalité progressive
        
            # Score Final
            score_faps = 100 - (sf_r + sf_l + sl_asy + dbs_penality + AmbulatoryAids + AssistiveDevice)
            st.markdown("### 📊 Résultats du score FAPS")
            st.write(f"**Score FAPS moyen ** : {max(0, round(score_faps, 2))}/100")
            st.write(f"Vitesse Norm. (GV) : {gv:.2f} (Cible: 1.49)")
            st.write(f"Pas Norm. (GSL)    : {(gsl_r+gsl_l)/2:.2f} (Cible: 0.77)")
            st.write(f"Temps Pas (ST)     : {(avg_st_r+avg_st_l)/2:.2f} s (Cible: 1.12)")
            st.write(f"**Lecture du test** : Un individu présentant une marche saine aura un score compris entre 95 et 100. Tout score en-dessous indique une atteinte à la fonctionnalité de la marche.")
        
         calculate_faps_fixed()
        
    except Exception as e:
        st.error(f"Erreur pendant l'analyse : {e}")
