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

    acq1 = ezc3d.c3d(tmp1_path)  # acquisition dynamique
    labels1 = acq1['parameters']['POINT']['LABELS']['value']
    freq1 = acq1['header']['points']['frame_rate']
    first_frame1 = acq1['header']['points']['first_frame']
    n_frames1 = acq1['data']['points'].shape[2]
    time_offset1 = first_frame1 / freq1
    time1 = np.arange(n_frames1) / freq1 + time_offset1

    acq2 = ezc3d.c3d(tmp2_path)  # acquisition dynamique
    labels2 = acq2['parameters']['POINT']['LABELS']['value']
    freq2 = acq2['header']['points']['frame_rate']
    first_frame2 = acq2['header']['points']['first_frame']
    n_frames2 = acq2['data']['points'].shape[2]
    time_offset2 = first_frame2 / freq2
    time2 = np.arange(n_frames2) / freq2 + time_offset2

    acq3 = ezc3d.c3d(tmp3_path)  # acquisition dynamique
    labels3 = acq3['parameters']['POINT']['LABELS']['value']
    freq3 = acq3['header']['points']['frame_rate']
    first_frame3 = acq3['header']['points']['first_frame']
    n_frames3 = acq3['data']['points'].shape[2]
    time_offset3 = first_frame3 / freq3
    time3 = np.arange(n_frames3) / freq3 + time_offset3
    
    statique = ezc3d.c3d(tmp_path)  # acquisition statique
    labelsStat = statique['parameters']['POINT']['LABELS']['value']
    freqStat = statique['header']['points']['frame_rate']
    first_frameStat = statique['header']['points']['first_frame']
    n_framesStat = statique['data']['points'].shape[2]
    time_offsetStat = first_frameStat / freqStat
    timeStat = np.arange(n_framesStat) / freqStat + time_offsetStat
    
    markersStat  = statique['data']['points']
    markers1 = acq1['data']['points']
    markers2 = acq2['data']['points']
    markers3 = acq3['data']['points']
    data1 = acq1['data']['points']
    data2 = acq2['data']['points']
    data3 = acq3['data']['points']

if st.button("Lancer le calcul du score FAPS"):
    try:
         # Extraction des coordonnées
        a1, a2, b1, b2, c1, c2 = markersStat[:,labelsStat.index('LASI'),:][0, 0], markersStat[:,labelsStat.index('LANK'),:][0, 0], markersStat[:,labelsStat.index('LASI'),:][1, 0], markersStat[:,labelsStat.index('LANK'),:][1, 0], markersStat[:,labelsStat.index('LASI'),:][2, 0], markersStat[:,labelsStat.index('LANK'),:][2, 0]
        LgJambeL = np.sqrt((a2-a1)*(a2-a1)+(b2-b1)*(b2-b1)+(c2-c1)*(c2-c1))

        d1, d2, e1, e2, f1, f2 = markersStat[:,labelsStat.index('RASI'),:][0, 0], markersStat[:,labelsStat.index('RANK'),:][0, 0], markersStat[:,labelsStat.index('RASI'),:][1, 0], markersStat[:,labelsStat.index('RANK'),:][1, 0], markersStat[:,labelsStat.index('RASI'),:][2, 0], markersStat[:,labelsStat.index('RANK'),:][2, 0]
        LgJambeR = np.sqrt((d2-d1)*(d2-d1)+(e2-e1)*(e2-e1)+(f2-f1)*(f2-f1))
      
        # Cycles première acquisition
        # Détection event gauche
        # Détection des cycles à partir du marqueur LHEE (talon gauche)
        points1 = acq1['data']['points']
        if "LHEE" in labels1:
            idx_lhee1 = labels1.index("LHEE")
            z_lhee1 = points1[2, idx_lhee1, :]
        
            # Inversion signal pour détecter les minima (contacts au sol)
            inverted_z1 = -z_lhee1
            min_distance1 = int(freq1 * 0.8)
        
            # Détection pics
            peaks1, _ = find_peaks(inverted_z1, distance = min_distance1, prominence = 1)
        
            # Début et fin des cycles = entre chaque pic
            lhee_cycle_start_indices1 = peaks1[:-1]
            lhee_cycle_end_indices1 = peaks1[1:]
            min_lhee_cycle_duration1 = int(0.5 * freq1)
            lhee_valid_cycles1 = [
              (start, end) for start, end in zip(lhee_cycle_start_indices1, lhee_cycle_end_indices1)
              if (end - start) >= min_lhee_cycle_duration1
            ]
            lhee_n_cycles1 = len(lhee_valid_cycles1)
        # Détection event droite
        # Détection des cycles à partir du marqueur RHEE (talon droite)
        points1 = acq1['data']['points']
        if "RHEE" in labels1:
            idx_rhee1 = labels1.index("RHEE")
            z_rhee1 = points1[2, idx_rhee1, :]
        
            # Inversion signal pour détecter les minima (contacts au sol)
            inverted_z1 = -z_rhee1
            min_distance1 = int(freq1 * 0.8)
        
            # Détection pics
            peaks1, _ = find_peaks(inverted_z1, distance = min_distance1, prominence = 1)
        
            # Début et fin des cycles = entre chaque pic
            rhee_cycle_start_indices1 = peaks1[:-1]
            rhee_cycle_end_indices1 = peaks1[1:]
            min_rhee_cycle_duration1 = int(0.5 * freq1)
            rhee_valid_cycles1 = [
              (start, end) for start, end in zip(rhee_cycle_start_indices1, rhee_cycle_end_indices1)
              if (end - start) >= min_rhee_cycle_duration1
            ]
            rhee_n_cycles1 = len(rhee_valid_cycles1)
        # Cycles deuxième acquisition
        # Détection event gauche
        # Détection des cycles à partir du marqueur LHEE (talon gauche)
        points2 = acq2['data']['points']
        if "LHEE" in labels2:
            idx_lhee2 = labels2.index("LHEE")
            z_lhee2 = points2[2, idx_lhee2, :]
        
            # Inversion signal pour détecter les minima (contacts au sol)
            inverted_z2 = -z_lhee2
            min_distance2 = int(freq2 * 0.8)
        
            # Détection pics
            peaks2, _ = find_peaks(inverted_z2, distance = min_distance2, prominence = 1)
        
            # Début et fin des cycles = entre chaque pic
            lhee_cycle_start_indices2 = peaks2[:-1]
            lhee_cycle_end_indices2 = peaks2[1:]
            min_lhee_cycle_duration2 = int(0.5 * freq2)
            lhee_valid_cycles2 = [
              (start, end) for start, end in zip(lhee_cycle_start_indices2, lhee_cycle_end_indices2)
              if (end - start) >= min_lhee_cycle_duration2
            ]
            lhee_n_cycles2 = len(lhee_valid_cycles2)
        # Détection event droite
        # Détection des cycles à partir du marqueur RHEE (talon droite)
        points2 = acq2['data']['points']
        if "RHEE" in labels2:
            idx_rhee2 = labels2.index("RHEE")
            z_rhee2 = points2[2, idx_rhee2, :]
        
            # Inversion signal pour détecter les minima (contacts au sol)
            inverted_z2 = -z_rhee2
            min_distance2 = int(freq2 * 0.8)
        
            # Détection pics
            peaks2, _ = find_peaks(inverted_z2, distance = min_distance2, prominence = 1)
        
            # Début et fin des cycles = entre chaque pic
            rhee_cycle_start_indices2 = peaks2[:-1]
            rhee_cycle_end_indices2 = peaks2[1:]
            min_rhee_cycle_duration2 = int(0.5 * freq2)
            rhee_valid_cycles2 = [
              (start, end) for start, end in zip(rhee_cycle_start_indices2, rhee_cycle_end_indices2)
              if (end - start) >= min_rhee_cycle_duration2
            ]
            rhee_n_cycles2 = len(rhee_valid_cycles2)
        # Cycles troisième acquisition
        # Détection event gauche
        # Détection des cycles à partir du marqueur LHEE (talon gauche)
        points3 = acq3['data']['points']
        if "LHEE" in labels3:
            idx_lhee3 = labels3.index("LHEE")
            z_lhee3 = points3[2, idx_lhee3, :]
        
            # Inversion signal pour détecter les minima (contacts au sol)
            inverted_z3 = -z_lhee3
            min_distance3 = int(freq3 * 0.8)
        
            # Détection pics
            peaks3, _ = find_peaks(inverted_z3, distance = min_distance3, prominence = 1)
        
            # Début et fin des cycles = entre chaque pic
            lhee_cycle_start_indices3 = peaks3[:-1]
            lhee_cycle_end_indices3 = peaks3[1:]
            min_lhee_cycle_duration3 = int(0.5 * freq3)
            lhee_valid_cycles3 = [
              (start, end) for start, end in zip(lhee_cycle_start_indices3, lhee_cycle_end_indices3)
              if (end - start) >= min_lhee_cycle_duration3
            ]
            lhee_n_cycles3 = len(lhee_valid_cycles3)
        # Détection event droite
        # Détection des cycles à partir du marqueur RHEE (talon droite)
        points3 = acq3['data']['points']
        if "RHEE" in labels3:
            idx_rhee3 = labels3.index("RHEE")
            z_rhee3 = points3[2, idx_rhee3, :]
        
            # Inversion signal pour détecter les minima (contacts au sol)
            inverted_z3 = -z_rhee3
            min_distance3 = int(freq3 * 0.8)
        
            # Détection pics
            peaks3, _ = find_peaks(inverted_z3, distance = min_distance3, prominence = 1)
        
            # Début et fin des cycles = entre chaque pic
            rhee_cycle_start_indices3 = peaks3[:-1]
            rhee_cycle_end_indices3 = peaks3[1:]
            min_rhee_cycle_duration3 = int(0.5 * freq3)
            rhee_valid_cycles3 = [
              (start, end) for start, end in zip(rhee_cycle_start_indices3, rhee_cycle_end_indices3)
              if (end - start) >= min_rhee_cycle_duration3
            ]
            rhee_n_cycles3 = len(rhee_valid_cycles3)
          # Longueur pas à droite
        LgPasR1 = []
        for i,j in rhee_valid_cycles1:
          a1, a2, b1, b2, c1, c2 = markers1[:,labels1.index('RANK'),:][0,i], markers1[:,labels1.index('RANK'),:][0,j], markers1[:,labels1.index('RANK'),:][1,i], markers1[:,labels1.index('RANK'),:][1,j], markers1[:,labels1.index('RANK'),:][2,i], markers1[:,labels1.index('RANK'),:][2,j]
          z = np.sqrt((a2-a1)*(a2-a1)+(b2-b1)*(b2-b1)+(c2-c1)*(c2-c1))
          LgPasR1.append(z)
        LgPasRmoy1 = np.mean(LgPasR1)
        VarLgPr1 = np.std(LgPasR1)
        
        # Longueur pas à gauche
        LgPasG1 = []
        for i,j in lhee_valid_cycles1:
          a1, a2, b1, b2, c1, c2 = markers1[:,labels1.index('LANK'),:][0,i], markers1[:,labels1.index('LANK'),:][0,j], markers1[:,labels1.index('LANK'),:][1,i], markers1[:,labels1.index('LANK'),:][1,j], markers1[:,labels1.index('LANK'),:][2,i], markers1[:,labels1.index('LANK'),:][2,j]
          z = np.sqrt((a2-a1)*(a2-a1)+(b2-b1)*(b2-b1)+(c2-c1)*(c2-c1))
          LgPasG1.append(z)
        LgPasLmoy1 = np.mean(LgPasG1)
        VarLgPl1 = np.std(LgPasG1)
        
        # Vitesse de marche
        Vmarche1 = round(((markers1[:,labels1.index('STRN'),:][0,-1]-markers1[:,labels1.index('STRN'),:][0,0]) / (len(markers1[:,labels1.index('STRN'),:][0,:]) / 100)) / 1000,2)
        # Calcul du ratio et des indices liés à la marche
        RatioLL_SLr1 = (LgPasRmoy1 / 100) / 2
        RatioLL_SLl1 = (LgPasLmoy1 / 100) / 2
        RatioV_LLr1 = (Vmarche1 / (LgJambeR/1000))
        RatioV_LLl1 = (Vmarche1 / (LgJambeL/1000))
        
        # Temps de cycle
        StepTimeCycleR1 = []
        for  i,j in rhee_valid_cycles1:
          z = (j - i)/freq1
          StepTimeCycleR1.append(z)
        
        StepTimeCycleL1 = []
        for  i,j in lhee_valid_cycles1:
          z = (j - i)/freq1
          StepTimeCycleL1.append(z)
        
        StepTimer1 = np.mean(StepTimeCycleR1)
        StepTimel1 = np.mean(StepTimeCycleL1)
        
        # Base de soutien dynamique
        # Lors du pas  coté droit
        za1 =[]
        
        for i,j in rhee_valid_cycles1 :
          a2, b2, c2 = markers1[:,labels1.index('LHEE'),:][0,i], markers1[:,labels1.index('LHEE'),:][1,i], markers1[:,labels1.index('LHEE'),:][2,i]
          a4, b4, c4 = markers1[:,labels1.index('RHEE'),:][0,i], markers1[:,labels1.index('RHEE'),:][1,i], markers1[:,labels1.index('RHEE'),:][2,i]
          a3, b3, c3 = markers1[:,labels1.index('RHEE'),:][0,j], markers1[:,labels1.index('RHEE'),:][1,j], markers1[:,labels1.index('RHEE'),:][2,j]
          a1, b1, c1 =  (a3+a4)/2, (b3+b4)/2, (c3+c4)/2
          z = np.sqrt((a2-a1)*(a2-a1)+(b2-b1)*(b2-b1)+(c2-c1)*(c2-c1))
          za1.append(z)
        
        DBSr1 = (np.mean(za1))
        
        # Lors du pas  coté gauche
        za1 =[]
        
        for i,j in lhee_valid_cycles1 :
          a2, b2, c2 = markers1[:,labels1.index('RHEE'),:][0,i], markers1[:,labels1.index('RHEE'),:][1,i], markers1[:,labels1.index('RHEE'),:][2,i]
          a4, b4, c4 = markers1[:,labels1.index('LHEE'),:][0,i], markers1[:,labels1.index('LHEE'),:][1,i], markers1[:,labels1.index('LHEE'),:][2,i]
          a3, b3, c3 = markers1[:,labels1.index('LHEE'),:][0,j], markers1[:,labels1.index('LHEE'),:][1,j], markers1[:,labels1.index('LHEE'),:][2,j]
          a1, b1, c1 =  (a3+a4)/2, (b3+b4)/2, (c3+c4)/2
          z = np.sqrt((a2-a1)*(a2-a1)+(b2-b1)*(b2-b1)+(c2-c1)*(c2-c1))
          za1.append(z)
        
        DBSl1 = (np.mean(za1))
        
        # Calcul sccore  Dynamique Base Support
        DynamiqueBaseSupport1 = ((DBSr1+DBSl1)/2)/2/100
        DynamiqueBaseSupport1 = np.abs(DynamiqueBaseSupport1-10)
        if DynamiqueBaseSupport1 > 8:
          DynamiqueBaseSupport1 = 8
        DynamiqueBaseSupport1 = np.abs(DynamiqueBaseSupport1 - 8)
        
        # Step function R
        StepFunctionR1 = np.abs(RatioV_LLr1 - 1.49)/0.082 + np.abs(RatioLL_SLr1 - 0.77)/0.046 + np.abs(StepTimer1 - 0.52) / 0.028
        if StepFunctionR1 > 22 :
          StepFunctionR1 = 22
        StepFunctionR1 = np.abs(StepFunctionR1- 22)
        
        # Step function L
        StepFunctionL1 = np.abs(RatioV_LLl1 - 1.49)/0.082 + np.abs(RatioLL_SLl1 - 0.77)/0.046 + np.abs(StepTimel1 - 0.52) / 0.028
        if StepFunctionL1 > 22 :
          StepFunctionL1 = 22
        StepFunctionL1 = np.abs(StepFunctionL1 - 22)
        
        # SL Asy
        SL_Asy1 = np.abs(RatioLL_SLr1 / RatioLL_SLl1) / 0.2
        if SL_Asy1 > 8 :
          SL_Asy1 = 8
        SL_Asy1 = np.abs(SL_Asy1 - 8)
        
        # Score final FAPS
        ScoreFAPS1 = np.round(100 - (StepFunctionR1 + StepFunctionL1 + SL_Asy1 + DynamiqueBaseSupport1 + AmbulatoryAids + AssistiveDevice),2)
        # Longueur pas à droite
        LgPasR2 = []
        for i,j in rhee_valid_cycles2:
          a1, a2, b1, b2, c1, c2 = markers2[:,labels2.index('RANK'),:][0,i], markers2[:,labels2.index('RANK'),:][0,j], markers2[:,labels2.index('RANK'),:][1,i], markers2[:,labels2.index('RANK'),:][1,j], markers2[:,labels2.index('RANK'),:][2,i], markers2[:,labels2.index('RANK'),:][2,j]
          z = np.sqrt((a2-a1)*(a2-a1)+(b2-b1)*(b2-b1)+(c2-c1)*(c2-c1))
          LgPasR2.append(z)
        LgPasRmoy2 = np.mean(LgPasR2)
        VarLgPr2 = np.std(LgPasR2)
        
        # Longueur pas à gauche
        LgPasG2 = []
        for i,j in lhee_valid_cycles2:
          a1, a2, b1, b2, c1, c2 = markers2[:,labels2.index('LANK'),:][0,i], markers2[:,labels2.index('LANK'),:][0,j], markers2[:,labels2.index('LANK'),:][1,i], markers2[:,labels2.index('LANK'),:][1,j], markers2[:,labels2.index('LANK'),:][2,i], markers2[:,labels2.index('LANK'),:][2,j]
          z = np.sqrt((a2-a1)*(a2-a1)+(b2-b1)*(b2-b1)+(c2-c1)*(c2-c1))
          LgPasG2.append(z)
        LgPasLmoy2 = np.mean(LgPasG2)
        VarLgPl2 = np.std(LgPasG2)
        
        
        # Vitesse de marche
        Vmarche2 = round(((markers2[:,labels2.index('STRN'),:][0,-1]-markers2[:,labels2.index('STRN'),:][0,0]) / (len(markers2[:,labels2.index('STRN'),:][0,:]) / 100)) / 1000,2)
        
        # Calcul du ratio et des indices liés à la marche
        RatioLL_SLr2 = (LgPasRmoy2 / 100) / 2
        RatioLL_SLl2 = (LgPasLmoy2 / 100) / 2
        RatioV_LLr2 = (Vmarche2 / (LgJambeR/1000))
        RatioV_LLl2 = (Vmarche2 / (LgJambeL/1000))
        
        # Temps de cycle
        StepTimeCycleR2 = []
        for  i,j in rhee_valid_cycles2:
          z = (j - i)/freq2
          StepTimeCycleR2.append(z)
        
        StepTimeCycleL2 = []
        for  i,j in lhee_valid_cycles2:
          z = (j - i)/freq2
          StepTimeCycleL2.append(z)
          
        StepTimer2 = np.mean(StepTimeCycleR2)
        StepTimel2 = np.mean(StepTimeCycleL2)
        
        # Base de soutien dynamique
        # Lors du pas  coté droit
        za2 =[]
        
        for i,j in rhee_valid_cycles2 :
          a2, b2, c2 = markers2[:,labels2.index('LHEE'),:][0,i], markers2[:,labels2.index('LHEE'),:][1,i], markers2[:,labels2.index('LHEE'),:][2,i]
          a4, b4, c4 = markers2[:,labels2.index('RHEE'),:][0,i], markers2[:,labels2.index('RHEE'),:][1,i], markers2[:,labels2.index('RHEE'),:][2,i]
          a3, b3, c3 = markers2[:,labels2.index('RHEE'),:][0,j], markers2[:,labels2.index('RHEE'),:][1,j], markers2[:,labels2.index('RHEE'),:][2,j]
          a1, b1, c1 =  (a3+a4)/2, (b3+b4)/2, (c3+c4)/2
          z = np.sqrt((a2-a1)*(a2-a1)+(b2-b1)*(b2-b1)+(c2-c1)*(c2-c1))
          za2.append(z)
        
        DBSr2 = (np.mean(za2))
        
        # Lors du pas  coté gauche
        za2 =[]
        
        for i,j in lhee_valid_cycles2 :
          a2, b2, c2 = markers2[:,labels2.index('RHEE'),:][0,i], markers2[:,labels2.index('RHEE'),:][1,i], markers2[:,labels2.index('RHEE'),:][2,i]
          a4, b4, c4 = markers2[:,labels2.index('LHEE'),:][0,i], markers2[:,labels2.index('LHEE'),:][1,i], markers2[:,labels2.index('LHEE'),:][2,i]
          a3, b3, c3 = markers2[:,labels2.index('LHEE'),:][0,j], markers2[:,labels2.index('LHEE'),:][1,j], markers2[:,labels2.index('LHEE'),:][2,j]
          a1, b1, c1 =  (a3+a4)/2, (b3+b4)/2, (c3+c4)/2
          z = np.sqrt((a2-a1)*(a2-a1)+(b2-b1)*(b2-b1)+(c2-c1)*(c2-c1))
          za2.append(z)
        
        DBSl2 = (np.mean(za2))
        
        # Calcul sccore  Dynamique Base Support
        DynamiqueBaseSupport2 = ((DBSr2+DBSl2)/2)/2/100
        DynamiqueBaseSupport2 = np.abs(DynamiqueBaseSupport2-10)
        if DynamiqueBaseSupport2 > 8:
          DynamiqueBaseSupport2 = 8
        DynamiqueBaseSupport2 = np.abs(DynamiqueBaseSupport2 - 8)
        
        # Step function R
        StepFunctionR2 = np.abs(RatioV_LLr2 - 1.49)/0.082 + np.abs(RatioLL_SLr2 - 0.77)/0.046 + np.abs(StepTimer2 - 0.52) / 0.028
        if StepFunctionR2 > 22 :
          StepFunctionR2 = 22
        StepFunctionR2 = np.abs(StepFunctionR2 - 22)
        
        # Step function L
        StepFunctionL2 = np.abs(RatioV_LLl2 - 1.49)/0.082 + np.abs(RatioLL_SLl2 - 0.77)/0.046 + np.abs(StepTimel2 - 0.52) / 0.028
        if StepFunctionL2 > 22 :
          StepFunctionL2 = 22
        StepFunctionL2 = np.abs(StepFunctionL2 - 22)
        
        # SL Asy
        SL_Asy2 = np.abs(RatioLL_SLr2 / RatioLL_SLl2) / 0.2
        if SL_Asy2 > 8 :
          SL_Asy2 = 8
        SL_Asy2 = np.abs(SL_Asy2 - 8)
        # Score final FAPS
        ScoreFAPS2 = np.round(100 - (StepFunctionR2 + StepFunctionL2 + SL_Asy2 + DynamiqueBaseSupport2 + AmbulatoryAids + AssistiveDevice),2)
        
        # Longueur pas à droite
        LgPasR3 = []
        for i,j in rhee_valid_cycles3:
          a1, a2, b1, b2, c1, c2 = markers3[:,labels3.index('RANK'),:][0,i], markers3[:,labels3.index('RANK'),:][0,j], markers3[:,labels3.index('RANK'),:][1,i], markers3[:,labels3.index('RANK'),:][1,j], markers3[:,labels3.index('RANK'),:][2,i], markers3[:,labels3.index('RANK'),:][2,j]
          z = np.sqrt((a2-a1)*(a2-a1)+(b2-b1)*(b2-b1)+(c2-c1)*(c2-c1))
          LgPasR3.append(z)
        LgPasRmoy3 = np.mean(LgPasR3)
        VarLgPr3 = np.std(LgPasR3)
        
        # Longueur pas à gauche
        LgPasG3 = []
        for i,j in lhee_valid_cycles3:
          a1, a2, b1, b2, c1, c2 = markers3[:,labels3.index('LANK'),:][0,i], markers3[:,labels3.index('LANK'),:][0,j], markers3[:,labels3.index('LANK'),:][1,i], markers3[:,labels3.index('LANK'),:][1,j], markers3[:,labels3.index('LANK'),:][2,i], markers3[:,labels3.index('LANK'),:][2,j]
          z = np.sqrt((a2-a1)*(a2-a1)+(b2-b1)*(b2-b1)+(c2-c1)*(c2-c1))
          LgPasG3.append(z)
        LgPasLmoy3 = np.mean(LgPasG3)
        VarLgPl3 = np.std(LgPasG3)
        
        # Vitesse de marche
        Vmarche3 = round(((markers3[:,labels3.index('STRN'),:][0,-1]-markers3[:,labels3.index('STRN'),:][0,0]) / (len(markers3[:,labels3.index('STRN'),:][0,:]) / 100)) / 1000,2)
        
        # Calcul du ratio et des indices liés à la marche
        RatioLL_SLr3 = (LgPasRmoy3 / 100) / 2
        RatioLL_SLl3 = (LgPasLmoy3 / 100) / 2
        RatioV_LLr3 = (Vmarche3 / (LgJambeR/1000))
        RatioV_LLl3 = (Vmarche3 / (LgJambeL/1000))
        
        # Temps de cycle
        StepTimeCycleR3 = []
        for  i,j in rhee_valid_cycles3:
          z = (j - i)/freq3
          StepTimeCycleR3.append(z)
        
        StepTimeCycleL3 = []
        for  i,j in lhee_valid_cycles3:
          z = (j - i)/freq3
          StepTimeCycleL3.append(z)
        
        StepTimer3 = np.mean(StepTimeCycleR3)
        StepTimel3 = np.mean(StepTimeCycleL3)
        
        # Base de soutien dynamique
        # Lors du pas  coté droit
        za3 =[]
        
        for i,j in rhee_valid_cycles3 :
          a2, b2, c2 = markers3[:,labels3.index('LHEE'),:][0,i], markers3[:,labels3.index('LHEE'),:][1,i], markers3[:,labels3.index('LHEE'),:][2,i]
          a4, b4, c4 = markers3[:,labels3.index('RHEE'),:][0,i], markers3[:,labels3.index('RHEE'),:][1,i], markers3[:,labels3.index('RHEE'),:][2,i]
          a3, b3, c3 = markers3[:,labels3.index('RHEE'),:][0,j], markers3[:,labels3.index('RHEE'),:][1,j], markers3[:,labels3.index('RHEE'),:][2,j]
          a1, b1, c1 =  (a3+a4)/2, (b3+b4)/2, (c3+c4)/2
          z = np.sqrt((a2-a1)*(a2-a1)+(b2-b1)*(b2-b1)+(c2-c1)*(c2-c1))
          za3.append(z)
        
        DBSr3 = (np.mean(za3))
        
        # Lors du pas  coté gauche
        za3 =[]
        
        for i,j in lhee_valid_cycles3 :
          a2, b2, c2 = markers3[:,labels3.index('RHEE'),:][0,i], markers3[:,labels3.index('RHEE'),:][1,i], markers3[:,labels3.index('RHEE'),:][2,i]
          a4, b4, c4 = markers3[:,labels3.index('LHEE'),:][0,i], markers3[:,labels3.index('LHEE'),:][1,i], markers3[:,labels3.index('LHEE'),:][2,i]
          a3, b3, c3 = markers3[:,labels3.index('LHEE'),:][0,j], markers3[:,labels3.index('LHEE'),:][1,j], markers3[:,labels3.index('LHEE'),:][2,j]
          a1, b1, c1 =  (a3+a4)/2, (b3+b4)/2, (c3+c4)/2
          z = np.sqrt((a2-a1)*(a2-a1)+(b2-b1)*(b2-b1)+(c2-c1)*(c2-c1))
          za3.append(z)
        
        DBSl3 = (np.mean(za3))
        
        # Calcul sccore  Dynamique Base Support
        DynamiqueBaseSupport3 = ((DBSr3+DBSl3)/2)/2/100
        DynamiqueBaseSupport3 = np.abs(DynamiqueBaseSupport3-10)
        if DynamiqueBaseSupport3 > 8:
          DynamiqueBaseSupport3 = 8
        DynamiqueBaseSupport3 = np.abs(DynamiqueBaseSupport3 - 8)
        
        # Step function R
        StepFunctionR3 = np.abs(RatioV_LLr3 - 1.49)/0.082 + np.abs(RatioLL_SLr3 - 0.77)/0.046 + np.abs(StepTimer3 - 0.52) / 0.028
        if StepFunctionR3 > 22 :
          StepFunctionR3 = 22
        StepFunctionR3 = np.abs(StepFunctionR3 - 22)
        
        # Step function L
        StepFunctionL3 = np.abs(RatioV_LLl3 - 1.49)/0.082 + np.abs(RatioLL_SLl3 - 0.77)/0.046 + np.abs(StepTimel3 - 0.52) / 0.028
        if StepFunctionL3 > 22 :
          StepFunctionL3 = 22
        StepFunctionL3 = np.abs(StepFunctionL3 - 22)
        # SL Asy
        SL_Asy3 = np.abs(RatioLL_SLr3 / RatioLL_SLl3) / 0.2
        if SL_Asy3 > 8 :
          SL_Asy3 = 8
        SL_Asy3 = np.abs(SL_Asy3 - 8)
      
        # Score final FAPS
        ScoreFAPS3 = np.round(100 - (StepFunctionR3 + StepFunctionL3 + SL_Asy3 + DynamiqueBaseSupport3 + AmbulatoryAids + AssistiveDevice),2)

        # Totale
        TotalFaps = [ScoreFAPS1, ScoreFAPS2, ScoreFAPS3]
        FAPS_M = round(np.mean(TotalFaps),2)
        STD_FAPS_M = round(np.std(TotalFaps),2)

        st.markdown("### 📊 Résultats du score FAPS")
        st.write(f"**Score FAPS moyen ** : {FAPS_M:.2f} +/- {STD_FAPS_M}")
        st.write(f"**Score FAPS acquisition n°1 ** : {ScoreFAPS1:.2f}")
        st.write(f"**Score FAPS acquisition n°2 ** : {ScoreFAPS2:.2f}")
        st.write(f"**Score FAPS acquisition n°3 ** : {ScoreFAPS3:.2f}")
        st.write(f"**Lecture du test** : Un individu présentant une marche saine aura un score compris entre 95 et 100. Tout score en-dessous indique une atteinte à la fonctionnalité de la marche.")
      
    except Exception as e:
        st.error(f"Erreur pendant l'analyse : {e}")
