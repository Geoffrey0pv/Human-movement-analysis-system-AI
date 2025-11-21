"""
Debug: Comparar features extraídas vs CSV
"""

import sys
sys.path.append('..')

from realtime_classifier_v4 import RealtimeMovementClassifierV4
import cv2
import pandas as pd
import numpy as np

# Cargar CSV
csv_path = '../src/data/mov_data_proccesed_temporal_features.csv'
df = pd.read_csv(csv_path)

print("="*60)
print("COMPARACIÓN: Features extraídas vs CSV")
print("="*60)

# Tomar un frame de sentarse
sentarse_rows = df[df['accion'] == 'sentarse']
print(f"\nFrames de 'sentarse' en CSV: {len(sentarse_rows)}")

# Tomar frame con movimiento (no el primero)
sample_row = sentarse_rows.iloc[50]  # Frame del medio

print("\nEjemplo del CSV (frame 50 de sentarse):")
print(f"  x_hombro_izq: {sample_row['x_hombro_izq']:.6f}")
print(f"  velocidad_hombro_izq: {sample_row['velocidad_hombro_izq']:.6f}")
print(f"  rodilla_izq_ang: {sample_row['rodilla_izq_ang']:.2f}")
print(f"  movement_magnitude: {sample_row['movement_magnitude']:.6f}")
print(f"  is_static: {sample_row['is_static']}")

# Ahora extraer del video
video_path = "../../Submission 1/src/data/videos/sentarse_lento_01.mp4"

classifier = RealtimeMovementClassifierV4()
cap = cv2.VideoCapture(video_path)

# Saltar al frame 50
for i in range(50):
    ret, frame = cap.read()

# Leer frame 50 y 51 para calcular velocidad
ret, frame1 = cap.read()
ret, frame2 = cap.read()

if ret:
    rgb1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    
    results1 = classifier.pose.process(rgb1)
    results2 = classifier.pose.process(rgb2)
    
    if results1.pose_landmarks and results2.pose_landmarks:
        # Extraer features con velocidad
        features_df = classifier.extract_features_like_training(
            results2.pose_landmarks.landmark,
            results1.pose_landmarks.landmark
        )
        
        print("\nFeatures extraídas del video (frame ~50):")
        print(f"  x_hombro_izq: {features_df['x_hombro_izq'].values[0]:.6f}")
        print(f"  velocidad_hombro_izq: {features_df['velocidad_hombro_izq'].values[0]:.6f}")
        print(f"  rodilla_izq_ang: {features_df['rodilla_izq_ang'].values[0]:.2f}")
        print(f"  movement_magnitude: {features_df['movement_magnitude'].values[0]:.6f}")
        print(f"  is_static: {features_df['is_static'].values[0]}")
        
        print("\n" + "="*60)
        print("COMPARACIÓN:")
        print("="*60)
        
        print(f"\nx_hombro_izq:")
        print(f"  CSV:   {sample_row['x_hombro_izq']:.6f}")
        print(f"  Video: {features_df['x_hombro_izq'].values[0]:.6f}")
        
        print(f"\nvelocidad_hombro_izq:")
        print(f"  CSV:   {sample_row['velocidad_hombro_izq']:.6f}")
        print(f"  Video: {features_df['velocidad_hombro_izq'].values[0]:.6f}")
        
        print(f"\nrodilla_izq_ang:")
        print(f"  CSV:   {sample_row['rodilla_izq_ang']:.2f}°")
        print(f"  Video: {features_df['rodilla_izq_ang'].values[0]:.2f}°")
        
        print(f"\nmovement_magnitude:")
        print(f"  CSV:   {sample_row['movement_magnitude']:.6f}")
        print(f"  Video: {features_df['movement_magnitude'].values[0]:.6f}")
        
        # Comparar todas las 62 features
        print("\n" + "="*60)
        print("DIFERENCIAS EN TODAS LAS FEATURES:")
        print("="*60)
        
        csv_features = sample_row.drop(['accion', 'velocidad_accion']).values
        video_features = features_df.values[0]
        
        print(f"\nTotal features: {len(video_features)}")
        print(f"Features del CSV: {len(csv_features)}")
        
        if len(csv_features) == len(video_features):
            diffs = np.abs(csv_features - video_features)
            max_diff_idx = np.argmax(diffs)
            feature_names = features_df.columns.tolist()
            
            print(f"\nDiferencia promedio: {np.mean(diffs):.6f}")
            print(f"Diferencia máxima: {diffs[max_diff_idx]:.6f} en '{feature_names[max_diff_idx]}'")
            
            # Mostrar top 10 diferencias
            print("\nTop 10 diferencias:")
            sorted_idx = np.argsort(diffs)[::-1][:10]
            for idx in sorted_idx:
                print(f"  {feature_names[idx]:30s}: CSV={csv_features[idx]:8.4f}, Video={video_features[idx]:8.4f}, Diff={diffs[idx]:.4f}")

cap.release()

print("\n✓ Comparación completada")
