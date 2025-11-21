"""
Pipeline completo: Aumentar dataset con videos de Submission 1
==============================================================
Este script:
1. Extrae features de los videos de Submission 1
2. Procesa las features (coordenadas, velocidades, ángulos)
3. Agrega features temporales (13 nuevas)
4. Combina con el dataset existente
5. Re-entrena el modelo con más datos
"""

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from pathlib import Path
import sys

class DataAugmentationPipeline:
    """Pipeline completo para aumentar el dataset"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Mapeo de landmarks MediaPipe a nombres del cuerpo
        self.body_parts = {
            'hombro_izq': 11,
            'hombro_der': 12,
            'codo_izq': 13,
            'codo_der': 14,
            'cadera_izq': 23,
            'cadera_der': 24,
            'rodilla_izq': 25,
            'rodilla_der': 26,
            'tobillo_izq': 27,
            'tobillo_der': 28
        }
    
    def extract_landmarks_from_video(self, video_path):
        """Extraer landmarks de un video"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"❌ Error: No se puede abrir {video_path}")
            return pd.DataFrame()
        
        landmarks_data = []
        frame_count = 0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  Total frames: {total_frames}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                frame_data = {'frame': frame_count}
                
                # Extraer todas las coordenadas (33 landmarks)
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    frame_data[f'x_{idx}'] = landmark.x
                    frame_data[f'y_{idx}'] = landmark.y
                    frame_data[f'z_{idx}'] = landmark.z
                    frame_data[f'visibility_{idx}'] = landmark.visibility
                
                landmarks_data.append(frame_data)
            
            frame_count += 1
            
            # Mostrar progreso cada 30 frames
            if frame_count % 30 == 0:
                print(f"  Progreso: {frame_count}/{total_frames} frames", end='\r')
        
        cap.release()
        return pd.DataFrame(landmarks_data)
    
    def calculate_angle(self, a, b, c):
        """Calcular ángulo entre 3 puntos"""
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
                  np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
        
        return angle
    
    def process_landmarks_to_features(self, df_raw):
        """
        Procesar landmarks crudos → features procesadas
        (Equivalente al notebook preproccessing.ipynb)
        """
        print("  Procesando landmarks → features...")
        
        processed_data = []
        total_rows = len(df_raw)
        
        for idx in range(total_rows):
            if idx % 50 == 0:
                print(f"  Progreso: {idx}/{total_rows} frames", end='\r')
            
            row = df_raw.iloc[idx]
            features = {}
            
            # 1. Coordenadas y velocidades (40 features)
            for part_name, lm_id in self.body_parts.items():
                # Coordenadas x, y, z
                features[f'x_{part_name}'] = row[f'x_{lm_id}']
                features[f'y_{part_name}'] = row[f'y_{lm_id}']
                features[f'z_{part_name}'] = row[f'z_{lm_id}']
                
                # Velocidad (distancia euclidiana 3D entre frames)
                if idx > 0:
                    prev_row = df_raw.iloc[idx - 1]
                    dx = row[f'x_{lm_id}'] - prev_row[f'x_{lm_id}']
                    dy = row[f'y_{lm_id}'] - prev_row[f'y_{lm_id}']
                    dz = row[f'z_{lm_id}'] - prev_row[f'z_{lm_id}']
                    velocidad = np.sqrt(dx**2 + dy**2 + dz**2)
                else:
                    velocidad = 0.0
                
                features[f'velocidad_{part_name}'] = velocidad
            
            # 2. Ángulos (8 features)
            # Extraer coordenadas necesarias
            def get_point(lm_id):
                return np.array([row[f'x_{lm_id}'], row[f'y_{lm_id}']])
            
            # Rodillas
            features['rodilla_izq_ang'] = self.calculate_angle(
                get_point(23), get_point(25), get_point(27)  # cadera-rodilla-tobillo
            )
            features['rodilla_der_ang'] = self.calculate_angle(
                get_point(24), get_point(26), get_point(28)
            )
            
            # Caderas
            features['cadera_izq_ang'] = self.calculate_angle(
                get_point(11), get_point(23), get_point(25)  # hombro-cadera-rodilla
            )
            features['cadera_der_ang'] = self.calculate_angle(
                get_point(12), get_point(24), get_point(26)
            )
            
            # Codos
            features['codo_izq_ang'] = self.calculate_angle(
                get_point(11), get_point(13), get_point(15)  # hombro-codo-muñeca
            )
            features['codo_der_ang'] = self.calculate_angle(
                get_point(12), get_point(14), get_point(16)
            )
            
            # Hombros
            features['hombro_izq_ang'] = self.calculate_angle(
                get_point(13), get_point(11), get_point(23)  # codo-hombro-cadera
            )
            features['hombro_der_ang'] = self.calculate_angle(
                get_point(14), get_point(12), get_point(24)
            )
            
            # 3. Inclinación del tronco (1 feature)
            hombro_medio_y = (row['y_11'] + row['y_12']) / 2
            cadera_medio_y = (row['y_23'] + row['y_24']) / 2
            hombro_medio_x = (row['x_11'] + row['x_12']) / 2
            cadera_medio_x = (row['x_23'] + row['x_24']) / 2
            
            features['inclinacion_tronco_ang'] = np.arctan2(
                abs(hombro_medio_x - cadera_medio_x),
                abs(hombro_medio_y - cadera_medio_y)
            ) * 180.0 / np.pi
            
            processed_data.append(features)
        
        return pd.DataFrame(processed_data)
    
    def add_temporal_features(self, df):
        """Agregar 13 features temporales"""
        print("  Agregando features temporales...")
        
        # Extraer columnas de velocidad
        velocity_cols = [col for col in df.columns if 'velocidad' in col.lower() and col != 'velocidad_accion']
        
        # 1. movement_magnitude: suma de todas las velocidades
        df['movement_magnitude'] = df[velocity_cols].abs().sum(axis=1)
        
        # 2. movement_average: promedio de velocidades
        df['movement_average'] = df[velocity_cols].abs().mean(axis=1)
        
        # 3. movement_max: velocidad máxima
        df['movement_max'] = df[velocity_cols].abs().max(axis=1)
        
        # 4. is_static: ¿está quieto? (threshold = 3.56 del análisis previo)
        df['is_static'] = (df['movement_magnitude'] < 3.56).astype(int)
        
        # 5. leg_movement: movimiento de piernas
        leg_cols = ['velocidad_cadera_izq', 'velocidad_cadera_der', 
                    'velocidad_rodilla_izq', 'velocidad_rodilla_der', 
                    'velocidad_tobillo_izq', 'velocidad_tobillo_der']
        df['leg_movement'] = df[leg_cols].abs().sum(axis=1)
        
        # 6. arm_movement: movimiento de brazos
        arm_cols = ['velocidad_hombro_izq', 'velocidad_hombro_der', 
                    'velocidad_codo_izq', 'velocidad_codo_der']
        df['arm_movement'] = df[arm_cols].abs().sum(axis=1)
        
        # 7. leg_arm_ratio: ratio pierna/brazo
        df['leg_arm_ratio'] = df['leg_movement'] / (df['arm_movement'] + 1e-6)
        
        # 8. vertical_position: posición vertical promedio
        vertical_cols = ['y_hombro_izq', 'y_hombro_der', 'y_cadera_izq', 'y_cadera_der']
        df['vertical_position'] = df[vertical_cols].mean(axis=1)
        
        # 9. vertical_range: rango vertical
        df['vertical_range'] = df[vertical_cols].max(axis=1) - df[vertical_cols].min(axis=1)
        
        # 10. movement_variance: varianza del movimiento
        df['movement_variance'] = df[velocity_cols].abs().var(axis=1)
        
        # 11. movement_cv: coeficiente de variación
        df['movement_cv'] = np.sqrt(df['movement_variance']) / (df['movement_average'] + 1e-6)
        
        # 12. depth_position: posición en profundidad (usando z)
        depth_cols = ['z_hombro_izq', 'z_hombro_der', 'z_cadera_izq', 'z_cadera_der']
        df['depth_position'] = df[depth_cols].mean(axis=1)
        
        # 13. forward_backward_movement: movimiento adelante/atrás
        # Calculado como diferencia de depth entre frames
        df['forward_backward_movement'] = df['depth_position'].diff().fillna(0)
        
        return df
    
    def process_video(self, video_path, action, speed):
        """Procesar un video completo: extracción → procesamiento → features temporales"""
        print(f"\n📹 Procesando: {video_path.name}")
        
        # 1. Extraer landmarks crudos
        df_raw = self.extract_landmarks_from_video(video_path)
        
        if df_raw.empty:
            print(f"  ❌ No se detectaron landmarks")
            return pd.DataFrame()
        
        print(f"  ✓ {len(df_raw)} frames extraídos")
        
        # 2. Procesar landmarks → features
        df_processed = self.process_landmarks_to_features(df_raw)
        
        # 3. Agregar features temporales
        df_with_temporal = self.add_temporal_features(df_processed)
        
        # 4. Agregar metadata
        df_with_temporal['accion'] = action
        df_with_temporal['velocidad_accion'] = speed
        
        print(f"  ✓ {len(df_with_temporal.columns)} features totales")
        
        return df_with_temporal
    
    def process_all_videos(self, video_dir):
        """Procesar todos los videos de un directorio"""
        video_dir = Path(video_dir)
        video_files = list(video_dir.glob('*.mp4'))
        
        if not video_files:
            print(f"❌ No se encontraron videos en: {video_dir}")
            return pd.DataFrame()
        
        print(f"\n{'='*60}")
        print(f"PROCESANDO {len(video_files)} VIDEOS")
        print(f"{'='*60}")
        
        all_data = []
        
        for video_file in video_files:
            # Parsear nombre del archivo
            # Ejemplo: caminar_adelante_01_lento.mp4
            parts = video_file.stem.split('_')
            
            # Extraer acción y velocidad
            if 'lento' in video_file.stem:
                speed = 'lento'
                action = video_file.stem.replace('_lento', '').replace('_01', '').replace('_02', '')
            elif 'rapido' in video_file.stem:
                speed = 'rapido'
                action = video_file.stem.replace('_rapido', '').replace('_01', '').replace('_02', '')
            else:
                speed = 'normal'
                action = '_'.join(parts[:-1]) if len(parts) > 1 else parts[0]
            
            # Procesar video
            df = self.process_video(video_file, action, speed)
            
            if not df.empty:
                all_data.append(df)
        
        if not all_data:
            print("\n❌ No se procesó ningún video correctamente")
            return pd.DataFrame()
        
        # Combinar todos los datos
        df_combined = pd.concat(all_data, ignore_index=True)
        
        print(f"\n{'='*60}")
        print(f"RESUMEN")
        print(f"{'='*60}")
        print(f"Total frames procesados: {len(df_combined)}")
        print(f"Acciones únicas: {df_combined['accion'].unique()}")
        print(f"Distribución de clases:")
        print(df_combined['accion'].value_counts())
        
        return df_combined


def main():
    """Función principal"""
    
    # Rutas
    video_dir = Path(__file__).parent.parent.parent / "Submission 1" / "src" / "data" / "videos"
    existing_csv = Path(__file__).parent.parent / "src" / "data" / "mov_data_proccesed_temporal_features.csv"
    output_csv = Path(__file__).parent.parent / "src" / "data" / "mov_data_proccesed_temporal_features_augmented.csv"
    
    print(f"\n{'='*60}")
    print("PIPELINE DE AUMENTO DE DATOS")
    print(f"{'='*60}")
    print(f"\nVideos a procesar: {video_dir}")
    print(f"Dataset existente: {existing_csv}")
    print(f"Dataset aumentado: {output_csv}")
    
    # 1. Procesar videos nuevos
    pipeline = DataAugmentationPipeline()
    df_new = pipeline.process_all_videos(video_dir)
    
    if df_new.empty:
        print("\n❌ Error: No se procesaron videos nuevos")
        return
    
    # 2. Cargar dataset existente
    print(f"\n{'='*60}")
    print("COMBINANDO DATASETS")
    print(f"{'='*60}")
    
    if existing_csv.exists():
        df_existing = pd.read_csv(existing_csv)
        print(f"\n✓ Dataset existente cargado: {len(df_existing)} frames")
        print(f"  Distribución de clases:")
        for action, count in df_existing['accion'].value_counts().items():
            print(f"    {action}: {count}")
        
        # 3. Combinar datasets
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        print(f"\n✓ Datasets combinados: {len(df_combined)} frames totales")
    else:
        print(f"\n⚠️ No se encontró dataset existente, usando solo datos nuevos")
        df_combined = df_new
    
    print(f"\nDistribución final de clases:")
    for action, count in df_combined['accion'].value_counts().items():
        print(f"  {action}: {count}")
    
    # 4. Guardar dataset aumentado
    df_combined.to_csv(output_csv, index=False)
    print(f"\n✓ Dataset aumentado guardado: {output_csv}")
    print(f"  Total frames: {len(df_combined)}")
    print(f"  Total features: {len(df_combined.columns)}")
    
    print(f"\n{'='*60}")
    print("✅ PIPELINE COMPLETADO")
    print(f"{'='*60}")
    print(f"\nPróximos pasos:")
    print(f"1. Re-entrenar modelo:")
    print(f"   cd ../src/models")
    print(f"   python my_model.py")
    print(f"\n2. Probar clasificador:")
    print(f"   cd ../../tests")
    print(f"   python realtime_classifier_v4.py '<video_path>'")


if __name__ == "__main__":
    main()
