import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from pathlib import Path

class MovementDataExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_landmarks_from_video(self, video_path):
        """Extrae landmarks de pose de un video"""
        cap = cv2.VideoCapture(str(video_path))
        
        landmarks_data = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Extraer coordenadas de cada landmark
                frame_data = {
                    'frame': frame_count,
                    'video': video_path.name
                }
                
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    frame_data[f'x_{idx}'] = landmark.x
                    frame_data[f'y_{idx}'] = landmark.y
                    frame_data[f'z_{idx}'] = landmark.z
                    frame_data[f'visibility_{idx}'] = landmark.visibility
                
                landmarks_data.append(frame_data)
            
            frame_count += 1
        
        cap.release()
        return pd.DataFrame(landmarks_data)
    
    def process_video_directory(self, video_dir, output_csv):
        """Procesa todos los videos en un directorio"""
        video_dir = Path(video_dir)
        
        # Verificar que el directorio existe
        if not video_dir.exists():
            raise FileNotFoundError(f"El directorio no existe: {video_dir}")
        
        all_data = []
        
        # Listar videos disponibles
        video_files = list(video_dir.glob('*.mp4'))
        print(f"Videos encontrados: {len(video_files)}")
        print(f"Ruta absoluta: {video_dir.absolute()}")
        
        if not video_files:
            raise ValueError(f"No se encontraron videos .mp4 en: {video_dir.absolute()}")
        
        for video_file in video_files:
            print(f"Procesando: {video_file.name}")
            df = self.extract_landmarks_from_video(video_file)
            
            if df.empty:
                print(f" No se detectaron landmarks en {video_file.name}")
                continue
            
            # Extraer metadata del nombre del archivo
            # Ejemplo: caminar_adelante_01_lento.mp4
            parts = video_file.stem.split('_')
            df['action'] = '_'.join(parts[:-2]) if len(parts) > 2 else parts[0]
            df['speed'] = parts[-1] if len(parts) > 1 else 'unknown'
            
            all_data.append(df)
            print(f"  ✓ {len(df)} frames procesados")
        
        # Verificar que hay datos para concatenar
        if not all_data:
            raise ValueError("No se pudo extraer datos de ningún video")
        
        # Combinar todos los datos
        final_df = pd.concat(all_data, ignore_index=True)
        
        # Crear directorio de salida si no existe
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        final_df.to_csv(output_csv, index=False)
        print(f"\n✓ Datos guardados en: {output_csv}")
        print(f"  Total de frames: {len(final_df)}")
        print(f"  Acciones únicas: {final_df['action'].unique()}")
        
        return final_df

if __name__ == "__main__":
    extractor = MovementDataExtractor()
    
    # Usar ruta absoluta o relativa desde la raíz del proyecto
    video_dir = Path(__file__).parent / "videos"
    output_csv = Path(__file__).parent.parent.parent.parent / "Submission 1" / "data" / "movement_data.csv"
    
    print(f"Buscando videos en: {video_dir.absolute()}")
    
    df = extractor.process_video_directory(
        video_dir,
        output_csv
    )