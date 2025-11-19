"""
Sistema de Clasificación de Movimientos en Tiempo Real - VERSIÓN 3 (PRODUCCIÓN)
- Usa NOMBRES DE FEATURES EXACTOS del CSV (sin warnings sklearn)
- Suprime warnings de protobuf
- Extracción perfecta de 49 features
- Soporte webcam + videos
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)  # Suprimir warnings sklearn y protobuf

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import time
import argparse


class RealtimeMovementClassifier:
    """Clasificador de producción con feature names correctos"""
    
    # NOMBRES EXACTOS de las 49 features del CSV
    FEATURE_NAMES = [
        'x_hombro_izq', 'y_hombro_izq', 'z_hombro_izq', 'velocidad_hombro_izq',
        'x_hombro_der', 'y_hombro_der', 'z_hombro_der', 'velocidad_hombro_der',
        'x_codo_izq', 'y_codo_izq', 'z_codo_izq', 'velocidad_codo_izq',
        'x_codo_der', 'y_codo_der', 'z_codo_der', 'velocidad_codo_der',
        'x_cadera_izq', 'y_cadera_izq', 'z_cadera_izq', 'velocidad_cadera_izq',
        'x_cadera_der', 'y_cadera_der', 'z_cadera_der', 'velocidad_cadera_der',
        'x_rodilla_izq', 'y_rodilla_izq', 'z_rodilla_izq', 'velocidad_rodilla_izq',
        'x_rodilla_der', 'y_rodilla_der', 'z_rodilla_der', 'velocidad_rodilla_der',
        'x_tobillo_izq', 'y_tobillo_izq', 'z_tobillo_izq', 'velocidad_tobillo_izq',
        'x_tobillo_der', 'y_tobillo_der', 'z_tobillo_der', 'velocidad_tobillo_der',
        'rodilla_izq_ang', 'rodilla_der_ang', 'cadera_izq_ang', 'cadera_der_ang',
        'codo_izq_ang', 'codo_der_ang', 'hombro_izq_ang', 'hombro_der_ang',
        'inclinacion_tronco_ang'
    ]
    
    # Mapeo de índices de MediaPipe a nombres
    MP_LANDMARKS = {
        'hombro_izq': 11,
        'hombro_der': 12,
        'codo_izq': 13,
        'codo_der': 14,
        'cadera_izq': 23,
        'cadera_der': 24,
        'rodilla_izq': 25,
        'rodilla_der': 26,
        'tobillo_izq': 27,
        'tobillo_der': 28,
    }
    
    def __init__(self, model_path='modelo_acciones.pkl'):
        # Cargar modelo
        print("Cargando modelo...")
        model_data = joblib.load(model_path)
        
        if isinstance(model_data, dict):
            self.model = model_data['model']
            self.label_encoder = model_data.get('label_encoder')
            self.classes = list(model_data.get('classes', []))
        else:
            self.model = model_data
            self.label_encoder = None
            self.classes = []
        
        print(f"✓ Modelo cargado")
        if self.classes:
            print(f"✓ Clases: {self.classes}")
        
        # MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # FPS
        self.prev_time = 0
        
        # Historial para suavizado y velocidad
        self.prediction_history = []
        self.history_size = 5
        self.prev_landmarks = None
        
        # Debugging
        self.debug = False
    
    def calculate_angle(self, a, b, c):
        """Calcula ángulo entre 3 puntos (en grados)"""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
        
        return angle
    
    def calculate_velocity(self, current, previous):
        """Calcula velocidad euclidiana entre dos puntos"""
        if previous is None:
            return 0.0
        
        dx = current.x - previous.x
        dy = current.y - previous.y
        dz = current.z - previous.z
        
        return np.sqrt(dx**2 + dy**2 + dz**2)
    
    def extract_features_from_landmarks(self, landmarks):
        """
        Extrae EXACTAMENTE las 49 features del CSV con NOMBRES CORRECTOS
        Esto elimina los warnings de sklearn
        """
        if not landmarks:
            return None
        
        lm = landmarks.landmark
        prev_lm = self.prev_landmarks.landmark if self.prev_landmarks else None
        
        features = {}
        
        # === PARTE 1: Coordenadas y velocidades de cada articulación ===
        for name, idx in self.MP_LANDMARKS.items():
            # Coordenadas x, y, z
            features[f'x_{name}'] = lm[idx].x
            features[f'y_{name}'] = lm[idx].y
            features[f'z_{name}'] = lm[idx].z
            
            # Velocidad (distancia euclidiana con frame anterior)
            if prev_lm:
                features[f'velocidad_{name}'] = self.calculate_velocity(lm[idx], prev_lm[idx])
            else:
                features[f'velocidad_{name}'] = 0.0
        
        # === PARTE 2: Ángulos de articulaciones ===
        # Rodilla izquierda: cadera_izq -> rodilla_izq -> tobillo_izq
        features['rodilla_izq_ang'] = self.calculate_angle(
            lm[23], lm[25], lm[27]
        )
        
        # Rodilla derecha: cadera_der -> rodilla_der -> tobillo_der
        features['rodilla_der_ang'] = self.calculate_angle(
            lm[24], lm[26], lm[28]
        )
        
        # Cadera izquierda: hombro_izq -> cadera_izq -> rodilla_izq
        features['cadera_izq_ang'] = self.calculate_angle(
            lm[11], lm[23], lm[25]
        )
        
        # Cadera derecha: hombro_der -> cadera_der -> rodilla_der
        features['cadera_der_ang'] = self.calculate_angle(
            lm[12], lm[24], lm[26]
        )
        
        # Codo izquierdo: hombro_izq -> codo_izq -> muñeca_izq
        features['codo_izq_ang'] = self.calculate_angle(
            lm[11], lm[13], lm[15]
        )
        
        # Codo derecho: hombro_der -> codo_der -> muñeca_der
        features['codo_der_ang'] = self.calculate_angle(
            lm[12], lm[14], lm[16]
        )
        
        # Hombro izquierdo: cadera_izq -> hombro_izq -> codo_izq
        features['hombro_izq_ang'] = self.calculate_angle(
            lm[23], lm[11], lm[13]
        )
        
        # Hombro derecho: cadera_der -> hombro_der -> codo_der
        features['hombro_der_ang'] = self.calculate_angle(
            lm[24], lm[12], lm[14]
        )
        
        # Inclinación del tronco: promedio de caderas -> promedio de hombros (ángulo vertical)
        avg_shoulder_x = (lm[11].x + lm[12].x) / 2
        avg_shoulder_y = (lm[11].y + lm[12].y) / 2
        avg_hip_x = (lm[23].x + lm[24].x) / 2
        avg_hip_y = (lm[23].y + lm[24].y) / 2
        
        # Ángulo con respecto a la vertical
        angle_rad = np.arctan2(avg_shoulder_x - avg_hip_x, avg_hip_y - avg_shoulder_y)
        features['inclinacion_tronco_ang'] = abs(angle_rad * 180.0 / np.pi)
        
        # Guardar landmarks actuales para el siguiente frame
        self.prev_landmarks = landmarks
        
        # Convertir a DataFrame con los nombres de columnas EXACTOS
        df = pd.DataFrame([features], columns=self.FEATURE_NAMES)
        
        if self.debug:
            print(f"Features extraídas: {len(df.columns)}")
            print(f"Sample: {list(features.keys())[:5]}...")
        
        return df
    
    def smooth_prediction(self, prediction):
        """Suaviza predicciones usando votación por mayoría"""
        self.prediction_history.append(prediction)
        
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)
        
        if len(self.prediction_history) >= 3:
            unique, counts = np.unique(self.prediction_history, return_counts=True)
            return unique[np.argmax(counts)]
        
        return prediction
    
    def process_frame(self, frame):
        """Procesa un frame y retorna predicción"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        prediction = None
        confidence = 0.0
        probas_dict = {}
        
        if results.pose_landmarks:
            # Dibujar landmarks
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            # Extraer features con nombres correctos (DataFrame)
            features_df = self.extract_features_from_landmarks(results.pose_landmarks)
            
            if features_df is not None:
                try:
                    # Predicción usando DataFrame (sklearn reconoce los nombres)
                    pred_encoded = self.model.predict(features_df)[0]
                    probas = self.model.predict_proba(features_df)[0]
                    
                    # Decodificar
                    if self.label_encoder is not None:
                        prediction = self.label_encoder.inverse_transform([pred_encoded])[0]
                        
                        # Crear diccionario de probabilidades
                        for i, class_name in enumerate(self.classes):
                            probas_dict[class_name] = probas[i]
                    else:
                        prediction = pred_encoded
                    
                    confidence = probas.max()
                    
                    # Suavizar
                    prediction = self.smooth_prediction(prediction)
                    
                    if self.debug:
                        print(f"\nPredicción: {prediction} ({confidence*100:.1f}%)")
                        print(f"Top 3 probabilidades:")
                        sorted_probas = sorted(probas_dict.items(), key=lambda x: x[1], reverse=True)[:3]
                        for action, prob in sorted_probas:
                            print(f"  {action}: {prob*100:.1f}%")
                    
                except Exception as e:
                    print(f"Error en predicción: {e}")
                    if self.debug:
                        import traceback
                        traceback.print_exc()
        
        return frame, prediction, confidence, probas_dict, results.pose_landmarks
    
    def draw_info(self, frame, prediction, confidence, probas_dict, fps):
        """Dibuja información mejorada en el frame"""
        h, w = frame.shape[:2]
        
        # Fondo semitransparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (min(400, w-10), min(200, h-10)), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y_offset = 35
        
        # Título
        cv2.putText(frame, "Clasificador de Movimientos V3", 
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", 
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += 25
        
        # Predicción
        if prediction:
            color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
            cv2.putText(frame, f"Accion: {prediction}", 
                        (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
            cv2.putText(frame, f"Confianza: {confidence*100:.1f}%", 
                        (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 30
            
            # Top 3 probabilidades
            if probas_dict:
                cv2.putText(frame, "Probabilidades:", 
                            (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                y_offset += 20
                
                sorted_probas = sorted(probas_dict.items(), key=lambda x: x[1], reverse=True)[:3]
                for action, prob in sorted_probas:
                    cv2.putText(frame, f"  {action}: {prob*100:.0f}%", 
                                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
                    y_offset += 18
        else:
            cv2.putText(frame, "Sin deteccion", 
                        (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Instrucciones
        cv2.putText(frame, "Presiona 'q' para salir | 'd' debug", 
                    (w-380, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run_webcam(self, camera_id=0):
        """Ejecuta clasificación en tiempo real con webcam"""
        print("\n" + "="*60)
        print("CLASIFICADOR EN TIEMPO REAL - WEBCAM")
        print("="*60)
        print("Controles:")
        print("  'q' - Salir")
        print("  'd' - Toggle debug")
        print("Iniciando cámara...")
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ Error: No se pudo abrir la cámara")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("✓ Cámara iniciada")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame, prediction, confidence, probas_dict, landmarks = self.process_frame(frame)
            
            current_time = time.time()
            fps = 1 / (current_time - self.prev_time) if self.prev_time > 0 else 0
            self.prev_time = current_time
            
            frame = self.draw_info(frame, prediction, confidence, probas_dict, fps)
            
            cv2.imshow('Clasificador V3 - Webcam', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                self.debug = not self.debug
                print(f"Debug: {'ON' if self.debug else 'OFF'}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Webcam cerrada")
    
    def run_video(self, video_path):
        """Ejecuta clasificación en un archivo de video"""
        print("\n" + "="*60)
        print(f"CLASIFICADOR - VIDEO: {Path(video_path).name}")
        print("="*60)
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"❌ Error: No se pudo abrir el video: {video_path}")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✓ Video cargado")
        print(f"  Frames: {total_frames}")
        print(f"  FPS: {fps_video:.1f}")
        print("\nControles:")
        print("  'q' - Salir")
        print("  'd' - Toggle debug")
        print("  SPACE - Pausar/Reanudar")
        
        frame_count = 0
        paused = False
        
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("\n✓ Video terminado")
                    break
                
                frame_count += 1
                frame, prediction, confidence, probas_dict, landmarks = self.process_frame(frame)
                
                current_time = time.time()
                fps = 1 / (current_time - self.prev_time) if self.prev_time > 0 else 0
                self.prev_time = current_time
                
                frame = self.draw_info(frame, prediction, confidence, probas_dict, fps)
                
                # Barra de progreso
                progress = frame_count / total_frames
                bar_width = frame.shape[1] - 20
                cv2.rectangle(frame, (10, frame.shape[0]-30), 
                              (10 + int(bar_width * progress), frame.shape[0]-20), 
                              (0, 255, 0), -1)
                cv2.putText(frame, f"{frame_count}/{total_frames}", 
                            (10, frame.shape[0]-35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow(f'Clasificador V3 - {Path(video_path).name}', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                self.debug = not self.debug
                print(f"Debug: {'ON' if self.debug else 'OFF'}")
            elif key == ord(' '):
                paused = not paused
                print(f"{'PAUSADO' if paused else 'REANUDADO'}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Video cerrado")


def main():
    """Función principal con argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description='Clasificador de Movimientos en Tiempo Real V3')
    parser.add_argument('--model', type=str, default='../src/models/modelo_acciones.pkl',
                        help='Ruta al modelo entrenado')
    parser.add_argument('--video', type=str, default=None,
                        help='Ruta a video para clasificar (opcional)')
    parser.add_argument('--camera', type=int, default=0,
                        help='ID de la cámara (default: 0)')
    parser.add_argument('--debug', action='store_true',
                        help='Activar modo debug')
    
    args = parser.parse_args()
    
    # Buscar modelo en rutas posibles (desde directorio tests/)
    model_path = Path(args.model)
    
    if not model_path.is_absolute():
        # Ruta relativa desde tests/
        script_dir = Path(__file__).parent
        model_path = script_dir / args.model
    
    if not model_path.exists():
        print(f"❌ Error: No se encontró el modelo")
        print(f"   Buscado en: {model_path.absolute()}")
        print(f"   Script en: {Path(__file__).parent.absolute()}")
        return
    
    # Crear clasificador
    classifier = RealtimeMovementClassifier(model_path)
    classifier.debug = args.debug
    
    # Ejecutar
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"❌ Error: No se encontró el video: {video_path}")
            return
        classifier.run_video(video_path)
    else:
        classifier.run_webcam(args.camera)


if __name__ == "__main__":
    main()
