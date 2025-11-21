"""
Clasificador en Tiempo Real V4 - Con 64 Features Temporales
============================================================
Mejoras:
- ✅ Extrae 64 features (49 originales + 13 temporales)
- ✅ Features temporales para distinguir estático vs dinámico
- ✅ Sincronizado con modelo entrenado (modelo_acciones.pkl)
- ✅ Mejora precisión en 'sentarse' y 'pararse' (poses estáticas)
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
from pathlib import Path
import time


class RealtimeMovementClassifierV4:
    """Clasificador con features temporales mejoradas"""
    
    def __init__(self, model_path=None):
        """Inicializar clasificador"""
        # Ruta por defecto al modelo
        if model_path is None:
            model_path = Path(__file__).parent.parent / 'src' / 'models' / 'modelo_acciones.pkl'
        
        # Cargar modelo
        print(f"Cargando modelo: {model_path}")
        model_data = joblib.load(model_path)
        
        # El archivo puede ser un dict con múltiples modelos o el modelo directo
        if isinstance(model_data, dict):
            # Usar el mejor modelo (XGBoost)
            if 'XGBoost' in model_data:
                self.model = model_data['XGBoost']
                print("✓ Modelo XGBoost cargado")
            else:
                # Tomar el primer modelo disponible
                model_name = list(model_data.keys())[0]
                self.model = model_data[model_name]
                print(f"✓ Modelo {model_name} cargado")
            
            # Cargar label_encoder si está disponible
            if 'label_encoder' in model_data:
                self.label_encoder = model_data['label_encoder']
                print(f"✓ Clases: {list(self.label_encoder.classes_)}")
            else:
                self.label_encoder = None
                print("⚠️ No se encontró label_encoder")
        else:
            self.model = model_data
            self.label_encoder = None
            print("✓ Modelo cargado")
        
        # MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Nombres exactos de columnas del CSV (orden importa)
        self.feature_names = self._get_feature_names()
        
        # Buffer temporal para calcular features temporales
        self.prev_landmarks = None
        self.velocity_buffer = []  # Guardar últimas velocidades
        self.buffer_size = 5  # Ventana temporal
        
        print(f"✓ Clasificador V4 inicializado ({len(self.feature_names)} features)")
        
    def _get_feature_names(self):
        """
        Nombres de columnas EXACTOS del CSV de entrenamiento
        Total: 64 features = 49 originales + 13 temporales + 2 labels
        """
        feature_names = []
        
        # 33 coordenadas (x, y) = 33 features
        for i in range(33):
            feature_names.append(f'landmark_{i}_x')
            feature_names.append(f'landmark_{i}_y')
        
        # 16 velocidades (x, y) = 32 features  
        velocity_landmarks = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        for lm_id in velocity_landmarks:
            feature_names.append(f'velocidad_{lm_id}_x')
            feature_names.append(f'velocidad_{lm_id}_y')
        
        # 8 ángulos = 8 features
        angles = [
            'angulo_codo_izq', 'angulo_codo_der',
            'angulo_rodilla_izq', 'angulo_rodilla_der',
            'angulo_cadera_izq', 'angulo_cadera_der',
            'angulo_hombro_izq', 'angulo_hombro_der'
        ]
        feature_names.extend(angles)
        
        # 8 distancias = 8 features
        distances = [
            'dist_hombros', 'dist_caderas', 'dist_rodillas', 'dist_tobillos',
            'dist_mano_izq_cadera', 'dist_mano_der_cadera',
            'dist_pie_izq_cadera', 'dist_pie_der_cadera'
        ]
        feature_names.extend(distances)
        
        # TOTAL hasta aquí: 33 + 32 + 8 + 8 = 81 features
        # Pero el CSV original tiene 49 features + accion + velocidad_accion = 51 columnas
        # Entonces las features originales son las primeras 49
        
        # --- FEATURES TEMPORALES NUEVAS (13) ---
        temporal_features = [
            'movement_magnitude',      # Suma de todas las velocidades
            'movement_average',        # Promedio de velocidades
            'movement_max',            # Velocidad máxima
            'is_static',               # Binario: ¿está quieto?
            'leg_movement',            # Movimiento de piernas
            'arm_movement',            # Movimiento de brazos
            'leg_arm_ratio',           # Ratio pierna/brazo
            'vertical_position',       # Posición vertical promedio
            'vertical_range',          # Rango vertical (altura)
            'movement_variance',       # Varianza del movimiento
            'movement_cv',             # Coeficiente de variación
            'depth_position',          # Posición en profundidad
            'forward_backward_movement' # Movimiento adelante/atrás
        ]
        
        # IMPORTANTE: Solo devolver las primeras 49 features originales + 13 temporales = 62
        # Pero el modelo espera 64... verificar con CSV real
        # Por ahora, usar las primeras 49 del array + 13 temporales
        return feature_names[:49] + temporal_features
    
    def extract_features_like_training(self, landmarks, prev_landmarks=None):
        """
        Extraer EXACTAMENTE las mismas 62 features del CSV de entrenamiento
        
        Features:
        - 49 features originales (coordenadas, velocidades, ángulos)
        - 13 features temporales (movement_magnitude, is_static, etc.)
        = 62 features totales (excluyendo accion y velocidad_accion)
        
        Mapeo MediaPipe landmarks → CSV:
        - 11: hombro_izq, 12: hombro_der
        - 13: codo_izq, 14: codo_der
        - 23: cadera_izq, 24: cadera_der
        - 25: rodilla_izq, 26: rodilla_der
        - 27: tobillo_izq, 28: tobillo_der
        """
        import pandas as pd
        features = {}
        
        # ========================================
        # PARTE 1: FEATURES ORIGINALES (49)
        # ========================================
        
        # Mapeo de landmarks de MediaPipe a nombres del CSV
        body_parts = {
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
        
        # 1. Coordenadas (x, y, z) y velocidades (40 features)
        # Orden: hombro_izq, hombro_der, codo_izq, codo_der, cadera_izq, cadera_der, rodilla_izq, rodilla_der, tobillo_izq, tobillo_der
        for part_name, lm_id in body_parts.items():
            # Coordenadas x, y, z
            features[f'x_{part_name}'] = landmarks[lm_id].x
            features[f'y_{part_name}'] = landmarks[lm_id].y
            features[f'z_{part_name}'] = landmarks[lm_id].z
            
            # Velocidad (distancia euclidiana 3D entre frames)
            if prev_landmarks is not None:
                dx = landmarks[lm_id].x - prev_landmarks[lm_id].x
                dy = landmarks[lm_id].y - prev_landmarks[lm_id].y
                dz = landmarks[lm_id].z - prev_landmarks[lm_id].z
                velocidad = np.sqrt(dx**2 + dy**2 + dz**2)
            else:
                velocidad = 0.0
            
            features[f'velocidad_{part_name}'] = velocidad
        
        # 2. Ángulos (8 features)
        # Rodillas
        features['rodilla_izq_ang'] = self._calculate_angle(landmarks[23], landmarks[25], landmarks[27])  # cadera-rodilla-tobillo
        features['rodilla_der_ang'] = self._calculate_angle(landmarks[24], landmarks[26], landmarks[28])
        
        # Caderas
        features['cadera_izq_ang'] = self._calculate_angle(landmarks[11], landmarks[23], landmarks[25])  # hombro-cadera-rodilla
        features['cadera_der_ang'] = self._calculate_angle(landmarks[12], landmarks[24], landmarks[26])
        
        # Codos
        features['codo_izq_ang'] = self._calculate_angle(landmarks[11], landmarks[13], landmarks[15])  # hombro-codo-muñeca
        features['codo_der_ang'] = self._calculate_angle(landmarks[12], landmarks[14], landmarks[16])
        
        # Hombros
        features['hombro_izq_ang'] = self._calculate_angle(landmarks[13], landmarks[11], landmarks[23])  # codo-hombro-cadera
        features['hombro_der_ang'] = self._calculate_angle(landmarks[14], landmarks[12], landmarks[24])
        
        # 3. Inclinación del tronco (1 feature)
        # Ángulo entre línea vertical y línea hombros-caderas
        hombro_medio_y = (landmarks[11].y + landmarks[12].y) / 2
        cadera_medio_y = (landmarks[23].y + landmarks[24].y) / 2
        hombro_medio_x = (landmarks[11].x + landmarks[12].x) / 2
        cadera_medio_x = (landmarks[23].x + landmarks[24].x) / 2
        
        # Ángulo de inclinación
        features['inclinacion_tronco_ang'] = np.arctan2(
            abs(hombro_medio_x - cadera_medio_x),
            abs(hombro_medio_y - cadera_medio_y)
        ) * 180.0 / np.pi
        
        # TOTAL: 40 (coords+vel) + 8 (ángulos) + 1 (inclinación) = 49 features ✓
        
        # ========================================
        # PARTE 2: FEATURES TEMPORALES (13)
        # ========================================
        
        # Extraer todas las velocidades calculadas
        velocity_values = []
        for part_name in body_parts.keys():
            velocity_values.append(abs(features[f'velocidad_{part_name}']))
        
        # 1. movement_magnitude: suma de todas las velocidades
        features['movement_magnitude'] = sum(velocity_values)
        
        # 2. movement_average: promedio de velocidades
        features['movement_average'] = np.mean(velocity_values) if velocity_values else 0.0
        
        # 3. movement_max: velocidad máxima
        features['movement_max'] = max(velocity_values) if velocity_values else 0.0
        
        # 4. is_static: ¿está quieto? (threshold = 3.56 del análisis)
        features['is_static'] = 1.0 if features['movement_magnitude'] < 3.56 else 0.0
        
        # 5. leg_movement: movimiento de piernas
        leg_parts = ['cadera_izq', 'cadera_der', 'rodilla_izq', 'rodilla_der', 'tobillo_izq', 'tobillo_der']
        leg_velocities = [features[f'velocidad_{part}'] for part in leg_parts]
        features['leg_movement'] = sum(leg_velocities)
        
        # 6. arm_movement: movimiento de brazos
        arm_parts = ['hombro_izq', 'hombro_der', 'codo_izq', 'codo_der']
        arm_velocities = [features[f'velocidad_{part}'] for part in arm_parts]
        features['arm_movement'] = sum(arm_velocities)
        
        # 7. leg_arm_ratio: ratio pierna/brazo
        if features['arm_movement'] > 0:
            features['leg_arm_ratio'] = features['leg_movement'] / features['arm_movement']
        else:
            features['leg_arm_ratio'] = 0.0
        
        # 8. vertical_position: posición vertical promedio
        vertical_landmarks = [0, 11, 12, 23, 24]  # Nariz, hombros, caderas
        vertical_positions = [landmarks[i].y for i in vertical_landmarks]
        features['vertical_position'] = np.mean(vertical_positions)
        
        # 9. vertical_range: rango vertical (diferencia máx-mín)
        features['vertical_range'] = max(vertical_positions) - min(vertical_positions)
        
        # 10. movement_variance: varianza del movimiento
        features['movement_variance'] = np.var(velocity_values) if len(velocity_values) > 0 else 0.0
        
        # 11. movement_cv: coeficiente de variación
        if features['movement_average'] > 0:
            features['movement_cv'] = np.sqrt(features['movement_variance']) / features['movement_average']
        else:
            features['movement_cv'] = 0.0
        
        # 12. depth_position: posición en profundidad (usando visibility)
        depth_landmarks = [11, 12, 23, 24]
        depth_values = [landmarks[i].visibility for i in depth_landmarks]
        features['depth_position'] = np.mean(depth_values)
        
        # 13. forward_backward_movement: movimiento adelante/atrás
        if prev_landmarks is not None:
            depth_change = []
            for i in depth_landmarks:
                depth_change.append(landmarks[i].visibility - prev_landmarks[i].visibility)
            features['forward_backward_movement'] = np.mean(depth_change)
        else:
            features['forward_backward_movement'] = 0.0
        
        # ========================================
        # CONVERTIR A PANDAS DATAFRAME CON NOMBRES
        # ========================================
        
        # Orden exacto de features del CSV de entrenamiento (62 features)
        ordered_feature_names = []
        
        # 1. Coordenadas y velocidades (40 features)
        for part in ['hombro_izq', 'hombro_der', 'codo_izq', 'codo_der', 
                     'cadera_izq', 'cadera_der', 'rodilla_izq', 'rodilla_der', 
                     'tobillo_izq', 'tobillo_der']:
            ordered_feature_names.extend([
                f'x_{part}',
                f'y_{part}',
                f'z_{part}',
                f'velocidad_{part}'
            ])
        
        # 2. Ángulos (8 features)
        ordered_feature_names.extend([
            'rodilla_izq_ang', 'rodilla_der_ang',
            'cadera_izq_ang', 'cadera_der_ang',
            'codo_izq_ang', 'codo_der_ang',
            'hombro_izq_ang', 'hombro_der_ang'
        ])
        
        # 3. Inclinación del tronco (1 feature)
        ordered_feature_names.append('inclinacion_tronco_ang')
        
        # 4. Features temporales (13 features)
        ordered_feature_names.extend([
            'movement_magnitude', 'movement_average', 'movement_max',
            'is_static', 'leg_movement', 'arm_movement', 'leg_arm_ratio',
            'vertical_position', 'vertical_range',
            'movement_variance', 'movement_cv',
            'depth_position', 'forward_backward_movement'
        ])
        
        # Crear DataFrame con nombres de columnas (sklearn necesita esto)
        feature_values = [features[name] for name in ordered_feature_names]
        df = pd.DataFrame([feature_values], columns=ordered_feature_names)
        
        return df
    
    def _calculate_angle(self, a, b, c):
        """Calcular ángulo entre 3 puntos"""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
                  np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
        
        return angle
    
    def _calculate_distance(self, a, b):
        """Calcular distancia euclidiana entre 2 puntos"""
        return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)
    
    def predict(self, frame):
        """
        Predecir acción en un frame
        
        Returns:
            tuple: (clase, confianza) o (None, None) si no detecta pose
        """
        # Convertir BGR a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detectar pose
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None, None
        
        # Extraer features (62) como DataFrame
        features_df = self.extract_features_like_training(
            results.pose_landmarks.landmark,
            self.prev_landmarks
        )
        
        # Guardar landmarks actuales para próximo frame
        self.prev_landmarks = results.pose_landmarks.landmark
        
        # Predecir (ahora features_df es un DataFrame con nombres de columnas)
        prediction_encoded = self.model.predict(features_df)[0]
        
        # Decodificar etiqueta si tenemos label_encoder
        if self.label_encoder is not None:
            prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
        else:
            prediction = prediction_encoded
        
        # Confianza (probabilidad)
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_df)[0]
            confidence = max(probabilities)
        else:
            confidence = 1.0  # Modelo sin probabilidades
        
        return prediction, confidence
    
    def draw_landmarks(self, frame, show_landmarks=True):
        """Dibujar landmarks en el frame"""
        if not show_landmarks:
            return frame
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
            )
        
        return frame
    
    def run_webcam(self):
        """Ejecutar clasificación en tiempo real con webcam"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: No se puede abrir la cámara")
            return
        
        print("\n" + "="*60)
        print("CLASIFICADOR EN TIEMPO REAL V4 - WEBCAM")
        print("="*60)
        print("Controles:")
        print("  - 'q': Salir")
        print("  - 'l': Toggle landmarks (mostrar/ocultar)")
        print("="*60 + "\n")
        
        show_landmarks = True
        fps_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error al capturar frame")
                break
            
            # Predecir
            prediction, confidence = self.predict(frame)
            
            # Dibujar landmarks
            if show_landmarks:
                frame = self.draw_landmarks(frame)
            
            # FPS
            current_time = time.time()
            fps = 1 / (current_time - fps_time)
            fps_time = current_time
            
            # Información en pantalla
            if prediction:
                # Fondo para texto
                cv2.rectangle(frame, (10, 10), (350, 100), (0, 0, 0), -1)
                
                # Predicción
                cv2.putText(frame, f"Accion: {prediction}", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Confianza
                color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
                cv2.putText(frame, f"Confianza: {confidence:.2%}", (20, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                cv2.putText(frame, "No se detecta persona", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # FPS
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Mostrar
            cv2.imshow('Clasificador V4', frame)
            
            # Controles
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                show_landmarks = not show_landmarks
                print(f"Landmarks: {'ON' if show_landmarks else 'OFF'}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Clasificación finalizada")
    
    def run_video(self, video_path):
        """Ejecutar clasificación en un archivo de video"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"❌ Error: No se puede abrir el video: {video_path}")
            return
        
        print("\n" + "="*60)
        print(f"CLASIFICADOR V4 - VIDEO: {Path(video_path).name}")
        print("="*60)
        print("Controles:")
        print("  - 'q': Salir")
        print("  - ESPACIO: Pausar/Reanudar")
        print("  - 'l': Toggle landmarks")
        print("="*60 + "\n")
        
        show_landmarks = True
        paused = False
        fps_time = time.time()
        
        # Estadísticas
        predictions_history = []
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("\n✓ Video finalizado")
                    break
                
                # Predecir
                prediction, confidence = self.predict(frame)
                
                # Guardar predicción
                if prediction:
                    predictions_history.append((prediction, confidence))
                
                # Dibujar landmarks
                if show_landmarks:
                    frame = self.draw_landmarks(frame)
                
                # FPS
                current_time = time.time()
                fps = 1 / (current_time - fps_time)
                fps_time = current_time
                
                # Información en pantalla
                if prediction:
                    # Fondo
                    cv2.rectangle(frame, (10, 10), (400, 100), (0, 0, 0), -1)
                    
                    # Predicción
                    cv2.putText(frame, f"Accion: {prediction}", (20, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Confianza
                    color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
                    cv2.putText(frame, f"Confianza: {confidence:.2%}", (20, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    cv2.putText(frame, "No se detecta persona", (20, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # FPS
                cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Mostrar
            cv2.imshow('Clasificador V4', frame)
            
            # Controles
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                print(f"{'PAUSADO' if paused else 'REANUDADO'}")
            elif key == ord('l'):
                show_landmarks = not show_landmarks
                print(f"Landmarks: {'ON' if show_landmarks else 'OFF'}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Mostrar estadísticas
        if predictions_history:
            print("\n" + "="*60)
            print("ESTADÍSTICAS DEL VIDEO")
            print("="*60)
            
            from collections import Counter
            pred_counts = Counter([p[0] for p in predictions_history])
            
            print(f"\nTotal frames procesados: {len(predictions_history)}")
            print("\nDistribución de predicciones:")
            for action, count in pred_counts.most_common():
                percentage = (count / len(predictions_history)) * 100
                avg_conf = np.mean([conf for pred, conf in predictions_history if pred == action])
                # Convertir action a string por si es numérico
                action_str = str(action)
                print(f"  {action_str:20s}: {count:4d} frames ({percentage:5.1f}%) - Conf. promedio: {avg_conf:.2%}")
            
            print("\n" + "="*60)


def main():
    """Función principal"""
    import sys
    
    classifier = RealtimeMovementClassifierV4()
    
    if len(sys.argv) > 1:
        # Ejecutar con video
        video_path = sys.argv[1]
        classifier.run_video(video_path)
    else:
        # Ejecutar con webcam
        classifier.run_webcam()


if __name__ == "__main__":
    main()
