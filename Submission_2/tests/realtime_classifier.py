"""
Sistema de Clasificación de Movimientos en Tiempo Real
Usa webcam para detectar y clasificar acciones humanas
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
from pathlib import Path
import time


class RealtimeMovementClassifier:
    """Clasificador de movimientos en tiempo real usando webcam"""
    
    def __init__(self, model_path='modelo_acciones.pkl'):
        # Cargar modelo entrenado
        print("Cargando modelo...")
        model_data = joblib.load(model_path)
        
        if isinstance(model_data, dict):
            self.model = model_data['model']
            self.label_encoder = model_data.get('label_encoder')
            self.classes = model_data.get('classes', [])
        else:
            # Compatibilidad con modelos antiguos
            self.model = model_data
            self.label_encoder = None
            self.classes = []
        
        print(f"✓ Modelo cargado")
        print(f"✓ Clases: {list(self.classes) if len(self.classes) > 0 else 'N/A'}")
        
        # MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Para FPS
        self.prev_time = 0
        
        # Historial de predicciones (suavizado)
        self.prediction_history = []
        self.history_size = 5
    
    def extract_features_from_landmarks(self, landmarks):
        """Extrae features desde landmarks de MediaPipe"""
        if not landmarks:
            return None
        
        features = []
        
        # Extraer coordenadas x, y, z de cada landmark
        for landmark in landmarks.landmark:
            features.extend([
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility
            ])
        
        return np.array(features)
    
    def calculate_angles(self, landmarks):
        """Calcula ángulos entre articulaciones (opcional)"""
        # Función helper para calcular ángulo
        def get_angle(a, b, c):
            """Calcula ángulo entre 3 puntos"""
            a = np.array([a.x, a.y])
            b = np.array([b.x, b.y])
            c = np.array([c.x, c.y])
            
            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians * 180.0 / np.pi)
            
            if angle > 180.0:
                angle = 360 - angle
            
            return angle
        
        lm = landmarks.landmark
        
        # Ángulos importantes
        angles = {}
        
        # Codo izquierdo
        angles['left_elbow'] = get_angle(
            lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            lm[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
            lm[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        )
        
        # Codo derecho
        angles['right_elbow'] = get_angle(
            lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            lm[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value],
            lm[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        )
        
        # Rodilla izquierda
        angles['left_knee'] = get_angle(
            lm[self.mp_pose.PoseLandmark.LEFT_HIP.value],
            lm[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
            lm[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        )
        
        # Rodilla derecha
        angles['right_knee'] = get_angle(
            lm[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
            lm[self.mp_pose.PoseLandmark.RIGHT_KNEE.value],
            lm[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        )
        
        return angles
    
    def smooth_prediction(self, prediction):
        """Suaviza predicciones usando historial"""
        self.prediction_history.append(prediction)
        
        # Mantener solo últimas N predicciones
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)
        
        # Votación por mayoría
        if len(self.prediction_history) >= 3:
            # Contar frecuencias
            unique, counts = np.unique(self.prediction_history, return_counts=True)
            # Retornar la más común
            return unique[np.argmax(counts)]
        
        return prediction
    
    def process_frame(self, frame):
        """Procesa un frame y retorna predicción"""
        # Convertir a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detectar pose
        results = self.pose.process(rgb_frame)
        
        prediction = None
        confidence = 0.0
        
        if results.pose_landmarks:
            # Dibujar landmarks
            self.mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            # Extraer features
            features = self.extract_features_from_landmarks(results.pose_landmarks)
            
            if features is not None:
                # Ajustar features al tamaño esperado (49 en tu caso)
                # Tu modelo espera 49 features, MediaPipe da 33 landmarks * 4 = 132
                # Debemos recortar o adaptar
                
                # IMPORTANTE: Asegúrate de que coincida con tu entrenamiento
                # Si entrenaste con 49 features, usa las mismas aquí
                features_subset = features[:49]  # Ajusta según tu caso
                
                # Predecir
                try:
                    features_reshaped = features_subset.reshape(1, -1)
                    pred_encoded = self.model.predict(features_reshaped)[0]
                    probas = self.model.predict_proba(features_reshaped)[0]
                    confidence = probas.max()
                    
                    # Decodificar si es necesario
                    if self.label_encoder is not None and isinstance(pred_encoded, (int, np.integer)):
                        prediction = self.label_encoder.inverse_transform([pred_encoded])[0]
                    else:
                        prediction = pred_encoded
                    
                    # Suavizar predicción
                    prediction = self.smooth_prediction(prediction)
                    
                except Exception as e:
                    print(f"Error en predicción: {e}")
        
        return frame, prediction, confidence, results.pose_landmarks
    
    def draw_info(self, frame, prediction, confidence, fps):
        """Dibuja información en el frame"""
        h, w = frame.shape[:2]
        
        # Fondo semitransparente para texto
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Título
        cv2.putText(frame, "Clasificador de Movimientos", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", 
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Predicción
        if prediction:
            color = (0, 255, 0) if confidence > 0.8 else (0, 165, 255)
            cv2.putText(frame, f"Accion: {prediction}", 
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Confianza: {confidence*100:.1f}%", 
                        (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cv2.putText(frame, "Sin deteccion", 
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Instrucciones
        cv2.putText(frame, "Presiona 'q' para salir", 
                    (w-300, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self, camera_id=0):
        """Ejecuta el clasificador en tiempo real"""
        print("\n" + "="*60)
        print("CLASIFICADOR DE MOVIMIENTOS EN TIEMPO REAL")
        print("="*60)
        print("Presiona 'q' para salir")
        print("Iniciando cámara...")
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ Error: No se pudo abrir la cámara")
            return
        
        # Configurar resolución
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("✓ Cámara iniciada")
        print("\nMuévete frente a la cámara para clasificar tus movimientos!")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("❌ Error al leer frame")
                break
            
            # Voltear horizontalmente (efecto espejo)
            frame = cv2.flip(frame, 1)
            
            # Procesar frame
            frame, prediction, confidence, landmarks = self.process_frame(frame)
            
            # Calcular FPS
            current_time = time.time()
            fps = 1 / (current_time - self.prev_time) if self.prev_time > 0 else 0
            self.prev_time = current_time
            
            # Dibujar información
            frame = self.draw_info(frame, prediction, confidence, fps)
            
            # Mostrar
            cv2.imshow('Clasificador de Movimientos', frame)
            
            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Limpiar
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Sistema cerrado correctamente")


def main():
    """Función principal"""
    # Ruta al modelo
    model_path = Path(__file__).parent / 'modelo_acciones.pkl'
    
    if not model_path.exists():
        print(f"❌ Error: No se encontró el modelo en {model_path}")
        print("Primero entrena el modelo ejecutando: python my_model.py")
        return
    
    # Crear clasificador
    classifier = RealtimeMovementClassifier(model_path)
    
    # Ejecutar
    classifier.run(camera_id=0)


if __name__ == "__main__":
    main()
