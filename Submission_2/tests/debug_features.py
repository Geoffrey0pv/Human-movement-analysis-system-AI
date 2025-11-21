"""
Script de debug: Verificar cuántas features extrae el clasificador V4
"""

import sys
sys.path.append('..')

from Submission_3.tests.realtime_classifier_v4 import RealtimeMovementClassifierV4
import numpy as np

# Crear clasificador
classifier = RealtimeMovementClassifierV4()

print("\n" + "="*60)
print("DEBUG: VERIFICACIÓN DE FEATURES")
print("="*60)

print(f"\nTotal features esperadas: {len(classifier.feature_names)}")
print(f"\nPrimeras 10 features:")
for i, name in enumerate(classifier.feature_names[:10]):
    print(f"  {i+1}. {name}")

print(f"\n...")
print(f"\nÚltimas 13 features (temporales):")
for i, name in enumerate(classifier.feature_names[-13:], start=len(classifier.feature_names)-12):
    print(f"  {i}. {name}")

# Simular extracción con landmarks dummy
print("\n" + "="*60)
print("TEST: Extracción con landmarks dummy")
print("="*60)

class DummyLandmark:
    def __init__(self):
        self.x = 0.5
        self.y = 0.5
        self.visibility = 0.9

dummy_landmarks = [DummyLandmark() for _ in range(33)]

features = classifier.extract_features_like_training(dummy_landmarks, None)

print(f"\nShape de features extraídas: {features.shape}")
print(f"Cantidad de features: {features.shape[1]}")

if features.shape[1] == 64:
    print("✅ CORRECTO: 64 features extraídas")
else:
    print(f"❌ ERROR: Se esperaban 64 features, se obtuvieron {features.shape[1]}")

print("\nPrimeros 10 valores:")
print(features[0, :10])

print("\nÚltimos 13 valores (temporales):")
print(features[0, -13:])
