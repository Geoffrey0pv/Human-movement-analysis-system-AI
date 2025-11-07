"""
Script de prueba para entrenar y evaluar el modelo de clasificación de movimientos
Ejecutar desde: Submission_2/
"""

import sys
from pathlib import Path

# Agregar src al path para imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.my_model import MovementClassifier

def main():
    print("🚀 Iniciando entrenamiento del modelo...\n")
    
    # Crear instancia del clasificador
    # Los datos se cargarán automáticamente de: src/data/mov_data_proccesed.csv
    classifier = MovementClassifier()
    
    # Ejecutar el pipeline completo
    classifier.run_full_pipeline()
    
    print("\n✅ Proceso completado!")
    print("\nArchivos generados:")
    print("  - modelo_acciones.pkl (modelo entrenado)")
    print("  - confusion_matrix.png (visualización)")
    
    print("\n📝 Para usar el modelo en producción:")
    print("  import joblib")
    print("  modelo = joblib.load('modelo_acciones.pkl')")
    print("  prediccion = modelo.predict(datos_nuevos)")

if __name__ == "__main__":
    main()
