"""
Verificar qué features espera realmente el modelo entrenado
"""
import joblib
from pathlib import Path

model_path = Path(__file__).parent.parent / 'src' / 'models' / 'modelo_acciones.pkl'
print(f"Cargando: {model_path}")

model_data = joblib.load(model_path)

print("\nContenido del archivo:")
print(f"  Tipo: {type(model_data)}")
print(f"  Keys: {list(model_data.keys()) if isinstance(model_data, dict) else 'N/A'}")

if isinstance(model_data, dict) and 'model' in model_data:
    model_pipeline = model_data['model']
    print(f"\nPipeline del modelo:")
    print(f"  Tipo: {type(model_pipeline)}")
    print(f"  Steps: {model_pipeline.steps if hasattr(model_pipeline, 'steps') else 'N/A'}")
    
    # Scaler
    if hasattr(model_pipeline, 'named_steps'):
        scaler = model_pipeline.named_steps.get('scaler')
        if scaler:
            print(f"\nScaler:")
            print(f"  n_features_in_: {scaler.n_features_in_}")
            if hasattr(scaler, 'feature_names_in_'):
                print(f"  feature_names_in_ (total {len(scaler.feature_names_in_)}):")
                for i, name in enumerate(scaler.feature_names_in_[:10]):
                    print(f"    {i+1}. {name}")
                print(f"    ...")
                for i, name in enumerate(scaler.feature_names_in_[-13:], start=len(scaler.feature_names_in_)-12):
                    print(f"    {i}. {name}")
            else:
                print("  No tiene feature_names_in_")
    
print("\n✓ Verificación completada")
