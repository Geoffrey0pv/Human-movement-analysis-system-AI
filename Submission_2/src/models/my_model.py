"""
Sistema de Clasificación de Movimientos Humanos
Entrena y evalúa modelos ML para clasificar acciones desde datos de pose
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib 
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Importar evaluador
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.evaluate import ModelEvaluator


class MovementClassifier:
    """Clasificador de movimientos humanos usando ML"""
    
    def __init__(self, data_path=None):
        self.data_path = data_path or Path(__file__).parent.parent / "data" / "mov_data_proccesed.csv"
        self.X_train = None
        self.X_test = None
        self.y_train = None  # Etiquetas numéricas
        self.y_test = None   # Etiquetas numéricas
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.models = {}
        
    def load_data(self):
        """Carga y prepara los datos"""
        try:
            df = pd.read_csv(self.data_path)
            print(f"✓ Datos cargados: {df.shape[0]} frames, {df.shape[1]} columnas")
            print(f"✓ Acciones: {df['accion'].unique()}")
            
            # Separar features (X) y target (y)
            y = df['accion']
            X = df.drop(columns=['accion', 'velocidad_accion'])
            
            print(f"✓ Features: {X.shape[1]} columnas")
            return X, y
            
        except FileNotFoundError:
            raise FileNotFoundError(f"No se encontró: {self.data_path}")
    
    def split_data(self, X, y, test_size=0.2):
        """Divide datos en entrenamiento y prueba"""
        # Codificar TODAS las etiquetas desde el inicio
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Trabajar SOLO con etiquetas numéricas
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=test_size, stratify=y_encoded, random_state=42
        )
        
        print(f"✓ Train: {len(self.X_train)} | Test: {len(self.X_test)}")
        print(f"✓ Etiquetas codificadas: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
    
    def create_pipelines(self):
        """Crea pipelines de modelos con estandarización"""
        self.models = {
            'RandomForest': Pipeline([
                ('scaler', StandardScaler()),
                ('model', RandomForestClassifier(n_estimators=100, random_state=42))
            ]),
            'SVM': Pipeline([
                ('scaler', StandardScaler()),
                ('model', SVC(kernel='rbf', probability=True, random_state=42))
            ]),
            'XGBoost': Pipeline([
                ('scaler', StandardScaler()),
                ('model', XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss'))
            ])
        }
        print(f"✓ Modelos creados: {list(self.models.keys())}")
    
    def evaluate_with_kfold(self, n_splits=5):
        """Evalúa modelos usando K-Fold Cross-Validation"""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        results = {}
        
        for name, pipeline in self.models.items():
            print(f"\n--- Evaluando {name} (K-Fold) ---")
            
            # TODOS los modelos usan etiquetas numéricas ahora
            scores = cross_val_score(
                pipeline, self.X_train, self.y_train, 
                cv=kf, scoring='f1_weighted'
            )
            results[name] = {
                'scores': scores,
                'mean': scores.mean(),
                'std': scores.std()
            }
            print(f"F1-Score por fold: {np.round(scores, 3)}")
            print(f"Media: {scores.mean():.3f} (+/- {scores.std():.3f})")
        
        return results
    
    def train_best_model(self, results):
        """Entrena el mejor modelo según K-Fold"""
        best_name = max(results, key=lambda x: results[x]['mean'])
        print(f"\n✓ Mejor modelo: {best_name} (F1={results[best_name]['mean']:.3f})")
        
        self.best_model = self.models[best_name]
        
        # Todos los modelos usan etiquetas numéricas
        self.best_model.fit(self.X_train, self.y_train)
        print("✓ Entrenamiento completado")
        
        return best_name
    
    def evaluate_final(self):
        """Evalúa modelo final en datos de test"""
        print("\n--- Evaluación Final (Test Set) ---")
        
        # Predecir (siempre números)
        y_pred = self.best_model.predict(self.X_test)
        
        # Decodificar SOLO para mostrar
        y_test_decoded = self.label_encoder.inverse_transform(self.y_test)
        y_pred_decoded = self.label_encoder.inverse_transform(y_pred)
        
        # Reporte de clasificación (con nombres legibles)
        print(classification_report(y_test_decoded, y_pred_decoded))
        
        # Matriz de confusión
        cm = confusion_matrix(self.y_test, y_pred)  # Con números
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        plt.title('Matriz de Confusión - Modelo Final')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        print("✓ Matriz guardada: confusion_matrix.png")
        plt.show()
    
    def save_model(self, filename='modelo_acciones.pkl'):
        """Guarda el modelo entrenado con el encoder"""
        output_path = Path(filename)
        
        # Guardar modelo y encoder juntos
        model_data = {
            'model': self.best_model,
            'label_encoder': self.label_encoder,
            'classes': self.label_encoder.classes_
        }
        
        joblib.dump(model_data, output_path)
        print(f"✓ Modelo guardado: {output_path.absolute()}")
        print(f"✓ Clases: {list(self.label_encoder.classes_)}")
        return output_path
    
    def detailed_evaluation(self, output_dir='evaluation_results'):
        """Evaluación detallada usando ModelEvaluator"""
        print("\n" + "="*60)
        print("EVALUACIÓN DETALLADA CON VISUALIZACIONES")
        print("="*60)
        
        # Decodificar para evaluación legible
        y_test_decoded = self.label_encoder.inverse_transform(self.y_test)
        
        # Wrapper simple que decodifica predicciones
        class DecodedModelWrapper:
            def __init__(self, model, encoder):
                self.model = model
                self.encoder = encoder
            
            def predict(self, X):
                pred_encoded = self.model.predict(X)
                return self.encoder.inverse_transform(pred_encoded)
            
            def predict_proba(self, X):
                return self.model.predict_proba(X)
        
        evaluator = ModelEvaluator()
        evaluator.model = DecodedModelWrapper(self.best_model, self.label_encoder)
        evaluator.full_evaluation(self.X_test, y_test_decoded, output_dir)
        
        return evaluator
    
    def run_full_pipeline(self):
        """Ejecuta el pipeline completo de entrenamiento"""
        print("="*60)
        print("ENTRENAMIENTO DE CLASIFICADOR DE MOVIMIENTOS")
        print("="*60)
        
        # 1. Cargar datos
        X, y = self.load_data()
        
        # 2. Dividir datos
        self.split_data(X, y)
        
        # 3. Crear modelos
        self.create_pipelines()
        
        # 4. Evaluar con K-Fold
        results = self.evaluate_with_kfold()
        
        # 5. Entrenar mejor modelo
        best_name = self.train_best_model(results)
        
        # 6. Evaluación básica
        self.evaluate_final()
        
        # 7. Evaluación detallada
        self.detailed_evaluation()
        
        # 8. Guardar modelo
        self.save_model()
        
        print("\n" + "="*60)
        print("✓ PIPELINE COMPLETADO")
        print("="*60)
        print(f"\n🏆 Modelo ganador: {best_name}")
        print(f"📊 F1-Score: {results[best_name]['mean']:.4f}")
        print("\n📁 Archivos generados:")
        print("  • modelo_acciones.pkl")
        print("  • evaluation_results/")
        print("    - confusion_matrix.png")
        print("    - confusion_matrix_normalized.png")
        print("    - class_performance.png")
        print("    - evaluation_report.txt")


if __name__ == "__main__":
    # Entrenar modelo
    classifier = MovementClassifier()
    classifier.run_full_pipeline()