"""
Script de Optimización de Hiperparámetros
Encuentra los mejores parámetros para cada modelo usando GridSearchCV
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


class HyperparameterOptimizer:
    """Optimiza hiperparámetros de modelos ML"""
    
    def __init__(self, data_path=None):
        self.data_path = data_path or Path(__file__).parent.parent / "data" / "mov_data_proccesed.csv"
        self.label_encoder = LabelEncoder()
        self.best_models = {}
        
    def load_and_split_data(self):
        """Carga y prepara datos"""
        df = pd.read_csv(self.data_path)
        print(f"✓ Datos cargados: {df.shape[0]} frames")
        
        y = df['accion']
        X = df.drop(columns=['accion', 'velocidad_accion'])
        
        # Codificar para XGBoost
        y_encoded = self.label_encoder.fit_transform(y)
        
        # División estratificada
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        _, _, y_train_encoded, y_test_encoded = train_test_split(
            X, y_encoded, test_size=0.2, stratify=y, random_state=42
        )
        
        return X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded
    
    def optimize_random_forest(self, X_train, y_train):
        """Optimiza RandomForest"""
        print("\n" + "="*60)
        print("OPTIMIZANDO RANDOM FOREST")
        print("="*60)
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(random_state=42))
        ])
        
        # Grid de parámetros
        param_grid = {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [10, 20, 30, None],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }
        
        print(f"Probando {np.prod([len(v) for v in param_grid.values()])} combinaciones...")
        
        grid = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=5, 
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid.fit(X_train, y_train)
        
        print(f"\n✓ Mejores parámetros: {grid.best_params_}")
        print(f"✓ Mejor F1-Score: {grid.best_score_:.4f}")
        
        self.best_models['RandomForest'] = grid.best_estimator_
        return grid.best_estimator_, grid.best_params_, grid.best_score_
    
    def optimize_svm(self, X_train, y_train):
        """Optimiza SVM"""
        print("\n" + "="*60)
        print("OPTIMIZANDO SVM")
        print("="*60)
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVC(probability=True, random_state=42))
        ])
        
        # Grid de parámetros
        param_grid = {
            'model__C': [0.1, 1, 10, 100],
            'model__kernel': ['rbf', 'poly'],
            'model__gamma': ['scale', 'auto', 0.1, 0.01]
        }
        
        print(f"Probando {np.prod([len(v) for v in param_grid.values()])} combinaciones...")
        
        grid = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=5, 
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid.fit(X_train, y_train)
        
        print(f"\n✓ Mejores parámetros: {grid.best_params_}")
        print(f"✓ Mejor F1-Score: {grid.best_score_:.4f}")
        
        self.best_models['SVM'] = grid.best_estimator_
        return grid.best_estimator_, grid.best_params_, grid.best_score_
    
    def optimize_xgboost(self, X_train, y_train_encoded):
        """Optimiza XGBoost"""
        print("\n" + "="*60)
        print("OPTIMIZANDO XGBOOST")
        print("="*60)
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', XGBClassifier(random_state=42, eval_metric='mlogloss'))
        ])
        
        # Grid de parámetros
        param_grid = {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [3, 5, 7, 10],
            'model__learning_rate': [0.01, 0.1, 0.3],
            'model__subsample': [0.8, 0.9, 1.0],
            'model__colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        print(f"Probando {np.prod([len(v) for v in param_grid.values()])} combinaciones...")
        print("⚠️ Esto puede tardar varios minutos...")
        
        grid = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=5, 
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid.fit(X_train, y_train_encoded)
        
        print(f"\n✓ Mejores parámetros: {grid.best_params_}")
        print(f"✓ Mejor F1-Score: {grid.best_score_:.4f}")
        
        self.best_models['XGBoost'] = grid.best_estimator_
        return grid.best_estimator_, grid.best_params_, grid.best_score_
    
    def run_optimization(self):
        """Ejecuta optimización completa"""
        print("="*60)
        print("OPTIMIZACIÓN DE HIPERPARÁMETROS")
        print("="*60)
        
        # Cargar datos
        X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded = self.load_and_split_data()
        
        results = {}
        
        # 1. RandomForest
        model_rf, params_rf, score_rf = self.optimize_random_forest(X_train, y_train)
        results['RandomForest'] = {
            'model': model_rf,
            'params': params_rf,
            'score': score_rf
        }
        
        # 2. SVM
        model_svm, params_svm, score_svm = self.optimize_svm(X_train, y_train)
        results['SVM'] = {
            'model': model_svm,
            'params': params_svm,
            'score': score_svm
        }
        
        # 3. XGBoost
        model_xgb, params_xgb, score_xgb = self.optimize_xgboost(X_train, y_train_encoded)
        results['XGBoost'] = {
            'model': model_xgb,
            'params': params_xgb,
            'score': score_xgb
        }
        
        # Resumen
        print("\n" + "="*60)
        print("RESUMEN DE OPTIMIZACIÓN")
        print("="*60)
        
        for name, data in results.items():
            print(f"\n{name}:")
            print(f"  F1-Score: {data['score']:.4f}")
            print(f"  Parámetros: {data['params']}")
        
        # Mejor modelo
        best_name = max(results, key=lambda x: results[x]['score'])
        best_model = results[best_name]['model']
        
        print(f"\n🏆 MEJOR MODELO: {best_name} (F1={results[best_name]['score']:.4f})")
        
        # Guardar
        model_data = {
            'model': best_model,
            'label_encoder': self.label_encoder,
            'classes': self.label_encoder.classes_,
            'best_params': results[best_name]['params'],
            'optimization_results': results
        }
        
        joblib.dump(model_data, 'modelo_optimizado.pkl')
        print(f"\n✓ Modelo optimizado guardado: modelo_optimizado.pkl")
        
        return results, best_model


if __name__ == "__main__":
    optimizer = HyperparameterOptimizer()
    results, best_model = optimizer.run_optimization()
