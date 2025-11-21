"""
Módulo de Evaluación para Modelos de Clasificación de Movimientos
Proporciona métricas detalladas y visualizaciones
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    ConfusionMatrixDisplay
)
import pandas as pd


class ModelEvaluator:
    """Evaluador completo para modelos de clasificación"""
    
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None
        self.results = {}
        
    def load_model(self, path=None):
        """Carga modelo entrenado"""
        path = path or self.model_path
        if not path or not Path(path).exists():
            raise FileNotFoundError(f"Modelo no encontrado: {path}")
        
        self.model = joblib.load(path)
        print(f"✓ Modelo cargado desde: {path}")
        return self.model
    
    def evaluate(self, X_test, y_test):
        """Evaluación completa del modelo"""
        if self.model is None:
            raise ValueError("Primero carga un modelo con load_model()")
        
        print("\n" + "="*60)
        print("EVALUACIÓN DETALLADA DEL MODELO")
        print("="*60)
        
        # Predicciones
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test) if hasattr(self.model, 'predict_proba') else None
        
        # Métricas básicas
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        # Guardar resultados
        self.results = {
            'y_true': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support
        }
        
        # Mostrar métricas
        print(f"\n📊 Métricas Generales:")
        print(f"  • Accuracy:  {accuracy:.4f}")
        print(f"  • Precision: {precision:.4f}")
        print(f"  • Recall:    {recall:.4f}")
        print(f"  • F1-Score:  {f1:.4f}")
        
        # Reporte detallado por clase
        print(f"\n📋 Reporte por Clase:")
        print(classification_report(y_test, y_pred))
        
        return self.results
    
    def plot_confusion_matrix(self, normalize=False, save_path='confusion_matrix.png'):
        """Genera y guarda matriz de confusión"""
        if not self.results:
            raise ValueError("Ejecuta evaluate() primero")
        
        y_true = self.results['y_true']
        y_pred = self.results['y_pred']
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Visualizar
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                    cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Valor Real')
        ax.set_title('Matriz de Confusión' + (' (Normalizada)' if normalize else ''))
        
        # Rotar etiquetas si son muchas
        labels = sorted(set(y_true))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels, rotation=0)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Matriz guardada: {save_path}")
        plt.show()
        
        return fig
    
    def plot_class_performance(self, save_path='class_performance.png'):
        """Gráfico de rendimiento por clase"""
        if not self.results:
            raise ValueError("Ejecuta evaluate() primero")
        
        y_true = self.results['y_true']
        y_pred = self.results['y_pred']
        
        # Métricas por clase
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        classes = sorted(set(y_true))
        
        # Crear gráfico
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(classes))
        width = 0.25
        
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Clases')
        ax.set_ylabel('Score')
        ax.set_title('Rendimiento por Clase')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico guardado: {save_path}")
        plt.show()
        
        return fig
    
    def generate_report(self, output_path='evaluation_report.txt'):
        """Genera reporte completo en texto"""
        if not self.results:
            raise ValueError("Ejecuta evaluate() primero")
        
        y_true = self.results['y_true']
        y_pred = self.results['y_pred']
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("REPORTE DE EVALUACIÓN DEL MODELO\n")
            f.write("="*70 + "\n\n")
            
            f.write("📊 MÉTRICAS GENERALES\n")
            f.write("-"*70 + "\n")
            f.write(f"Accuracy:  {self.results['accuracy']:.4f}\n")
            f.write(f"Precision: {self.results['precision']:.4f}\n")
            f.write(f"Recall:    {self.results['recall']:.4f}\n")
            f.write(f"F1-Score:  {self.results['f1_score']:.4f}\n\n")
            
            f.write("📋 REPORTE DETALLADO POR CLASE\n")
            f.write("-"*70 + "\n")
            f.write(classification_report(y_true, y_pred))
            f.write("\n")
            
            f.write("🎯 MATRIZ DE CONFUSIÓN\n")
            f.write("-"*70 + "\n")
            cm = confusion_matrix(y_true, y_pred)
            f.write(str(cm) + "\n")
        
        print(f"✓ Reporte guardado: {output_path}")
        return output_path
    
    def full_evaluation(self, X_test, y_test, output_dir='evaluation_results'):
        """Ejecuta evaluación completa y genera todos los archivos"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Evaluar
        self.evaluate(X_test, y_test)
        
        # Generar visualizaciones
        self.plot_confusion_matrix(
            save_path=output_dir / 'confusion_matrix.png'
        )
        self.plot_confusion_matrix(
            normalize=True,
            save_path=output_dir / 'confusion_matrix_normalized.png'
        )
        self.plot_class_performance(
            save_path=output_dir / 'class_performance.png'
        )
        
        # Generar reporte
        self.generate_report(
            output_path=output_dir / 'evaluation_report.txt'
        )
        
        print(f"\n✅ Evaluación completa guardada en: {output_dir.absolute()}")
        
        return self.results


# Función de conveniencia para uso rápido
def quick_evaluate(model_path, X_test, y_test):
    """Evaluación rápida de un modelo"""
    evaluator = ModelEvaluator(model_path)
    evaluator.load_model()
    return evaluator.full_evaluation(X_test, y_test)


if __name__ == "__main__":
    print("Módulo de evaluación cargado correctamente")
    print("Uso:")
    print("  from evaluation.evaluate import ModelEvaluator")
    print("  evaluator = ModelEvaluator('modelo.pkl')")
    print("  evaluator.load_model()")
    print("  evaluator.full_evaluation(X_test, y_test)")
