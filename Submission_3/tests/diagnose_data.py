"""
Script de Diagnóstico: Analizar datos de entrenamiento
Identifica problemas de balance, confusión de clases y calidad de features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

class DataDiagnostic:
    """Diagnóstico completo de datos de entrenamiento"""
    
    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)
        self.df = None
        self.report = []
        
    def load_data(self):
        """Cargar CSV"""
        print(f"Cargando: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        print(f"✓ {len(self.df)} filas, {len(self.df.columns)} columnas")
        self.report.append(f"Dataset: {len(self.df)} filas, {len(self.df.columns)} columnas")
        
    def check_class_balance(self):
        """Verificar balance de clases"""
        print("\n" + "="*60)
        print("1. BALANCE DE CLASES")
        print("="*60)
        
        class_counts = self.df['accion'].value_counts().sort_index()
        
        print("\nDistribución:")
        for action, count in class_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {action:20s}: {count:5d} ({percentage:5.1f}%)")
        
        # Verificar desbalance
        max_count = class_counts.max()
        min_count = class_counts.min()
        ratio = max_count / min_count
        
        print(f"\nRatio max/min: {ratio:.2f}x")
        
        if ratio > 2:
            self.report.append(f"⚠️ DESBALANCE DETECTADO: Ratio {ratio:.2f}x")
            print(f"⚠️ DESBALANCE: La clase más común tiene {ratio:.2f}x más datos")
        else:
            self.report.append("✓ Balance aceptable entre clases")
            print("✓ Balance aceptable")
        
        # Gráfico
        plt.figure(figsize=(10, 6))
        class_counts.plot(kind='bar', color='skyblue')
        plt.title('Distribución de Clases')
        plt.xlabel('Acción')
        plt.ylabel('Cantidad de Frames')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('diagnostic_class_balance.png', dpi=150)
        print("\n✓ Guardado: diagnostic_class_balance.png")
        
    def analyze_feature_variance(self):
        """Analizar varianza de features por clase"""
        print("\n" + "="*60)
        print("2. VARIANZA DE FEATURES POR CLASE")
        print("="*60)
        
        # Features numéricas (excluir accion y velocidad_accion)
        feature_cols = [col for col in self.df.columns if col not in ['accion', 'velocidad_accion']]
        
        print(f"\nAnalizando {len(feature_cols)} features...")
        
        variance_by_class = {}
        for action in self.df['accion'].unique():
            action_data = self.df[self.df['accion'] == action][feature_cols]
            variance_by_class[action] = action_data.var().mean()
        
        print("\nVarianza promedio por clase:")
        for action, var in sorted(variance_by_class.items(), key=lambda x: x[1], reverse=True):
            print(f"  {action:20s}: {var:.6f}")
        
        # Clases con baja varianza = movimientos estáticos
        low_var_threshold = min(variance_by_class.values()) * 1.5
        low_var_classes = [k for k, v in variance_by_class.items() if v < low_var_threshold]
        
        if low_var_classes:
            print(f"\n⚠️ Clases con baja varianza (estáticas): {low_var_classes}")
            self.report.append(f"⚠️ Clases estáticas detectadas: {', '.join(low_var_classes)}")
        
    def check_feature_correlation_confusion(self):
        """Verificar si clases tienen features similares (confusión potencial)"""
        print("\n" + "="*60)
        print("3. SIMILITUD ENTRE CLASES (Confusión Potencial)")
        print("="*60)
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        feature_cols = [col for col in self.df.columns if col not in ['accion', 'velocidad_accion']]
        
        # Promedios de features por clase
        class_means = self.df.groupby('accion')[feature_cols].mean()
        
        # Similitud entre clases
        similarity_matrix = cosine_similarity(class_means)
        
        print("\nSimilitud entre clases (cosine similarity):")
        print("(1.0 = idénticas, 0.0 = completamente diferentes)\n")
        
        classes = class_means.index.tolist()
        
        # Crear matriz de texto
        confusion_pairs = []
        for i, class1 in enumerate(classes):
            for j, class2 in enumerate(classes):
                if i < j:  # Solo triángulo superior
                    sim = similarity_matrix[i][j]
                    print(f"{class1:20s} vs {class2:20s}: {sim:.4f}", end="")
                    if sim > 0.95:
                        print(" ⚠️ MUY SIMILARES - POSIBLE CONFUSIÓN")
                        confusion_pairs.append((class1, class2, sim))
                    else:
                        print()
        
        if confusion_pairs:
            self.report.append("⚠️ PARES CONFUSOS DETECTADOS:")
            for c1, c2, sim in confusion_pairs:
                self.report.append(f"   - {c1} vs {c2}: {sim:.4f}")
        
        # Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, 
                    xticklabels=classes, 
                    yticklabels=classes,
                    annot=True, 
                    fmt='.3f',
                    cmap='RdYlGn',
                    vmin=0.8, vmax=1.0)
        plt.title('Similitud entre Clases\n(Valores altos = Posible confusión)')
        plt.tight_layout()
        plt.savefig('diagnostic_class_similarity.png', dpi=150)
        print("\n✓ Guardado: diagnostic_class_similarity.png")
        
    def analyze_static_vs_dynamic(self):
        """Identificar si hay features dinámicas (velocidad)"""
        print("\n" + "="*60)
        print("4. ANÁLISIS DE FEATURES DINÁMICAS vs ESTÁTICAS")
        print("="*60)
        
        velocity_cols = [col for col in self.df.columns if 'velocidad' in col.lower()]
        
        print(f"\nFeatures de velocidad encontradas: {len(velocity_cols)}")
        
        if len(velocity_cols) == 0:
            print("❌ NO HAY FEATURES DE VELOCIDAD")
            print("   → El modelo no puede distinguir movimiento vs quieto")
            self.report.append("❌ CRÍTICO: No hay features temporales (velocidad)")
            self.report.append("   Recomendación: Agregar velocidad de cambio entre frames")
        else:
            print(f"✓ Encontradas: {velocity_cols[:5]}...")
            
            # Verificar si velocidades son útiles
            for col in velocity_cols[:3]:
                var = self.df[col].var()
                print(f"  {col}: varianza = {var:.6f}")
        
    def check_sequence_information(self):
        """Verificar si hay información de secuencia temporal"""
        print("\n" + "="*60)
        print("5. INFORMACIÓN TEMPORAL")
        print("="*60)
        
        # Buscar columnas de frame/tiempo
        temporal_cols = [col for col in self.df.columns if any(x in col.lower() for x in ['frame', 'time', 'timestamp', 'sequence'])]
        
        if temporal_cols:
            print(f"✓ Columnas temporales: {temporal_cols}")
        else:
            print("⚠️ No hay columnas de secuencia temporal")
            print("   → Frames tratados como independientes")
            self.report.append("⚠️ Sin información de secuencia temporal")
            self.report.append("   Recomendación: Usar ventanas de tiempo (LSTM/secuencias)")
        
    def recommend_solutions(self):
        """Recomendar soluciones basadas en diagnóstico"""
        print("\n" + "="*60)
        print("6. RECOMENDACIONES")
        print("="*60)
        
        recommendations = []
        
        # Basado en análisis
        if any("DESBALANCE" in r for r in self.report):
            recommendations.append("1. BALANCEAR DATOS:")
            recommendations.append("   - Grabar más videos de clases minoritarias")
            recommendations.append("   - Data augmentation (flip, velocidad)")
            recommendations.append("   - SMOTE para balanceo sintético")
        
        if any("CRÍTICO: No hay features temporales" in r for r in self.report):
            recommendations.append("2. AGREGAR FEATURES TEMPORALES: ⚠️ CRÍTICO")
            recommendations.append("   - Velocidad de cambio (diff entre frames)")
            recommendations.append("   - Aceleración (segunda derivada)")
            recommendations.append("   - Ventanas de tiempo (últimos N frames)")
        
        if any("PARES CONFUSOS" in r for r in self.report):
            recommendations.append("3. MEJORAR SEPARABILIDAD:")
            recommendations.append("   - Agregar features discriminativas")
            recommendations.append("   - Feature engineering específico")
            recommendations.append("   - Usar modelos de secuencia (LSTM/GRU)")
        
        if any("estáticas" in r for r in self.report):
            recommendations.append("4. DIFERENCIAR ESTÁTICO vs DINÁMICO:")
            recommendations.append("   - Feature: suma de movimiento en ventana")
            recommendations.append("   - Feature: frames sin cambio > umbral")
            recommendations.append("   - Post-procesamiento: si velocidad = 0 → pararse/sentarse")
        
        for rec in recommendations:
            print(rec)
        
        self.report.extend(recommendations)
        
    def save_report(self, filename='diagnostic_report.txt'):
        """Guardar reporte completo"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("REPORTE DE DIAGNÓSTICO DE DATOS\n")
            f.write("="*60 + "\n\n")
            f.write(f"Dataset: {self.csv_path}\n\n")
            f.write("\n".join(self.report))
        
        print(f"\n✓ Reporte guardado: {filename}")
        
    def run_full_diagnostic(self):
        """Ejecutar diagnóstico completo"""
        self.load_data()
        self.check_class_balance()
        self.analyze_feature_variance()
        self.check_feature_correlation_confusion()
        self.analyze_static_vs_dynamic()
        self.check_sequence_information()
        self.recommend_solutions()
        self.save_report()
        
        print("\n" + "="*60)
        print("DIAGNÓSTICO COMPLETADO")
        print("="*60)
        print("\nArchivos generados:")
        print("  - diagnostic_class_balance.png")
        print("  - diagnostic_class_similarity.png")
        print("  - diagnostic_report.txt")


def main():
    # Ruta al CSV de entrenamiento
    csv_path = Path(__file__).parent.parent / 'src' / 'data' / 'mov_data_proccesed.csv'
    
    if not csv_path.exists():
        # Intentar ruta alternativa
        csv_path = Path(__file__).parent.parent / 'src' / 'data' / 'movement_data.csv'
    
    if not csv_path.exists():
        print(f"❌ Error: No se encontró el CSV de datos")
        print(f"   Buscado en: {csv_path.absolute()}")
        return
    
    diagnostic = DataDiagnostic(csv_path)
    diagnostic.run_full_diagnostic()
    
    print("\n" + "="*60)
    print("PRÓXIMO PASO:")
    print("="*60)
    print("Revisa los archivos generados y luego ejecuta:")
    print("  python improve_features.py  # Para agregar features temporales")


if __name__ == "__main__":
    main()
