"""
Script de Mejora: Agregar Features Temporales para Separar Clases
Soluciona: confusión entre clases estáticas vs dinámicas
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class TemporalFeatureEngineer:
    """Agregar features temporales críticas para distinguir movimiento"""
    
    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)
        self.df = None
        self.df_improved = None
        
    def load_data(self):
        """Cargar CSV"""
        print(f"Cargando: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        
        # Convertir columnas numéricas (excepto 'accion' y 'velocidad_accion')
        for col in self.df.columns:
            if col not in ['accion', 'velocidad_accion']:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Eliminar filas con NaN (si las hay)
        if self.df.isnull().any().any():
            before = len(self.df)
            self.df = self.df.dropna()
            print(f"⚠️ Eliminadas {before - len(self.df)} filas con valores inv\u00e1lidos")
        
        print(f"✓ {len(self.df)} filas, {len(self.df.columns)} columnas")
        
    def add_movement_magnitude(self):
        """
        Feature CRÍTICA: Magnitud de movimiento
        Suma de todas las velocidades -> distingue quieto vs movimiento
        """
        print("\n" + "="*60)
        print("1. AGREGANDO: Magnitud de Movimiento Total")
        print("="*60)
        
        velocity_cols = [col for col in self.df.columns 
                         if 'velocidad' in col.lower() and col != 'velocidad_accion']
        
        # Suma de todas las velocidades (movimiento total)
        self.df['movement_magnitude'] = self.df[velocity_cols].abs().sum(axis=1)
        
        # Promedio de movimiento
        self.df['movement_average'] = self.df[velocity_cols].abs().mean(axis=1)
        
        # Máximo movimiento (parte del cuerpo con más movimiento)
        self.df['movement_max'] = self.df[velocity_cols].abs().max(axis=1)
        
        print("✓ Agregadas 3 features de magnitud de movimiento")
        
        # Mostrar estadísticas por clase
        print("\nMagnitud de movimiento por clase:")
        for action in sorted(self.df['accion'].unique()):
            mag = self.df[self.df['accion'] == action]['movement_magnitude'].mean()
            print(f"  {action:20s}: {mag:8.2f}")
        
    def add_is_static_feature(self):
        """
        Feature CRÍTICA: ¿Está quieto?
        Binario: movimiento < umbral -> quieto (pararse/sentarse)
        """
        print("\n" + "="*60)
        print("2. AGREGANDO: Feature Estático/Dinámico")
        print("="*60)
        
        # Umbral de movimiento (ajustar basado en datos)
        threshold = self.df['movement_magnitude'].quantile(0.25)
        
        self.df['is_static'] = (self.df['movement_magnitude'] < threshold).astype(int)
        
        print(f"✓ Umbral de movimiento: {threshold:.2f}")
        print(f"✓ Frames estáticos: {self.df['is_static'].sum()} ({self.df['is_static'].mean()*100:.1f}%)")
        
        # Estáticos por clase
        print("\n% de frames estáticos por clase:")
        for action in sorted(self.df['accion'].unique()):
            static_pct = self.df[self.df['accion'] == action]['is_static'].mean() * 100
            print(f"  {action:20s}: {static_pct:5.1f}%")
        
    def add_body_part_ratios(self):
        """
        Features: Ratios de movimiento por parte del cuerpo
        Ej: piernas moviéndose más que brazos -> caminar
        """
        print("\n" + "="*60)
        print("3. AGREGANDO: Ratios de Partes del Cuerpo")
        print("="*60)
        
        # Movimiento de piernas
        leg_velocity_cols = [col for col in self.df.columns if 'velocidad' in col and any(x in col for x in ['cadera', 'rodilla', 'tobillo'])]
        self.df['leg_movement'] = self.df[leg_velocity_cols].abs().mean(axis=1)
        
        # Movimiento de brazos
        arm_velocity_cols = [col for col in self.df.columns if 'velocidad' in col and any(x in col for x in ['hombro', 'codo'])]
        self.df['arm_movement'] = self.df[arm_velocity_cols].abs().mean(axis=1)
        
        # Ratio piernas/brazos (evitar división por cero)
        self.df['leg_arm_ratio'] = self.df['leg_movement'] / (self.df['arm_movement'] + 0.001)
        
        print("✓ Agregadas 3 features de ratios corporales")
        
        # Mostrar ratios por clase
        print("\nRatio piernas/brazos por clase:")
        for action in sorted(self.df['accion'].unique()):
            ratio = self.df[self.df['accion'] == action]['leg_arm_ratio'].mean()
            print(f"  {action:20s}: {ratio:6.2f}")
        
    def add_posture_features(self):
        """
        Features de postura: altura relativa, inclinación
        Útil para distinguir sentarse vs pararse
        """
        print("\n" + "="*60)
        print("4. AGREGANDO: Features de Postura")
        print("="*60)
        
        # Altura relativa (ya existe inclinacion_tronco_ang)
        # Agregar: posición vertical promedio
        y_cols = [col for col in self.df.columns if col.startswith('y_')]
        self.df['vertical_position'] = self.df[y_cols].mean(axis=1)
        
        # Rango vertical (diferencia min-max en Y)
        self.df['vertical_range'] = self.df[y_cols].max(axis=1) - self.df[y_cols].min(axis=1)
        
        print("✓ Agregadas 2 features de postura")
        
        # Altura promedio por clase
        print("\nPosición vertical promedio por clase:")
        for action in sorted(self.df['accion'].unique()):
            pos = self.df[self.df['accion'] == action]['vertical_position'].mean()
            print(f"  {action:20s}: {pos:6.3f}")
        
    def add_velocity_variance(self):
        """
        Feature: Varianza de velocidades
        Alto = movimiento irregular (girar), Bajo = movimiento uniforme (caminar)
        """
        print("\n" + "="*60)
        print("5. AGREGANDO: Varianza de Movimiento")
        print("="*60)
        
        velocity_cols = [col for col in self.df.columns if 'velocidad' in col.lower() and col != 'velocidad_accion']
        
        # Varianza de velocidades (irregularidad del movimiento)
        self.df['movement_variance'] = self.df[velocity_cols].var(axis=1)
        
        # Coeficiente de variación
        self.df['movement_cv'] = self.df['movement_variance'] / (self.df['movement_average'] + 0.001)
        
        print("✓ Agregadas 2 features de varianza de movimiento")
        
        # Varianza por clase
        print("\nVarianza de movimiento por clase:")
        for action in sorted(self.df['accion'].unique()):
            var = self.df[self.df['accion'] == action]['movement_variance'].mean()
            print(f"  {action:20s}: {var:8.2f}")
        
    def add_direction_features(self):
        """
        Features de dirección: movimiento hacia adelante/atrás
        """
        print("\n" + "="*60)
        print("6. AGREGANDO: Features de Dirección")
        print("="*60)
        
        # Cambio en Z promedio (profundidad - adelante/atrás)
        z_cols = [col for col in self.df.columns if col.startswith('z_') and 'velocidad' not in col]
        
        if z_cols:
            # Promedio de posición en Z
            self.df['depth_position'] = self.df[z_cols].mean(axis=1)
            
            # Velocidad en Z promedio (indica dirección adelante/atrás)
            z_velocity_cols = [col.replace('z_', 'velocidad_').replace('_izq', '_izq').replace('_der', '_der') 
                               for col in z_cols if col.replace('z_', 'velocidad_') in self.df.columns]
            
            if z_velocity_cols:
                self.df['forward_backward_movement'] = self.df[z_velocity_cols].mean(axis=1)
                print("✓ Agregadas 2 features de dirección")
            else:
                print("⚠️ No hay velocidades en Z disponibles")
        else:
            print("⚠️ No hay coordenadas Z disponibles")
        
    def visualize_improvements(self):
        """Visualizar cómo las nuevas features separan las clases"""
        print("\n" + "="*60)
        print("7. VISUALIZANDO MEJORAS")
        print("="*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Magnitud de movimiento por clase
        self.df.boxplot(column='movement_magnitude', by='accion', ax=axes[0, 0])
        axes[0, 0].set_title('Magnitud de Movimiento por Clase')
        axes[0, 0].set_xlabel('Acción')
        axes[0, 0].set_ylabel('Magnitud')
        
        # 2. Estático vs Dinámico
        static_counts = self.df.groupby(['accion', 'is_static']).size().unstack(fill_value=0)
        static_counts.plot(kind='bar', stacked=True, ax=axes[0, 1])
        axes[0, 1].set_title('Frames Estáticos vs Dinámicos')
        axes[0, 1].set_xlabel('Acción')
        axes[0, 1].set_ylabel('Cantidad de Frames')
        axes[0, 1].legend(['Dinámico', 'Estático'])
        
        # 3. Ratio piernas/brazos
        self.df.boxplot(column='leg_arm_ratio', by='accion', ax=axes[1, 0])
        axes[1, 0].set_title('Ratio Movimiento Piernas/Brazos')
        axes[1, 0].set_xlabel('Acción')
        axes[1, 0].set_ylabel('Ratio')
        
        # 4. Posición vertical
        self.df.boxplot(column='vertical_position', by='accion', ax=axes[1, 1])
        axes[1, 1].set_title('Posición Vertical Promedio')
        axes[1, 1].set_xlabel('Acción')
        axes[1, 1].set_ylabel('Posición Y')
        
        plt.tight_layout()
        plt.savefig('improved_features_visualization.png', dpi=150)
        print("✓ Guardado: improved_features_visualization.png")
        
    def save_improved_dataset(self, output_path=None):
        """Guardar dataset mejorado"""
        print("\n" + "="*60)
        print("8. GUARDANDO DATASET MEJORADO")
        print("="*60)
        
        if output_path is None:
            output_path = self.csv_path.parent / f"{self.csv_path.stem}_temporal_features.csv"
        
        self.df.to_csv(output_path, index=False)
        
        print(f"✓ Guardado: {output_path}")
        print(f"  Filas: {len(self.df)}")
        print(f"  Columnas: {len(self.df.columns)} (antes: {len(self.df.columns) - len(self.get_new_features())})")
        print(f"  Nuevas features: {len(self.get_new_features())}")
        
        return output_path
        
    def get_new_features(self):
        """Lista de nuevas features agregadas"""
        new_features = [
            'movement_magnitude',
            'movement_average',
            'movement_max',
            'is_static',
            'leg_movement',
            'arm_movement',
            'leg_arm_ratio',
            'vertical_position',
            'vertical_range',
            'movement_variance',
            'movement_cv',
        ]
        
        # Agregar features condicionales
        if 'depth_position' in self.df.columns:
            new_features.append('depth_position')
        if 'forward_backward_movement' in self.df.columns:
            new_features.append('forward_backward_movement')
        
        return new_features
        
    def run_improvement(self):
        """Ejecutar pipeline completo de mejora"""
        self.load_data()
        self.add_movement_magnitude()
        self.add_is_static_feature()
        self.add_body_part_ratios()
        self.add_posture_features()
        self.add_velocity_variance()
        self.add_direction_features()
        self.visualize_improvements()
        output_path = self.save_improved_dataset()
        
        print("\n" + "="*60)
        print("MEJORA COMPLETADA ✓")
        print("="*60)
        print(f"\nDataset mejorado: {output_path}")
        print("\nPRÓXIMO PASO:")
        print("  cd ../src/models")
        print("  python my_model.py  # Re-entrenar con dataset mejorado")
        print("\nRecuerda actualizar la ruta del CSV en my_model.py:")
        print(f"  CSV_PATH = '{output_path.name}'")
        
        return output_path


def main():
    # Ruta al CSV de entrenamiento
    csv_path = Path(__file__).parent.parent / 'src' / 'data' / 'mov_data_proccesed.csv'
    
    if not csv_path.exists():
        csv_path = Path(__file__).parent.parent / 'src' / 'data' / 'movement_data.csv'
    
    if not csv_path.exists():
        print(f"❌ Error: No se encontró el CSV de datos")
        print(f"   Buscado en: {csv_path.absolute()}")
        return
    
    engineer = TemporalFeatureEngineer(csv_path)
    engineer.run_improvement()


if __name__ == "__main__":
    main()
