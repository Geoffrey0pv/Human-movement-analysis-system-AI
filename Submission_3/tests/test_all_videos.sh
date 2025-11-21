#!/bin/bash

# Script para probar TODOS los videos automáticamente
# y generar un reporte completo

VIDEO_DIR="../../Submission 1/src/data/videos"
OUTPUT_FILE="validation_report.txt"

echo "=============================================" > $OUTPUT_FILE
echo "REPORTE DE VALIDACIÓN - CLASIFICADOR V4" >> $OUTPUT_FILE
echo "=============================================" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "Fecha: $(date)" >> $OUTPUT_FILE
echo "Videos analizados:" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# Lista de videos
VIDEOS=(
    "caminar_adelante_01_lento.mp4"
    "caminar_adelante_02_rapido.mp4"
    "caminar_atras_01_lento.mp4"
    "caminar_atras_02_rapido.mp4"
    "girar_rapido_02.mp4"
    "giro_lento_01.mp4"
    "pararse_lento_01.mp4"
    "sentarse_lento_01.mp4"
)

# Probar cada video
for video in "${VIDEOS[@]}"; do
    echo "=============================================" >> $OUTPUT_FILE
    echo "VIDEO: $video" >> $OUTPUT_FILE
    echo "=============================================" >> $OUTPUT_FILE
    
    # Ejecutar clasificador y capturar solo las estadísticas
    timeout 30 python realtime_classifier_v4.py "$VIDEO_DIR/$video" 2>/dev/null | \
        grep -A 20 "ESTADÍSTICAS" >> $OUTPUT_FILE
    
    echo "" >> $OUTPUT_FILE
done

echo "=============================================" >> $OUTPUT_FILE
echo "FIN DEL REPORTE" >> $OUTPUT_FILE
echo "=============================================" >> $OUTPUT_FILE

# Mostrar reporte
cat $OUTPUT_FILE

echo ""
echo "✓ Reporte guardado en: $OUTPUT_FILE"
