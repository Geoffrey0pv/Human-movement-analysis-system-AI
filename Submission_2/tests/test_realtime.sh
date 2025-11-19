#!/bin/bash
# Script de prueba para el clasificador mejorado
# EJECUTAR DESDE: Submission_2/tests/

echo "=========================================="
echo "TEST CLASIFICADOR DE MOVIMIENTOS V3"
echo "=========================================="
echo ""

# Activar entorno virtual (desde tests/)
if [ -d "../../venv" ]; then
    echo "✓ Activando entorno virtual..."
    source ../../venv/bin/activate
fi

# Rutas relativas desde tests/
MODEL_PATH="../src/models/modelo_acciones.pkl"
VIDEO_DIR="../../Submission 1/src/data/videos"

echo ""
echo "OPCIONES DE USO:"
echo ""
echo "1. WEBCAM (tiempo real):"
echo "   python realtime_classifier_v3.py"
echo ""
echo "2. VIDEO (archivo):"
echo "   python realtime_classifier_v3.py --video $VIDEO_DIR/caminar_adelante_01_lento.mp4"
echo ""
echo "3. DEBUG MODE:"
echo "   python realtime_classifier_v3.py --debug"
echo ""
echo "=========================================="
echo ""

# Verificar modelo
if [ -f "$MODEL_PATH" ]; then
    echo "✓ Modelo encontrado: $MODEL_PATH"
else
    echo "❌ Modelo NO encontrado en: $MODEL_PATH"
    echo "   Entrena el modelo primero: cd ../src/models && python my_model.py"
    exit 1
fi

# Verificar videos
echo ""
echo "Videos disponibles:"
if [ -d "$VIDEO_DIR" ]; then
    ls -1 "$VIDEO_DIR"/*.mp4 2>/dev/null | while read video; do
        echo "  - $(basename "$video")"
    done
else
    echo "  (directorio de videos no encontrado en: $VIDEO_DIR)"
fi

echo ""
echo "=========================================="
echo "Selecciona modo:"
echo "  1) Webcam"
echo "  2) Video - caminar_adelante_01_lento"
echo "  3) Video - girar_rapido_02"
echo "  4) Video - sentarse_lento_01"
echo "  5) Salir"
echo ""
read -p "Opción: " opcion

case $opcion in
    1)
        echo ""
        echo "Iniciando webcam..."
        python realtime_classifier_v3.py --debug
        ;;
    2)
        echo ""
        echo "Clasificando: caminar_adelante_01_lento.mp4"
        python realtime_classifier_v3.py --video "$VIDEO_DIR/caminar_adelante_01_lento.mp4" --debug
        ;;
    3)
        echo ""
        echo "Clasificando: girar_rapido_02.mp4"
        python realtime_classifier_v3.py --video "$VIDEO_DIR/girar_rapido_02.mp4" --debug
        ;;
    4)
        echo ""
        echo "Clasificando: sentarse_lento_01.mp4"
        python realtime_classifier_v3.py --video "$VIDEO_DIR/sentarse_lento_01.mp4" --debug
        ;;
    5)
        echo "Saliendo..."
        exit 0
        ;;
    *)
        echo "Opción no válida"
        exit 1
        ;;
esac
