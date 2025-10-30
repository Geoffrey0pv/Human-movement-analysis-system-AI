# Recorre todos los archivos mp4 del directorio actual
$archivos = Get-ChildItem -Filter "videos/*.mp4"

# Define los grupos principales con patrones que los identifican
$grupos = @{
    "caminar_adelante" = "adelante"
    "caminar_atras"    = "atras"
    "pararse"          = "para|levan"
    "sentarse"         = "sentar"
    "girar"            = "gir"
}

# Diccionario para llevar la cuenta del número secuencial
$contador = @{
    "caminar_adelante" = 1
    "caminar_atras"    = 1
    "pararse"          = 1
    "sentarse"         = 1
    "girar"            = 1
}

foreach ($archivo in $archivos) {

    $nombre = $archivo.BaseName
    $nuevoNombre = $null

    foreach ($accion in $grupos.Keys) {
        if ($nombre -match $grupos[$accion]) {
            # Determinar velocidad
            if ($nombre -match "lento") {
                $velocidad = "lento"
            } elseif ($nombre -match "rapido") {
                $velocidad = "rapido"
            } else {
                $velocidad = "rapido"  # Por defecto
            }

            # Número con formato 2 dígitos
            $num = "{0:D2}" -f $contador[$accion]
            $contador[$accion]++

            # Construir nombre nuevo
            $nuevoNombre = "${accion}_${num}_${velocidad}.mp4"
            break
        }
    }

    if ($nuevoNombre) {
        Write-Host "Renombrando '$($archivo.Name)' → '$nuevoNombre'"
        Rename-Item -Path $archivo.FullName -NewName $nuevoNombre
    } else {
        Write-Host "⚠ No se detectó grupo para '$($archivo.Name)', se omite."
    }
}
