# Reconocimiento de Objetos en TEMI

App Android para el robot TEMI que permite **crear un dataset de segmentación de objetos**: se fotografía un objeto, se marca con unos toques, un servidor con **SAM 2** genera su máscara y la app guarda la imagen y su etiqueta en **formato YOLO de segmentación**, lista para entrenar un modelo de reconocimiento de objetos.

## Cómo funciona

1. **Captura:** la cámara muestra la imagen en directo y se toma la foto pulsando **Capture**.
2. **Puntos:** el usuario marca el objeto con puntos verdes (**Add point**) y, si hace falta, excluye zonas con puntos rojos (**Remove point**). **Clear points** borra todos los puntos.
3. **Máscara:** **Generate Mask** envía la foto y los puntos al servidor, que devuelve la imagen con la máscara dibujada.
4. **Revisión:** el usuario elige la clase del objeto y pulsa **Confirm mask**, o repite con **Redo mask**.
5. **Guardado:** la app recibe los polígonos del contorno, guarda la imagen y la etiqueta, y vuelve a la cámara.

## Arquitectura

```text
┌──────────────────────────┐   HTTP (misma red)   ┌──────────────────────────────┐
│ App Android (TEMI)       │ ───────────────────▶ │ Servidor FastAPI + SAM 2.1   │
│ Kotlin · Jetpack Compose │ ◀─────────────────── │ Hiera Tiny (ordenador)       │
│ CameraX · OkHttp         │   /segment /confirm  │ Pixel4Design/SAM2Server      │
└──────────────────────────┘                      └──────────────────────────────┘
```

- **App:** Kotlin, Jetpack Compose (Material 3), CameraX 1.3.4, OkHttp 4.12.0 y SDK de TEMI 1.138.0. `minSdk` 24.
- **Servidor:** FastAPI + Uvicorn con el modelo SAM 2.1 Hiera Tiny. Repositorio: <https://github.com/Pixel4Design/SAM2Server>

## Estructura del proyecto

```text
app/src/main/java/com/example/objectrecognitionfortemi/
├── MainActivity.kt                  # Punto de entrada; lanza SegmentationFlow
└── segmentation/
    ├── SegmentationFlow.kt          # Orquesta: cámara → puntos → máscara → guardado
    ├── LiveCameraScreen.kt          # Cámara en directo y captura con botón
    ├── PointsEditorScreen.kt        # Marcado de puntos (añadir, quitar, borrar)
    ├── MaskReviewScreen.kt          # Revisión de la máscara y elección de clase
    ├── SegmentationApi.kt           # Cliente HTTP del servidor (/segment, /confirm)
    ├── SegmentationStorage.kt       # Guardado de imagen y etiquetas en formato YOLO
    └── Labels.kt                    # Lista de clases
```

## Clases

El orden define el índice de clase guardado en las etiquetas:

| Índice | Clase |
|---|---|
| 0 | Glass Bottle |
| 1 | Plastic Bottle |
| 2 | Can |
| 3 | Cardboard/Box |
| 4 | Plastic Container |
| 5 | Plastic Cup |
| 6 | Paper |

Para añadir o cambiar clases, edita `Labels.kt`. Si cambias el orden, los índices de las etiquetas ya guardadas dejarán de corresponder.

## Puesta en marcha

### 1. Servidor SAM 2

Requisitos: Python 3.10 o superior (recomendado 3.11 o 3.12), Git y pip.

```bash
git clone https://github.com/Pixel4Design/SAM2Server.git
cd SAM2Server

python3 -m venv sam2_env
source sam2_env/bin/activate          # Windows: sam2_env\Scripts\activate

pip install torch torchvision
pip install -e .
pip install fastapi uvicorn python-multipart opencv-python numpy

curl -L -o checkpoints/sam2.1_hiera_tiny.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt

uvicorn server:app --host 0.0.0.0 --port 8000
```

El servidor queda en `http://127.0.0.1:8000` (documentación interactiva en `/docs`). Para detenerlo, `Ctrl + C`.
La configuración del modelo ya viene en el repositorio: `configs/sam2.1/sam2.1_hiera_t.yaml`.

### 2. Dirección del servidor en la app

En `segmentation/SegmentationApi.kt`, cambia la IP por la del ordenador que ejecuta el servidor:

```kotlin
private const val SERVER_URL = "http://192.168.1.25:8000"   // ejemplo: usa tu IP
```

Para saber la IP del ordenador:

- **Mac:** `ipconfig getifaddr en0`
- **Linux:** `hostname -I`
- **Windows:** `ipconfig` (campo «Dirección IPv4»)

El TEMI y el ordenador deben estar en la misma red. En el emulador de Android Studio se usa `http://10.0.2.2:8000`.

### 3. Compilar e instalar

Abre el proyecto en Android Studio, haz *Sync* y ejecútalo en el TEMI o en una tablet Android. La primera vez, la app pide permiso de cámara.

## Resultado generado

La app guarda los datos en el almacenamiento propio del dispositivo:

```text
/sdcard/Android/data/com.example.objectrecognitionfortemi/files/imagenesSegmentadas/
├── images/   <marca-de-tiempo>_<Clase>.jpg
└── labels/   <marca-de-tiempo>_<Clase>.txt
```

- Las imágenes son JPG con un lado largo máximo de 1280 px.
- Cada línea del `.txt` es un polígono: `clase x1 y1 x2 y2 ... xn yn`, con coordenadas normalizadas entre 0 y 1 (formato YOLO de segmentación).

Para copiarlo al ordenador:

```bash
adb pull /sdcard/Android/data/com.example.objectrecognitionfortemi/files/imagenesSegmentadas ./imagenesSegmentadas
```

Si el TEMI se conecta por Wi-Fi, activa ADB en el robot y usa antes `adb connect IP_DEL_TEMI:5555`. Al desinstalar la app, Android borra esta carpeta: copia los datos antes.

## API del servidor

Contrato que usa la app (según `SegmentationApi.kt`):

| Endpoint | Entrada | Respuesta |
|---|---|---|
| `POST /segment` | Formulario multipart: `file` (JPEG) y `points` (JSON: `[{"x":300,"y":250,"label":1}]`, con `1` = añadir y `0` = quitar) | Imagen con la máscara dibujada |
| `POST /confirm` | Formulario: `label` (clase) | JSON con `width`, `height` y `boxes[].points` (polígonos en píxeles) de la última máscara generada |

## Notas

- La app usa HTTP sin cifrar (`usesCleartextTraffic`), pensado para una red local de confianza.
- El servidor conserva la última máscara generada, así que conviene que solo un dispositivo lo use a la vez.
- Permisos de la app: cámara e internet.

## Autoría y licencia

*Pendiente de completar.*
