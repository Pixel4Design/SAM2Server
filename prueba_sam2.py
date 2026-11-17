
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ==================================================
# CONFIGURACIÓN
# ==================================================

CHECKPOINT = "checkpoints/sam2.1_hiera_tiny.pt"
CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
IMAGE_PATH = "images.jpeg"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo utilizado: {device}")


# ==================================================
# CARGAR IMAGEN Y MODELO
# ==================================================

image = np.array(
    Image.open(IMAGE_PATH).convert("RGB")
)

model = build_sam2(
    CONFIG,
    CHECKPOINT,
    device=device
)

predictor = SAM2ImagePredictor(model)

# Calculamos los embeddings de la imagen una sola vez
predictor.set_image(image)


# ==================================================
# VARIABLES GLOBALES
# ==================================================

points = []
labels = []

current_mask = None
current_bbox = None


# ==================================================
# CONFIGURAR VISUALIZACIÓN
# ==================================================

fig, ax = plt.subplots(figsize=(12, 8))

ax.imshow(image)
ax.set_title(
    "Clic izquierdo: incluir | "
    "Clic derecho: excluir | "
    "R: reiniciar | Q: salir"
)

ax.axis("off")


# ==================================================
# FUNCIÓN DE SEGMENTACIÓN
# ==================================================

def segment_image():

    global current_mask
    global current_bbox

    if len(points) == 0:
        return

    input_points = np.array(
        points,
        dtype=np.float32
    )

    input_labels = np.array(
        labels,
        dtype=np.int32
    )

    # Ejecutar inferencia
    with torch.inference_mode():

        if device == "cuda":
            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16
            ):
                masks, scores, logits = predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    multimask_output=True
                )
        else:
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True
            )

    # Seleccionar la máscara con mayor puntuación
    best = np.argmax(scores)

    current_mask = masks[best]

    # ==================================================
    # CALCULAR BOUNDING BOX
    # ==================================================

    ys, xs = np.where(current_mask > 0)

    if len(xs) > 0:

        x_min = int(xs.min())
        y_min = int(ys.min())

        x_max = int(xs.max())
        y_max = int(ys.max())

        current_bbox = (
            x_min,
            y_min,
            x_max,
            y_max
        )

        print(
            f"Bounding Box: "
            f"X: {x_min}-{x_max}, "
            f"Y: {y_min}-{y_max}"
        )

    else:
        current_bbox = None

    # ==================================================
    # ACTUALIZAR VISUALIZACIÓN
    # ==================================================

    ax.clear()

    ax.imshow(image)

    # Mostrar máscara
    ax.imshow(
        current_mask,
        alpha=0.5,
        cmap="Blues"
    )

    # Dibujar bounding box
    if current_bbox is not None:

        x_min, y_min, x_max, y_max = current_bbox

        rect = patches.Rectangle(
            (x_min, y_min),
            x_max - x_min + 1,
            y_max - y_min + 1,
            linewidth=2,
            edgecolor="lime",
            facecolor="none"
        )

        ax.add_patch(rect)

        ax.text(
            x_min,
            max(0, y_min - 10),
            "Bounding Box",
            color="lime",
            fontsize=11,
            weight="bold"
        )

    # Dibujar puntos positivos y negativos
    for point, label in zip(points, labels):

        color = "lime" if label == 1 else "red"

        ax.scatter(
            point[0],
            point[1],
            color=color,
            s=80,
            edgecolors="white"
        )

    ax.set_title(
        "Clic izquierdo: incluir | "
        "Clic derecho: excluir | "
        "R: reiniciar | Q: salir"
    )

    ax.axis("off")

    fig.canvas.draw_idle()


# ==================================================
# EVENTO DEL RATÓN
# ==================================================

def on_click(event):

    if event.inaxes != ax:
        return

    if event.xdata is None or event.ydata is None:
        return

    # Coordenadas X e Y
    x = event.xdata
    y = event.ydata

    # Clic izquierdo = positivo
    if event.button == 1:

        points.append([x, y])
        labels.append(1)

        print(f"Punto positivo: ({x:.1f}, {y:.1f})")

    # Clic derecho = negativo
    elif event.button == 3:

        points.append([x, y])
        labels.append(0)

        print(f"Punto negativo: ({x:.1f}, {y:.1f})")

    segment_image()


# ==================================================
# EVENTO DEL TECLADO
# ==================================================

def on_key(event):

    global current_mask
    global current_bbox

    # Reiniciar
    if event.key == "r":

        points.clear()
        labels.clear()

        current_mask = None
        current_bbox = None

        ax.clear()
        ax.imshow(image)

        ax.set_title(
            "Clic izquierdo: incluir | "
            "Clic derecho: excluir | "
            "R: reiniciar | Q: salir"
        )

        ax.axis("off")

        fig.canvas.draw_idle()

        print("Segmentación reiniciada.")

    # Salir
    elif event.key == "q":

        plt.close(fig)


# ==================================================
# CONECTAR EVENTOS
# ==================================================

fig.canvas.mpl_connect(
    "button_press_event",
    on_click
)

fig.canvas.mpl_connect(
    "key_press_event",
    on_key
)


# ==================================================
# INICIAR INTERFAZ
# ==================================================

plt.show()


# ==================================================
# GUARDAR RESULTADOS
# ==================================================

if current_mask is not None:

    # Guardar máscara
    mask_image = (
        current_mask.astype(np.uint8) * 255
    )

    Image.fromarray(mask_image).save(
        "mascara_sam2.png"
    )

    print("Máscara guardada: mascara_sam2.png")

    # Guardar recorte del bounding box
    if current_bbox is not None:

        x_min, y_min, x_max, y_max = current_bbox

        cropped_image = image[
            y_min:y_max + 1,
            x_min:x_max + 1
        ]

        Image.fromarray(cropped_image).save(
            "recorte_bbox.png"
        )

        print("Recorte guardado: recorte_bbox.png")

        print(
            f"Dimensiones del recorte: "
            f"{cropped_image.shape[1]}x"
            f"{cropped_image.shape[0]}"
        )

else:

    print("No hay ninguna segmentación que guardar.")