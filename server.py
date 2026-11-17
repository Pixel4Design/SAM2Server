
import torch
import numpy as np
import cv2
import json
import threading

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ==================================================
# 1. CONFIGURATION
# ==================================================

CHECKPOINT = "checkpoints/sam2.1_hiera_tiny.pt"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"

app = FastAPI(
    title="SAM 2 Interactive Segmentation API",
    description="Interactive segmentation and annotation API",
    version="2.1.0"
)


# ==================================================
# 2. GLOBAL VARIABLES
# ==================================================

current_image = None
current_mask = None
current_score = None

# Protect SAM 2 and shared segmentation state
inference_lock = threading.Lock()


# ==================================================
# 3. DEVICE SELECTION
# ==================================================

if torch.cuda.is_available():

    device = "cuda"

elif (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
):

    device = "mps"

else:

    device = "cpu"


print(f"Selected device: {device}")


# ==================================================
# 4. LOAD SAM 2
# ==================================================

print("Loading SAM 2 model...")

model = build_sam2(
    MODEL_CONFIG,
    CHECKPOINT,
    device=device
)

predictor = SAM2ImagePredictor(model)

print("SAM 2 model loaded successfully")


# ==================================================
# 5. STATUS ENDPOINT
# ==================================================

@app.get("/")
def home():

    return {
        "status": "online",
        "model": "SAM 2",
        "device": device,
        "segmentation_available": current_mask is not None
    }


# ==================================================
# 6. SEGMENT ENDPOINT
# ==================================================

@app.post("/segment")
def segment_image(

    file: UploadFile = File(...),

    points: str = Form(...)

):

    global current_image
    global current_mask
    global current_score

    # ----------------------------------------------
    # 6.1 READ IMAGE
    # ----------------------------------------------

    image_bytes = file.file.read()

    if not image_bytes:

        raise HTTPException(
            status_code=400,
            detail="No image received"
        )

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise HTTPException(
            status_code=400,
            detail="Invalid image"
        )

    # Convert BGR to RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    height, width = image.shape[:2]


    # ----------------------------------------------
    # 6.2 PROCESS POINTS
    # ----------------------------------------------

    try:

        points_data = json.loads(points)

        if not isinstance(points_data, list) or not points_data:

            raise ValueError("Empty points list")

        input_points = []
        input_labels = []

        for point in points_data:

            x = int(point["x"])
            y = int(point["y"])
            point_label = int(point["label"])

            if not (0 <= x < width and 0 <= y < height):

                raise ValueError("Point outside image boundaries")

            if point_label not in (0, 1):

                raise ValueError("Invalid point label")

            input_points.append([x, y])
            input_labels.append(point_label)

        input_points = np.array(
            input_points,
            dtype=np.float32
        )

        input_labels = np.array(
            input_labels,
            dtype=np.int32
        )

    except (
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError
    ) as e:

        raise HTTPException(
            status_code=400,
            detail=f"Invalid segmentation data: {str(e)}"
        )


    # ----------------------------------------------
    # 6.3 RUN SAM 2
    # ----------------------------------------------

    try:

        with inference_lock:

            with torch.inference_mode():

                predictor.set_image(image)

                masks, scores, logits = predictor.predict(

                    point_coords=input_points,

                    point_labels=input_labels,

                    multimask_output=True

                )

            best_index = np.argmax(scores)

            best_mask = masks[best_index].copy()

            best_score = float(scores[best_index])

            # Save current segmentation state
            current_image = image.copy()

            current_mask = best_mask.copy()

            current_score = best_score


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Segmentation error: {str(e)}"
        )


    # ----------------------------------------------
    # 6.4 PREPARE MASK
    # ----------------------------------------------

    mask_bool = best_mask.astype(bool)


    # ----------------------------------------------
    # 6.5 GENERATE OVERLAY
    # ----------------------------------------------

    overlay = image.copy()

    mask_color = np.array(
        [0, 255, 0],
        dtype=np.uint8
    )

    alpha = 0.45

    overlay[mask_bool] = (

        (1 - alpha) * image[mask_bool] +

        alpha * mask_color

    ).astype(np.uint8)


    # ----------------------------------------------
    # 6.6 DRAW CONTOURS
    # ----------------------------------------------

    mask_uint8 = mask_bool.astype(np.uint8) * 255

    contours, _ = cv2.findContours(

        mask_uint8,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE

    )

    cv2.drawContours(

        overlay,

        contours,

        -1,

        (255, 255, 0),

        3

    )


    # ----------------------------------------------
    # 6.7 ENCODE OVERLAY
    # ----------------------------------------------

    overlay_bgr = cv2.cvtColor(

        overlay,

        cv2.COLOR_RGB2BGR

    )

    success, encoded_overlay = cv2.imencode(

        ".jpg",

        overlay_bgr,

        [cv2.IMWRITE_JPEG_QUALITY, 95]

    )

    if not success:

        raise HTTPException(
            status_code=500,
            detail="Error generating overlay image"
        )


    # ----------------------------------------------
    # 6.8 RETURN IMAGE
    # ----------------------------------------------

    return Response(

        content=encoded_overlay.tobytes(),

        media_type="image/jpeg"

    )


# ==================================================
# 7. CONFIRM ENDPOINT
# ==================================================

@app.post("/confirm")
def confirm_segmentation(

    label: str = Form(...)

):

    global current_image
    global current_mask
    global current_score

    # ----------------------------------------------
    # 7.1 CHECK CURRENT SEGMENTATION
    # ----------------------------------------------

    with inference_lock:

        if current_image is None or current_mask is None:

            raise HTTPException(
                status_code=400,
                detail="No segmentation available. Call /segment first."
            )

        image_height, image_width = current_image.shape[:2]

        mask_bool = current_mask.astype(bool)

        score = current_score

        # ------------------------------------------
        # 7.2 EXTRACT CONTOURS
        # ------------------------------------------

        mask_uint8 = mask_bool.astype(np.uint8) * 255

        contours, _ = cv2.findContours(

            mask_uint8,

            cv2.RETR_EXTERNAL,

            cv2.CHAIN_APPROX_SIMPLE

        )

        boxes = []

        for contour in contours:

            if cv2.contourArea(contour) < 5:

                continue

            x, y, w, h = cv2.boundingRect(contour)

            center_x = x + w / 2

            center_y = y + h / 2

            perimeter = cv2.arcLength(
                contour,
                True
            )

            epsilon = 0.002 * perimeter

            polygon = cv2.approxPolyDP(

                contour,

                epsilon,

                True

            )

            polygon_points = [

                [
                    int(point[0][0]),
                    int(point[0][1])
                ]

                for point in polygon

            ]

            if len(polygon_points) < 3:

                continue

            boxes.append({

                "type": "polygon",

                "label": label.strip(),

                "x": center_x,

                "y": center_y,

                "width": w,

                "height": h,

                "points": polygon_points

            })


        # ------------------------------------------
        # 7.3 CHECK ANNOTATIONS
        # ------------------------------------------

        if not boxes:

            raise HTTPException(
                status_code=400,
                detail="No valid annotations found"
            )


        # ------------------------------------------
        # 7.4 GENERATE ANNOTATION JSON
        # ------------------------------------------

        annotation = {

            "boxes": boxes,

            "height": image_height,

            "width": image_width,

            "sam2_score": score

        }

        # ------------------------------------------
        # 7.5 CLEAR TEMPORARY STATE
        # ------------------------------------------

        current_image = None
        current_mask = None
        current_score = None


    # ----------------------------------------------
    # 7.6 RETURN ANNOTATIONS
    # ----------------------------------------------

    return annotation


# ==================================================
# 8. RUN SERVER
# ==================================================

# Execute:
# uvicorn server:app --host 0.0.0.0 --port 8000
