from pathlib import Path
from collections import deque

import cv2
import numpy as np
import torch
import mediapipe as mp
from PIL import Image

from src.models.gaze_spatial import GazeSpatialModel


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

VIDEO_PATH = ROOT / "data" / "demo" / "gaze_video.mp4"

CHECKPOINT_PATH = (
    ROOT / "outputs" / "E1B_seed456" / "best_model.pth"
)

OUTPUT_PATH = ROOT / "outputs" / "gaze_demo_final.mp4"


# ============================================================
# MODEL SETTINGS
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 224
HEAD_SIZE = 112

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


# ============================================================
# VIDEO / SMOOTHING SETTINGS
# ============================================================

# Number of previous gaze points shown in trajectory
TRAJECTORY_LENGTH = 60

# EMA smoothing:
# closer to 1.0 = smoother but slower
# closer to 0.0 = more responsive but jittery
SMOOTHING_ALPHA = 0.25


# ============================================================
# IMAGE HELPERS
# ============================================================

def normalize_image(img):
    """
    RGB uint8 image -> normalized CHW float32
    """

    img = img.astype(np.float32) / 255.0

    img = (
        img - IMAGENET_MEAN
    ) / IMAGENET_STD

    img = np.transpose(
        img,
        (2, 0, 1)
    )

    return img.astype(np.float32)


def clamp_box(
    xmin,
    ymin,
    xmax,
    ymax
):
    xmin = max(
        0.0,
        min(1.0, xmin)
    )

    ymin = max(
        0.0,
        min(1.0, ymin)
    )

    xmax = max(
        0.0,
        min(1.0, xmax)
    )

    ymax = max(
        0.0,
        min(1.0, ymax)
    )

    return (
        xmin,
        ymin,
        xmax,
        ymax
    )


# ============================================================
# FACE -> HEAD BOX
# ============================================================

def face_to_head_box(face_box):
    """
    MediaPipe provides a face bounding box.

    E1B was trained using GazeFollow head boxes,
    so we create a tighter approximate head box.

    This is a heuristic, not a learned head detector.
    """

    xmin, ymin, xmax, ymax = face_box

    w = xmax - xmin
    h = ymax - ymin

    # Small horizontal expansion
    expand_x = 0.08 * w

    # Add some space above the face
    expand_top = 0.25 * h

    # Very small bottom expansion
    expand_bottom = 0.02 * h

    xmin -= expand_x
    xmax += expand_x

    ymin -= expand_top
    ymax += expand_bottom

    return clamp_box(
        xmin,
        ymin,
        xmax,
        ymax
    )


# ============================================================
# HEAD MASK
# ============================================================

def create_head_mask(box):
    """
    Create a 224x224 binary head mask.
    """

    xmin, ymin, xmax, ymax = box

    mask = np.zeros(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        dtype=np.float32
    )

    x1 = int(
        xmin * IMAGE_SIZE
    )

    y1 = int(
        ymin * IMAGE_SIZE
    )

    x2 = int(
        xmax * IMAGE_SIZE
    )

    y2 = int(
        ymax * IMAGE_SIZE
    )

    x1 = max(
        0,
        min(
            IMAGE_SIZE - 1,
            x1
        )
    )

    y1 = max(
        0,
        min(
            IMAGE_SIZE - 1,
            y1
        )
    )

    x2 = max(
        x1 + 1,
        min(
            IMAGE_SIZE,
            x2
        )
    )

    y2 = max(
        y1 + 1,
        min(
            IMAGE_SIZE,
            y2
        )
    )

    mask[
        y1:y2,
        x1:x2
    ] = 1.0

    return mask


# ============================================================
# HEAD CROP
# ============================================================

def crop_head(
    image,
    box,
    expand=0.20
):
    """
    Crop head using the same basic 20% expansion
    used by the E1B training pipeline.
    """

    h, w = image.shape[:2]

    xmin, ymin, xmax, ymax = box

    bw = xmax - xmin
    bh = ymax - ymin

    xmin -= expand * bw
    xmax += expand * bw

    ymin -= expand * bh
    ymax += expand * bh

    xmin, ymin, xmax, ymax = clamp_box(
        xmin,
        ymin,
        xmax,
        ymax
    )

    x1 = max(
        0,
        int(xmin * w)
    )

    y1 = max(
        0,
        int(ymin * h)
    )

    x2 = min(
        w,
        int(xmax * w)
    )

    y2 = min(
        h,
        int(ymax * h)
    )

    if x2 <= x1 or y2 <= y1:
        return None

    return image[
        y1:y2,
        x1:x2
    ]


# ============================================================
# MODEL
# ============================================================

def load_model():

    print(
        f"Device: {DEVICE}"
    )

    print(
        "Loading checkpoint:"
    )

    print(
        CHECKPOINT_PATH
    )

    model = GazeSpatialModel(
        variant="two_pathway",
        pretrained=False,
        heatmap_size=56
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        print(
            "Checkpoint epoch:",
            checkpoint.get(
                "epoch",
                "unknown"
            )
        )

        print(
            "Validation distance:",
            checkpoint.get(
                "val_dist",
                "unknown"
            )
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.to(DEVICE)

    model.eval()

    print(
        "E1B model loaded successfully."
    )

    return model


# ============================================================
# EMA SMOOTHING
# ============================================================

def smooth_gaze(
    previous,
    current
):
    """
    Exponential moving average.
    """

    if previous is None:
        return current

    px, py = previous
    cx, cy = current

    sx = (
        SMOOTHING_ALPHA * cx
        + (1.0 - SMOOTHING_ALPHA) * px
    )

    sy = (
        SMOOTHING_ALPHA * cy
        + (1.0 - SMOOTHING_ALPHA) * py
    )

    return (
        sx,
        sy
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not VIDEO_PATH.exists():

        raise FileNotFoundError(
            f"Video not found:\n{VIDEO_PATH}"
        )

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load E1B
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # MediaPipe Face Detection
    # --------------------------------------------------------

    mp_face_detection = (
        mp.solutions.face_detection
    )

    face_detector = (
        mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )
    )

    # --------------------------------------------------------
    # Open input video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video:\n{VIDEO_PATH}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 25.0

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print()
    print(
        f"Video: {VIDEO_PATH}"
    )

    print(
        f"Resolution: "
        f"{width} x {height}"
    )

    print(
        f"FPS: {fps:.2f}"
    )

    print(
        f"Frames: {total_frames}"
    )

    print()

    # --------------------------------------------------------
    # Video writer
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        fourcc,
        fps,
        (
            width,
            height
        )
    )

    if not writer.isOpened():

        raise RuntimeError(
            "Could not create output video."
        )

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    trajectory = deque(
        maxlen=TRAJECTORY_LENGTH
    )

    last_head_box = None

    smoothed_gaze = None

    frame_number = 0

    # --------------------------------------------------------
    # Process video
    # --------------------------------------------------------

    with torch.no_grad():

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            # =================================================
            # BGR -> RGB
            # =================================================

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # =================================================
            # FACE DETECTION
            # =================================================

            results = (
                face_detector.process(
                    rgb
                )
            )

            face_box = None

            if results.detections:

                detection = max(
                    results.detections,
                    key=lambda d:
                    d.score[0]
                )

                bbox = (
                    detection
                    .location_data
                    .relative_bounding_box
                )

                fx = bbox.xmin
                fy = bbox.ymin

                fw = bbox.width
                fh = bbox.height

                face_box = clamp_box(
                    fx,
                    fy,
                    fx + fw,
                    fy + fh
                )

            # =================================================
            # HEAD BOX
            # =================================================

            if face_box is not None:

                head_box = (
                    face_to_head_box(
                        face_box
                    )
                )

                last_head_box = head_box

            else:

                head_box = last_head_box

            predicted_gaze = None

            # =================================================
            # E1B INFERENCE
            # =================================================

            if head_box is not None:

                # ------------------------------------------------
                # Scene
                # ------------------------------------------------

                scene_img = (
                    Image.fromarray(
                        rgb
                    ).resize(
                        (
                            IMAGE_SIZE,
                            IMAGE_SIZE
                        ),
                        Image.Resampling.BILINEAR
                    )
                )

                scene = np.array(
                    scene_img,
                    copy=True
                )

                scene = normalize_image(
                    scene
                )

                # ------------------------------------------------
                # Head crop
                # ------------------------------------------------

                crop = crop_head(
                    rgb,
                    head_box,
                    expand=0.20
                )

                if crop is not None:

                    crop_img = (
                        Image.fromarray(
                            crop
                        ).resize(
                            (
                                HEAD_SIZE,
                                HEAD_SIZE
                            ),
                            Image.Resampling.BILINEAR
                        )
                    )

                    crop = np.array(
                        crop_img,
                        copy=True
                    )

                    crop = normalize_image(
                        crop
                    )

                    # ------------------------------------------------
                    # Head mask
                    # ------------------------------------------------

                    mask = create_head_mask(
                        head_box
                    )

                    # ------------------------------------------------
                    # Tensors
                    # ------------------------------------------------

                    scene_tensor = (
                        torch.from_numpy(
                            scene
                        )
                        .unsqueeze(0)
                        .to(DEVICE)
                    )

                    crop_tensor = (
                        torch.from_numpy(
                            crop
                        )
                        .unsqueeze(0)
                        .to(DEVICE)
                    )

                    mask_tensor = (
                        torch.from_numpy(
                            mask
                        )
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(DEVICE)
                    )

                    # ------------------------------------------------
                    # E1B
                    # ------------------------------------------------

                    heatmap = model(
                        scene_tensor,
                        mask_tensor,
                        crop_tensor
                    )

                    # ------------------------------------------------
                    # Heatmap -> normalized gaze
                    # ------------------------------------------------

                    gaze = model.soft_argmax(
                        heatmap
                    )

                    gaze = (
                        gaze
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(-1)
                    )

                    gx = float(
                        np.clip(
                            gaze[0],
                            0.0,
                            1.0
                        )
                    )

                    gy = float(
                        np.clip(
                            gaze[1],
                            0.0,
                            1.0
                        )
                    )

                    raw_gaze = (
                        gx,
                        gy
                    )

                    # ------------------------------------------------
                    # Temporal smoothing
                    # ------------------------------------------------

                    smoothed_gaze = (
                        smooth_gaze(
                            smoothed_gaze,
                            raw_gaze
                        )
                    )

                    predicted_gaze = (
                        smoothed_gaze
                    )

                    # ------------------------------------------------
                    # Add to trajectory
                    # ------------------------------------------------

                    px = int(
                        predicted_gaze[0]
                        * width
                    )

                    py = int(
                        predicted_gaze[1]
                        * height
                    )

                    trajectory.append(
                        (px, py)
                    )

            # =================================================
            # DRAW OUTPUT
            # =================================================

            output_frame = (
                frame.copy()
            )

            # ------------------------------------------------
            # Face box - GREEN
            # ------------------------------------------------

            if face_box is not None:

                fx1 = int(
                    face_box[0]
                    * width
                )

                fy1 = int(
                    face_box[1]
                    * height
                )

                fx2 = int(
                    face_box[2]
                    * width
                )

                fy2 = int(
                    face_box[3]
                    * height
                )

                cv2.rectangle(
                    output_frame,
                    (fx1, fy1),
                    (fx2, fy2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    output_frame,
                    "Face",
                    (
                        fx1,
                        max(
                            25,
                            fy1 - 8
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            # ------------------------------------------------
            # Head box - BLUE
            # ------------------------------------------------

            if head_box is not None:

                hx1 = int(
                    head_box[0]
                    * width
                )

                hy1 = int(
                    head_box[1]
                    * height
                )

                hx2 = int(
                    head_box[2]
                    * width
                )

                hy2 = int(
                    head_box[3]
                    * height
                )

                cv2.rectangle(
                    output_frame,
                    (hx1, hy1),
                    (hx2, hy2),
                    (255, 180, 0),
                    1
                )

            # ------------------------------------------------
            # Gaze trajectory - RED
            # ------------------------------------------------

            if len(trajectory) >= 2:

                points = list(
                    trajectory
                )

                for i in range(
                    1,
                    len(points)
                ):

                    cv2.line(
                        output_frame,
                        points[i - 1],
                        points[i],
                        (0, 0, 255),
                        2
                    )

            # ------------------------------------------------
            # Current gaze point
            # ------------------------------------------------

            if predicted_gaze is not None:

                gx, gy = (
                    predicted_gaze
                )

                px = int(
                    gx * width
                )

                py = int(
                    gy * height
                )

                # Outer white ring
                cv2.circle(
                    output_frame,
                    (px, py),
                    14,
                    (255, 255, 255),
                    2
                )

                # Red gaze point
                cv2.circle(
                    output_frame,
                    (px, py),
                    8,
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    output_frame,
                    f"Gaze: "
                    f"({gx:.2f}, {gy:.2f})",
                    (
                        20,
                        35
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            else:

                cv2.putText(
                    output_frame,
                    "Gaze: No face detected",
                    (
                        20,
                        35
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            # ------------------------------------------------
            # Project title
            # ------------------------------------------------

            cv2.putText(
                output_frame,
                "GazeSense - E1B Gaze Estimation",
                (
                    20,
                    height - 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            # ------------------------------------------------
            # Write frame
            # ------------------------------------------------

            writer.write(
                output_frame
            )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if frame_number % 30 == 0:

                print(
                    f"Processed "
                    f"{frame_number}/"
                    f"{total_frames} frames"
                )

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()
    writer.release()
    face_detector.close()

    print()
    print(
        "======================================"
    )

    print(
        "Video inference completed!"
    )

    print(
        "======================================"
    )

    print(
        f"Output:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()