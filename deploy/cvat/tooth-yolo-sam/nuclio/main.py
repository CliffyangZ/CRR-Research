# SPDX-License-Identifier: MIT
"""CVAT Nuclio entry point for automatic tooth instance segmentation."""

import base64
import io
import json

from PIL import Image

from model_handler import ModelHandler


def init_context(context):
    context.logger.info("Loading Tooth YOLO + SAM segmentation models")
    context.user_data.model = ModelHandler()
    context.logger.info("Tooth YOLO + SAM segmentation models are ready")


def handler(context, event):
    try:
        data = event.body
        image_bytes = base64.b64decode(data["image"], validate=True)
        image = Image.open(io.BytesIO(image_bytes))
        threshold = float(data.get("threshold", 0.25))
        shapes = context.user_data.model.infer(image, threshold)
        return context.Response(
            body=json.dumps(shapes),
            headers={},
            content_type="application/json",
            status_code=200,
        )
    except (KeyError, TypeError, ValueError) as exc:
        context.logger.warn_with("Invalid segmentation request", error=str(exc))
        return context.Response(
            body=json.dumps({"error": str(exc)}),
            headers={},
            content_type="application/json",
            status_code=400,
        )
    except Exception as exc:
        context.logger.error_with("Tooth YOLO + SAM segmentation failed", error=str(exc))
        return context.Response(
            body=json.dumps({"error": "Tooth YOLO + SAM segmentation failed"}),
            headers={},
            content_type="application/json",
            status_code=500,
        )
