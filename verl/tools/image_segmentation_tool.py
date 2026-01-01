# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Flow for a single segmentation call

1. Tool code awaits the remote call: result = await self.execution_pool.execute.remote(self._execute_once, instance_id, threshold, extra)

2. Ray schedules execute() on the SegmentationExecutionWorker actor (subject to max_concurrency).

3. execute() acquires a global token from TokenBucketWorker (blocks if all tokens in use).

4. It runs the provided function (here _execute_once), which:

   - Serializes the in-memory PIL image to bytes + mime.
   - Builds payload (multipart or json_base64).
   - Calls requests.post() (respecting timeout).
   - Parses result (image/* or JSON) into a PIL image and text.

5. On return (or exception), the ExitStack triggers release() to free the token.

6. The Tool returns ToolResponse(image=[...], text=...), plus metrics (status_code, latency_ms, api_request_error, content_type).

Tuning guidelines

- num_workers (actor max_concurrency):

  - Increase to allow more concurrent in-process operations (network-bound calls benefit).
  - If you see CPU contention, you can keep this moderate (e.g., 10–50) as requests.post typically releases the GIL during network I/O.

- rate_limit (global concurrency cap):

  - Use to bound total in-flight requests across your cluster or across multiple tools sharing the same limiter.
  - Set lower than or equal to the sum of max_concurrency across all relevant actors to avoid overloading your service.

- Unique limiter names:
  - If multiple tools should not share capacity, assign distinct names in _init_rate_limit.

"""

"""
Default Contract

REQUEST (client → service)
- Method: POST
- Headers: includes "Content-Type: application/json" plus any config.request_headers (e.g., Authorization)
- JSON body: { "<image_field_name>": "<RAW_BASE64_IMAGE>", // raw base64 of the image bytes, NO data-URI prefix "threshold": 0.5, // optional; float in [0,1] for binarizing soft masks ...config.request_extra_fields, // static fields from tool config ...parameters.extra // per-call fields from tool.execute(parameters={"extra": {...}}) }

Notes:

- <RAW_BASE64_IMAGE> is just the base64-encoded bytes (e.g., "/9j/4AAQSkZJRgABAQ..."), without "data:image/png;base64,".
- The client encodes the original PIL image to a standard format (PNG/JPEG/WEBP/BMP) before base64. Your service should base64-decode and load the image; most image libraries auto-detect format from the byte signature.
- The optional "threshold" is provided only if the user passes it in the tool call.

RESPONSE (service → client) — either of the following is accepted:

A) Direct image body Content-Type: image/* (e.g., image/png) Body: bytes of either the final colored overlay (if the service already composites) or a grayscale/binary mask.

- If the tool is configured with response_content_is_overlay=true, the body is treated as the final overlay.
- Otherwise, the body is treated as a mask; the tool will optionally binarize (using "threshold") and composite a colored overlay.

B) JSON body (Content-Type: application/json) One of the following fields (first match wins): { "overlay": "<RAW_BASE64_IMAGE>", // final overlay image (raw base64 string, no data-URI), or "overlay_url": "[](https://host/overlay.png)<https://host/overlay.png>",

```javascript
 "mask": "<RAW_BASE64_IMAGE>",      // mask image (raw base64 string, no data-URI), or
 "mask_png": "<RAW_BASE64_IMAGE>",  // alias accepted
 "mask_url": "https://host/mask.png",

 "overlays": [ "<RAW_BASE64_IMAGE>" | {"overlay": "...", "overlay_url": "..."} , ... ],
 "masks":    [ "<RAW_BASE64_IMAGE>" | {"mask": "...",    "mask_url": "..."} , ... ]
```

}

Tool behavior on JSON responses:

- If an overlay is provided, it is returned as the final image.

- If a mask is provided:

  - If return_image="mask": the grayscale mask is returned.
  - If return_image="overlay": the tool composites a colored overlay on the original (overlay_color, overlay_alpha).

Metrics returned to the agent include: status_code, latency_ms, api_request_error, content_type.
"""

import base64
import io
import json
import logging
import os
import threading
import time
from contextlib import ExitStack
from enum import Enum
from typing import Any, Optional, TypeVar
from uuid import uuid4

import ray
import ray.actor
import requests
from PIL import Image
from qwen_vl_utils import fetch_image

from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")


class PoolMode(Enum):
    """Execution pool mode enumeration."""

    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    """Ray actor for rate limiting using token bucket algorithm."""

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0  # For observability
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        """Acquire a token from the bucket."""
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        """Release a token back to the bucket."""
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        """Get current number of acquired tokens."""
        return self.current_count


class SegmentationExecutionWorker:
    """Worker for executing segmentation requests with optional rate limiting."""

    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        """Initialize singleton rate limiter."""
        return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        """Health check method."""
        return True

    def execute(self, fn, *fn_args, **fn_kwargs):
        """Execute function with optional rate limiting."""
        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                try:
                    return fn(*fn_args, **fn_kwargs)
                except Exception as e:
                    logger.warning(f"Error when executing segmentation: {e}")
                    raise
        else:
            return fn(*fn_args, **fn_kwargs)


def init_segmentation_execution_pool(
    num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode
):
    """Initialize segmentation execution pool."""
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(SegmentationExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
        )
    else:
        raise NotImplementedError("Process mode is not implemented yet")


class ImageSegmentationTool(BaseTool):
    """Segment an image using a remote HTTP service and return a masked/overlay image.

    Configuration (config dict):
      - segmentation_service_url (str, required): The HTTP endpoint to call.
      - payload_format (str, optional): "multipart" (default) | "json_base64".
      - image_field_name (str, optional): Form/JSON field name for the image. Default "image".
      - request_headers (dict, optional): Extra headers for the request.
      - request_extra_fields (dict, optional): Extra fields for the request (both multipart and json).
      - json_base64_data_uri (bool, optional): If True and payload_format="json_base64", encode as data URI
        e.g. "data:image/png;base64,...". Default False (raw base64 string).
      - response_content_is_overlay (bool, optional): If the service returns an image body that is already the
        overlay (not a mask). Default False (assume image body is a mask).
      - overlay_color (tuple/list, optional): RGB color used for overlaying mask. Default (255, 0, 0).
      - overlay_alpha (float, optional): Alpha [0..1] for overlay. Default 0.5.
      - return_image (str, optional): "overlay" (default) or "mask" to return mask alone.
      - timeout (int, optional): Request timeout seconds. Default 30.
      - num_workers (int, optional): Max concurrent requests. Default 20.
      - rate_limit (int, optional): Token bucket size per process. Default 50.
      - enable_global_rate_limit (bool, optional): Enable global rate limiting. Default True.

    Tool schema (example):
    {
      "type": "function",
      "function": {
        "name": "segment_image",
        "description": "Segment the current image using a remote segmentation service and return a masked image.",
        "parameters": {
          "type": "object",
          "properties": {
            "threshold": {"type": "number", "description": "Optional threshold for binarizing soft masks (0..1)."},
            "extra": {"type": "object", "description": "Extra fields to send to the service."}
          },
          "required": []
        }
      }
    }

    Expected service responses handled:
      - Direct image body (Content-Type startswith image/): treated as mask (unless response_content_is_overlay=True)
      - JSON containing one of:
          "overlay" (base64 or data-URI), "overlay_url" (URL),
          "mask" (base64 or data-URI), "mask_url" (URL),
          "masks" (list) or "overlays" (list): first element used. Keys inside follow same patterns.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Worker / rate limiting
        self.num_workers = config.get("num_workers", 20)
        self.rate_limit = config.get("rate_limit", 50)
        self.timeout = config.get("timeout", 30)
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_segmentation_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )

        # Service configuration
        url = config.get("segmentation_service_url")
        if not url:
            raise ValueError("segmentation_service_url is not set")
        self.segmentation_service_url: str = url
        self.payload_format = config.get("payload_format", "json_base64")
        assert self.payload_format in ("multipart", "json_base64"), (
            "payload_format must be 'multipart' or 'json_base64'"
        )

        self.image_field_name = config.get("image_field_name", "image")
        self.request_headers = config.get("request_headers", {}) or {}
        self.request_extra_fields = config.get("request_extra_fields", {}) or {}
        self.json_base64_data_uri = config.get("json_base64_data_uri", False)

        # Output configuration
        color = config.get("overlay_color", (255, 0, 0))
        self.overlay_color = tuple(color) if isinstance(color, (list, tuple)) else (255, 0, 0)
        self.overlay_alpha = float(config.get("overlay_alpha", 0.5))
        if self.overlay_alpha < 0.0:
            self.overlay_alpha = 0.0
        if self.overlay_alpha > 1.0:
            self.overlay_alpha = 1.0

        self.return_image = config.get("return_image", "overlay")  # "overlay" | "mask"
        if self.return_image not in ("overlay", "mask"):
            self.return_image = "overlay"

        self.response_content_is_overlay = bool(config.get("response_content_is_overlay", True))

        logger.info(f"Initialized ImageSegmentationTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a new instance bound to a single image.

        Accepts:
          - image: PIL.Image.Image | http(s) URL | local path | file:// URI | data URI (base64)
          Optionally can pass as create_kwargs={"image": ...}

        Dataset reminder (tools_kwargs → create_kwargs.image):
        - When preparing data (e.g., examples/data_preprocess/geo3k_multiturn_w_tool.py pattern),
          ensure your dataset row provides, per tool name, an image via tools_kwargs:
              tools_kwargs = {
                "segment_image": {
                  "create_kwargs": {
                    "image": "<http(s) URL | file:///path | data URI | PIL.Image>"
                  }
                }
              }
        - ToolAgentLoop will pass tools_kwargs["segment_image"]["create_kwargs"] into this create() call.
        - If 'image' is missing here, create() will raise ValueError.

        Naming reminder:
        - The tools_kwargs key "segment_image" must exactly match the tool function name defined in your tool config
          (recipe/tools/vision_tools.yaml → tool_schema.function.name). If the dataset uses a different key, the
          ToolAgentLoop will not find the tool and the call will fail.
        """
        if instance_id is None:
            instance_id = str(uuid4())

        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            kwargs.update(create_kwargs)

        image = kwargs.get("image")
        if image is None:
            raise ValueError("Missing required 'image' parameter in create()")

        # Accepts multiple input forms and returns a PIL image:
        #  - PIL.Image.Image (passes through)
        #  - http(s) URL (downloads and decodes)
        #  - local file path or file:// URI (loads from disk)
        #  - base64 data URI (e.g., data:image/png;base64,...) or raw base64

        # Handles the common glue so callers don’t need to branch on input type.
        # This is why it’s used in ImageZoomInTool,
        # and why I reused it here—to be consistent and robust with various image sources.
        img = fetch_image({"image": image})
        self._instance_dict[instance_id] = {
            "image": img,
        }
        return instance_id, ToolResponse()

    def _pil_to_bytes(self, pil_image, format_hint: Optional[str] = None) -> tuple[bytes, str]:
        """Convert a PIL image to bytes and return (bytes, mime)."""
        buf = io.BytesIO()
        # Decide a sensible default format
        fmt = (format_hint or getattr(pil_image, "format", None) or "PNG").upper()
        if fmt not in ("PNG", "JPEG", "WEBP", "BMP"):
            fmt = "PNG"
        mime = {
            "PNG": "image/png",
            "JPEG": "image/jpeg",
            "WEBP": "image/webp",
            "BMP": "image/bmp",
        }[fmt]
        pil_image.save(buf, format=fmt)
        return buf.getvalue(), mime

    def _build_multipart(self, image_bytes: bytes, mime: str, field_name: str, fields: dict[str, Any]):
        files = {
            field_name: (f"image.{mime.split('/')[-1]}", image_bytes, mime),
        }
        data = {}
        for k, v in fields.items():
            # Flatten simple extras as form fields
            data[k] = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
        return files, data

    def _build_json_base64(self, image_bytes: bytes, mime: str, field_name: str, fields: dict[str, Any]):
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
        payload = dict(fields)
        if self.json_base64_data_uri:
            payload[field_name] = f"data:{mime};base64,{img_b64}"
        else:
            payload[field_name] = img_b64
        return payload

    def _perform_request(
        self,
        service_url: str,
        headers: dict[str, str],
        payload_format: str,
        image_bytes: bytes,
        mime: str,
        image_field_name: str,
        fields: dict[str, Any],
        timeout: int,
    ) -> tuple[bytes, str, dict]:
        """Perform HTTP request to segmentation service. Returns (body_bytes, content_type, metadata)."""
        start = time.time()
        status_code = -1
        api_request_error = None
        try:
            if payload_format == "multipart":
                files, data = self._build_multipart(image_bytes, mime, image_field_name, fields)
                resp = requests.post(service_url, headers=headers, files=files, data=data, timeout=timeout)
            else:
                payload = self._build_json_base64(image_bytes, mime, image_field_name, fields)
                resp = requests.post(
                    service_url,
                    headers={"Content-Type": "application/json", **headers},
                    json=payload,
                    timeout=timeout,
                )

            status_code = resp.status_code
            content_type = resp.headers.get("Content-Type", "")
            body_bytes = resp.content

            latency_ms = int((time.time() - start) * 1000)
            metadata = {
                "status_code": status_code,
                "latency_ms": latency_ms,
                "api_request_error": api_request_error,
                "content_type": content_type,
            }
            return body_bytes, content_type, metadata
        except requests.RequestException as e:
            api_request_error = f"Request failed: {e}"
            latency_ms = int((time.time() - start) * 1000)
            metadata = {
                "status_code": status_code,
                "latency_ms": latency_ms,
                "api_request_error": api_request_error,
                "content_type": "",
            }
            return json.dumps({"error": api_request_error}).encode("utf-8"), "application/json", metadata

    def _decode_base64_image_to_pil(self, data: str) -> Image.Image:
        """Decode a base64 string or data URI to a PIL image."""
        if data.startswith("data:"):
            # data URI
            header, b64 = data.split(",", 1)
            return Image.open(io.BytesIO(base64.b64decode(b64)))
        # plain base64
        return Image.open(io.BytesIO(base64.b64decode(data)))

    def _load_image_from_url(self, url: str, timeout: int) -> Image.Image:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content))

    def _ensure_size(self, img: Image.Image, size: tuple[int, int]) -> Image.Image:
        if img.size != size:
            return img.resize(size, Image.NEAREST)
        return img

    def _mask_to_overlay(
        self, original: Image.Image, mask_img: Image.Image, alpha: float, color: tuple[int, int, int]
    ) -> Image.Image:
        """Create a color overlay on original guided by a single-channel mask."""
        orig_rgba = original.convert("RGBA")
        mask_gray = mask_img.convert("L")
        mask_gray = self._ensure_size(mask_gray, orig_rgba.size)

        # Alpha mask scaled by overlay alpha
        def scale(x: int) -> int:
            v = int(x * alpha)
            return 255 if v > 255 else v

        alpha_mask = mask_gray.point(scale)
        overlay = Image.new("RGBA", orig_rgba.size, (color[0], color[1], color[2], 0))
        overlay.putalpha(alpha_mask)
        composed = Image.alpha_composite(orig_rgba, overlay)
        return composed

    def _parse_response_to_image(
        self,
        original_image: Image.Image,
        body_bytes: bytes,
        content_type: str,
        threshold: Optional[float],
        timeout: int,
    ) -> tuple[Image.Image, str]:
        """Parse service response into a PIL image and description text."""
        # If image/*, decode directly
        if content_type.startswith("image/"):
            img = Image.open(io.BytesIO(body_bytes))
            if self.response_content_is_overlay:
                out = self._ensure_size(img, original_image.size)
                return out, "Returned overlay from service."
            else:
                # Treat as mask
                mask = self._ensure_size(img, original_image.size)
                if threshold is not None and 0.0 <= threshold <= 1.0:
                    # Binarize if desired
                    mask = mask.convert("L").point(lambda p: 255 if p >= int(threshold * 255) else 0)
                if self.return_image == "mask":
                    return mask.convert("L"), "Returned mask from service."
                overlay = self._mask_to_overlay(original_image, mask, self.overlay_alpha, self.overlay_color)
                return overlay, "Composited overlay from service mask."

        # Else try JSON
        text = body_bytes.decode("utf-8", errors="ignore")
        try:
            j = json.loads(text)
        except Exception:
            # Fallback: not JSON and not image
            # Return original image with warning text
            return original_image, f"Unrecognized response content_type='{content_type}'. Returning original image."

        # Support list containers
        if isinstance(j, dict):
            # overlay direct
            if "overlay" in j and isinstance(j["overlay"], str):
                overlay_img = self._decode_base64_image_to_pil(j["overlay"])
                overlay_img = self._ensure_size(overlay_img, original_image.size)
                return overlay_img, "Used overlay from JSON 'overlay' field."
            if "overlay_url" in j and isinstance(j["overlay_url"], str):
                overlay_img = self._load_image_from_url(j["overlay_url"], timeout)
                overlay_img = self._ensure_size(overlay_img, original_image.size)
                return overlay_img, "Used overlay from JSON 'overlay_url' field."

            # mask direct
            mask_candidate = None
            if "mask" in j and isinstance(j["mask"], str):
                mask_candidate = j["mask"]
            elif "mask_png" in j and isinstance(j["mask_png"], str):
                mask_candidate = j["mask_png"]
            if mask_candidate is not None:
                mask_img = self._decode_base64_image_to_pil(mask_candidate)
                mask_img = self._ensure_size(mask_img, original_image.size)
                if threshold is not None and 0.0 <= threshold <= 1.0:
                    mask_img = mask_img.convert("L").point(lambda p: 255 if p >= int(threshold * 255) else 0)
                if self.return_image == "mask":
                    return mask_img.convert("L"), "Used mask from JSON."
                overlay = self._mask_to_overlay(original_image, mask_img, self.overlay_alpha, self.overlay_color)
                return overlay, "Composited overlay from JSON mask."

            if "mask_url" in j and isinstance(j["mask_url"], str):
                mask_img = self._load_image_from_url(j["mask_url"], timeout)
                mask_img = self._ensure_size(mask_img, original_image.size)
                if threshold is not None and 0.0 <= threshold <= 1.0:
                    mask_img = mask_img.convert("L").point(lambda p: 255 if p >= int(threshold * 255) else 0)
                if self.return_image == "mask":
                    return mask_img.convert("L"), "Used mask from JSON 'mask_url'."
                overlay = self._mask_to_overlay(original_image, mask_img, self.overlay_alpha, self.overlay_color)
                return overlay, "Composited overlay from JSON 'mask_url'."

            # lists
            if "overlays" in j and isinstance(j["overlays"], list) and j["overlays"]:
                ov0 = j["overlays"][0]
                if isinstance(ov0, str):
                    overlay_img = self._decode_base64_image_to_pil(ov0)
                    overlay_img = self._ensure_size(overlay_img, original_image.size)
                    return overlay_img, "Used first overlay from JSON 'overlays'."
                if isinstance(ov0, dict):
                    if "overlay" in ov0 and isinstance(ov0["overlay"], str):
                        overlay_img = self._decode_base64_image_to_pil(ov0["overlay"])
                        overlay_img = self._ensure_size(overlay_img, original_image.size)
                        return overlay_img, "Used first overlay from JSON 'overlays[0].overlay'."
                    if "overlay_url" in ov0 and isinstance(ov0["overlay_url"], str):
                        overlay_img = self._load_image_from_url(ov0["overlay_url"], timeout)
                        overlay_img = self._ensure_size(overlay_img, original_image.size)
                        return overlay_img, "Used first overlay from JSON 'overlays[0].overlay_url'."

            if "masks" in j and isinstance(j["masks"], list) and j["masks"]:
                m0 = j["masks"][0]
                if isinstance(m0, str):
                    mask_img = self._decode_base64_image_to_pil(m0)
                elif isinstance(m0, dict):
                    if "mask" in m0 and isinstance(m0["mask"], str):
                        mask_img = self._decode_base64_image_to_pil(m0["mask"])
                    elif "mask_url" in m0 and isinstance(m0["mask_url"], str):
                        mask_img = self._load_image_from_url(m0["mask_url"], timeout)
                    else:
                        mask_img = None
                else:
                    mask_img = None

                if mask_img is not None:
                    mask_img = self._ensure_size(mask_img, original_image.size)
                    if threshold is not None and 0.0 <= threshold <= 1.0:
                        mask_img = mask_img.convert("L").point(lambda p: 255 if p >= int(threshold * 255) else 0)
                    if self.return_image == "mask":
                        return mask_img.convert("L"), "Used first mask from JSON 'masks'."
                    overlay = self._mask_to_overlay(original_image, mask_img, self.overlay_alpha, self.overlay_color)
                    return overlay, "Composited overlay from JSON 'masks[0]'."

        # Fallback: return original image
        return original_image, "Response did not contain recognized mask/overlay; returning original image."

    def _execute_once(
        self,
        instance_id: str,
        threshold: Optional[float],
        extra: dict[str, Any],
    ) -> tuple[Image.Image, dict, str]:
        """Synchronous execution body to be called through the execution pool."""
        original = self._instance_dict[instance_id]["image"]
        image_bytes, mime = self._pil_to_bytes(original)

        fields = dict(self.request_extra_fields)
        if threshold is not None:
            fields["threshold"] = threshold
        # Allow per-call override/append
        if extra:
            fields.update(extra)

        body_bytes, content_type, metadata = self._perform_request(
            service_url=self.segmentation_service_url,
            headers=self.request_headers,
            payload_format=self.payload_format,
            image_bytes=image_bytes,
            mime=mime,
            image_field_name=self.image_field_name,
            fields=fields,
            timeout=self.timeout,
        )

        try:
            out_img, how = self._parse_response_to_image(
                original_image=original,
                body_bytes=body_bytes,
                content_type=content_type,
                threshold=threshold,
                timeout=self.timeout,
            )
            return out_img, metadata, how
        except Exception as e:
            logger.error(f"Failed to parse segmentation response: {e}")
            return original, {**metadata, "api_request_error": f"parse_error: {e}"}, "Failed to parse response."

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute segmentation for the bound image.

        parameters:
          - threshold (float, optional): threshold in [0,1] if server returns soft masks and you want binarization
          - extra (dict, optional): merged into request payload
        """
        if instance_id not in self._instance_dict:
            error_msg = f"Instance {instance_id} not found. Call create(image=...) first."
            logger.error(error_msg)
            return ToolResponse(text=json.dumps({"result": error_msg})), 0.0, {}

        threshold = parameters.get("threshold", None)
        if threshold is not None:
            try:
                threshold = float(threshold)
            except Exception:
                threshold = None
        extra = parameters.get("extra", {}) or {}

        try:
            out_img, metadata, how = await self.execution_pool.execute.remote(
                self._execute_once, instance_id, threshold, extra
            )

            metrics = {
                "status_code": metadata.get("status_code", -1),
                "latency_ms": metadata.get("latency_ms", -1),
                "api_request_error": metadata.get("api_request_error"),
                "content_type": metadata.get("content_type"),
            }
            return ToolResponse(image=[out_img], text=how), 0.0, metrics
        except Exception as e:
            error_result = json.dumps({"result": f"Segmentation execution failed: {e}"})
            logger.error(f"[ImageSegmentationTool] Execution failed: {e}")
            return ToolResponse(text=error_result), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        # No numerical reward for segmentation; return 0.0
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
