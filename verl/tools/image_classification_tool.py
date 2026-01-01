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
from qwen_vl_utils import fetch_image

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

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


class ClassificationExecutionWorker:
    """Worker for executing classification requests with optional rate limiting."""

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
                    # TODO surface to caller if needed
                    logger.warning(f"Error when executing classification: {e}")
                    raise
        else:
            return fn(*fn_args, **fn_kwargs)


def init_classification_execution_pool(
    num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode
):
    """Initialize classification execution pool."""
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(ClassificationExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
        )
    else:
        raise NotImplementedError("Process mode is not implemented yet")


class ImageClassificationTool(BaseTool):
    """Classify an image using a remote HTTP service.

    Configuration (config dict):
      - classification_service_url (str, required): The HTTP endpoint to call.
      - payload_format (str, optional): "multipart" (default) | "json_base64".
      - image_field_name (str, optional): Form/JSON field name for the image. Default "image".
      - request_headers (dict, optional): Extra headers for the request.
      - request_extra_fields (dict, optional): Extra fields for the request (both multipart and json).
      - json_base64_data_uri (bool, optional): If True and payload_format="json_base64", encode as data URI
        e.g. "data:image/jpeg;base64,...". Default False (raw base64 string).
      - response_text_key (str, optional): If your service returns JSON and you want to extract a specific field
        for ToolResponse.text (e.g., "predictions"). If omitted, return raw text body.
      - response_strict_json (bool, optional): If True, only accept JSON responses; otherwise fall back to raw text.
      - num_workers (int, optional): Max concurrent logical requests. Default 20.
      - rate_limit (int, optional): Global token bucket size per process. Default 50.
      - timeout (int, optional): Request timeout (seconds). Default 30.
      - enable_global_rate_limit (bool, optional): Enable global rate limiting. Default True.

    Tool schema (example):
    {
      "type": "function",
      "function": {
        "name": "classify_image",
        "description": "Classify the current image using a remote classification service.",
        "parameters": {
          "type": "object",
          "properties": {
            "topk": {"type": "integer", "description": "Top-K predictions to return."},
            "extra": {"type": "object", "description": "Extra fields to send to the service."}
          },
          "required": []
        }
      }
    }

    Usage:
      tool = ImageClassificationTool(config, tool_schema)
      instance_id, _ = await tool.create(image="file:///path/to/image.jpg")
      resp, reward, metrics = await tool.execute(instance_id, parameters={"topk": 5})
      # resp.text will contain the service result (or extracted response_text_key)
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Worker / rate limiting
        self.num_workers = config.get("num_workers", 20)
        self.rate_limit = config.get("rate_limit", 50)
        self.timeout = config.get("timeout", 30)
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_classification_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )

        # Service configuration
        url = config.get("classification_service_url")
        if not url:
            raise ValueError("classification_service_url is not set")
        self.classification_service_url: str = url
        self.payload_format = config.get("payload_format", "multipart")
        assert self.payload_format in ("multipart", "json_base64"), "payload_format must be 'multipart' or 'json_base64'"

        self.image_field_name = config.get("image_field_name", "image")
        self.request_headers = config.get("request_headers", {}) or {}
        self.request_extra_fields = config.get("request_extra_fields", {}) or {}
        self.json_base64_data_uri = config.get("json_base64_data_uri", False)

        # Response parsing
        self.response_text_key = config.get("response_text_key")  # If provided, try to extract from JSON
        self.response_strict_json = config.get("response_strict_json", False)

        logger.info(f"Initialized ImageClassificationTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a new instance bound to a single image.

        Accepts:
          - image: PIL.Image.Image | http(s) URL | local path | file:// URI | data URI (base64)
          Optionally can pass as create_kwargs={"image": ...}
        """
        if instance_id is None:
            instance_id = str(uuid4())

        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            kwargs.update(create_kwargs)

        image = kwargs.get("image")
        if image is None:
            raise ValueError("Missing required 'image' parameter in create()")

        img = fetch_image({"image": image})
        self._instance_dict[instance_id] = {
            "image": img,
            "responses": [],
        }
        return instance_id, ToolResponse()

    def _pil_to_bytes(self, pil_image, format_hint: Optional[str] = None) -> tuple[bytes, str]:
        """Convert a PIL image to bytes and return (bytes, mime)."""
        buf = io.BytesIO()
        # Decide a sensible default format
        fmt = (format_hint or getattr(pil_image, "format", None) or "JPEG").upper()
        if fmt not in ("JPEG", "PNG", "WEBP", "BMP"):
            fmt = "JPEG"
        mime = {
            "JPEG": "image/jpeg",
            "PNG": "image/png",
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
    ) -> tuple[str, dict]:
        """Perform HTTP request to classification service. Returns (result_text, metadata)."""
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
            text = resp.text

            # Extract field if requested
            if self.response_text_key is not None or self.response_strict_json:
                try:
                    j = resp.json()
                    if self.response_text_key is not None:
                        # tolerate missing key by falling back to full JSON string
                        if self.response_text_key in j:
                            text = json.dumps(j[self.response_text_key], ensure_ascii=False)
                        else:
                            text = json.dumps(j, ensure_ascii=False)
                    else:
                        text = json.dumps(j, ensure_ascii=False)
                except Exception as e:
                    if self.response_strict_json:
                        api_request_error = f"JSON parse error: {e}"
                        text = json.dumps({"error": api_request_error, "raw": resp.text})
                    # else: keep raw text
            latency_ms = int((time.time() - start) * 1000)
            metadata = {
                "status_code": status_code,
                "latency_ms": latency_ms,
                "api_request_error": api_request_error,
            }
            return text, metadata
        except requests.RequestException as e:
            api_request_error = f"Request failed: {e}"
            latency_ms = int((time.time() - start) * 1000)
            metadata = {
                "status_code": status_code,
                "latency_ms": latency_ms,
                "api_request_error": api_request_error,
            }
            return json.dumps({"error": api_request_error}), metadata

    def _execute_once(
        self,
        instance_id: str,
        topk: Optional[int],
        extra: dict[str, Any],
    ) -> tuple[str, dict]:
        """Synchronous execution body to be called through the execution pool."""
        img = self._instance_dict[instance_id]["image"]
        image_bytes, mime = self._pil_to_bytes(img)

        fields = dict(self.request_extra_fields)
        if topk is not None:
            fields["topk"] = topk
        # Allow per-call override/append
        if extra:
            fields.update(extra)

        result_text, metadata = self._perform_request(
            service_url=self.classification_service_url,
            headers=self.request_headers,
            payload_format=self.payload_format,
            image_bytes=image_bytes,
            mime=mime,
            image_field_name=self.image_field_name,
            fields=fields,
            timeout=self.timeout,
        )
        return result_text, metadata

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute classification for the bound image.

        parameters:
          - topk (int, optional)
          - extra (dict, optional): merged into request payload
        """
        if instance_id not in self._instance_dict:
            error_msg = f"Instance {instance_id} not found. Call create(image=...) first."
            logger.error(error_msg)
            return ToolResponse(text=json.dumps({"result": error_msg})), 0.0, {}

        topk = parameters.get("topk", None)
        extra = parameters.get("extra", {}) or {}

        try:
            result_text, metadata = await self.execution_pool.execute.remote(
                self._execute_once, instance_id, topk, extra
            )

            # Track last response for reward history (text only)
            self._instance_dict[instance_id]["responses"].append(result_text.strip())

            metrics = {
                "status_code": metadata.get("status_code", -1),
                "latency_ms": metadata.get("latency_ms", -1),
                "api_request_error": metadata.get("api_request_error"),
            }
            return ToolResponse(text=result_text), 0.0, metrics
        except Exception as e:
            error_result = json.dumps({"result": f"Classification execution failed: {e}"})
            logger.error(f"[ImageClassificationTool] Execution failed: {e}")
            return ToolResponse(text=error_result), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> list[str]:
        return self._instance_dict[instance_id]["responses"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
