# Copyright 2024 Bytedance Ltd. and/or its affiliates
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


# CLUSTER LEVEL
# ╰─ AgentLoopManager (local, single)
#       ╰─ launches(instantiates) N remote and stateful workers across the cluster

# DISTRIBUTED WORKERS (Ray Actors)
# ╰─ AgentLoopWorker   <--- MUST be remote and stateful for maintaining its own eventloop for task scheduling
#        ╰─ initiates and runs an event loop per worker
#        ╰─ spawns N concurrent agent loop tasks
#        ╰─ owns AND instantiates routing helper-AsyncLLMServerManager
#        ╰─ owns AND instantiates a remote stateful reward ray actor
#        ╰─ interacts with vLLM servers

# PER-SAMPLE EXECUTION (local inside worker)
# ╰─ AgentLoopBase subclass   <--- MUST be local
#        ╰─ one instance per sample
#        ╰─ short-lived
#        ╰─ uses AsyncLLMServerManager.generate()

# ROUTING HELPER (local inside worker)
# ╰─ AsyncLLMServerManager    <--- MUST be local
#        ╰─ keeps heap, LRU, sticky routing
#        ╰─ talks to vLLM servers

# LLM SERVERS
# ╰─ vLLM actors  <--- remote


import asyncio
import heapq
import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Any, Optional

import hydra
import numpy as np
import ray
import torch
from cachetools import LRUCache
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict
from transformers import AutoProcessor, AutoTokenizer

from verl.experimental.reward import RewardManagerWorker
from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils.rollout_trace import (
    RolloutTraceConfig,
    rollout_trace_attr,
    rollout_trace_op,
)
from verl.utils.transferqueue_utils import tqbridge
from verl.workers.rollout.replica import TokenOutput, get_rollout_replica_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AsyncLLMServerManager:
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least requests load balancing
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching

    Flow:
    Request received
        ↓
    Check LRU cache → reuse server if exists
        ↓
    Else pick least-loaded server (heap top)
        ↓
    Dispatch async generation task to that server
        ↓
    Return future / Ray ObjectRef to caller
    """

    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], max_cache_size: int = 10000):
        """Initialize the AsyncLLMServerManager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            max_cache_size (int, optional): max cache size for request_id to server mapping. Defaults to 10000.
        """
        self.config = config
        self.server_handles = server_handles
        random.shuffle(self.server_handles)

        # Build a min-heap (O(n)) for least-requests load balancing; allows O(1) access to least-loaded server and O(log n) updates.
        self.weighted_serveres = [[0, (hash(server), server)] for server in server_handles]
        heapq.heapify(self.weighted_serveres)

        # Bounded O(1) LRU cache for request→server mapping; prevents unbounded dict growth and preserves request stickiness.
        # Same multi-turn completion ids goes to the same server initiating the completion for reusing the KV cache
        # When number of existing request ids exceed max cache size, drop the least used
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)

    def _choose_server(self, request_id: str) -> ray.actor.ActorHandle:
        # TODO: implement server pressure awareness load balancing
        # For now, fetch the corresponding existing server for the request_id
        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]

        # server weight heapfied, so poping the first always gets the least used server
        server = self.weighted_serveres[0][1][1]
        # least used server is used, so increment its usage
        self.weighted_serveres[0][0] += 1
        # update the heapfied list and reorder weight, so the first server is always the least used.
        heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])
        self.request_id_to_server[request_id] = server
        return server

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.

        Returns:
            TokenOutput: token output
        """
        server = self._choose_server(request_id)
        output = await server.generate.remote(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
        )
        return output


class AgentLoopMetrics(BaseModel):
    """Agent loop performance metrics."""

    generate_sequences: float = 0.0
    tool_calls: float = 0.0


class AgentLoopOutput(BaseModel):
    """Agent loop output."""

    prompt_ids: list[int]
    """Prompt token ids."""
    response_ids: list[int]
    """Response token ids including LLM generated token, tool response token."""
    response_mask: list[int]
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    response_logprobs: Optional[list[float]] = None
    """Log probabilities for the response tokens."""
    multi_modal_data: Optional[dict[str, Any]] = None
    """Multi-modal data for multi-modal tools."""
    reward_score: Optional[float] = None
    """Reward score for the trajectory."""
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    metrics: AgentLoopMetrics
    """Auxiliary performance metrics"""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class _InternalAgentLoopOutput(AgentLoopOutput):
    """Internal agent loop output with padded sequences."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: torch.Tensor
    """Padded prompt token ids."""
    response_ids: torch.Tensor
    """Padded response token ids."""
    input_ids: torch.Tensor
    """Padded input ids(prompt_ids + response_ids)."""
    position_ids: torch.Tensor
    """Padded position ids."""
    response_mask: torch.Tensor
    """Padded response mask."""
    attention_mask: torch.Tensor
    """Padded attention mask."""
    response_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities for the response tokens."""
    multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None
    """Multi-modal inputs for processors (e.g., pixel_values, image_grid_thw)."""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


# make hydra.utils.instantiate happy — wraps full DictConfig as one argument
# -----------------------------------------------------------------------------
# Hydra normally unpacks YAML config fields into keyword args when instantiating:
#   YAML:
#     _target_: my_module.MyManager
#     param1: 42
#     param2: "hello"
#   → calls: MyManager(param1=42, param2="hello")
#
# If MyManager expects a single DictConfig instead:
#   class MyManager:
#       def __init__(self, config: DictConfig): ...
# Hydra will raise:
#   TypeError: MyManager.__init__() got an unexpected keyword argument 'param1'
#
# _DummyConfig fixes this by wrapping the full DictConfig:
#   YAML:
#     _target_: my_module._DummyConfig
#     config:
#       _target_: my_module.MyManager
#       param1: 42
#       param2: "hello"
#
# Usage:
#   cfg = OmegaConf.load("conf/config.yaml")
#   dummy = hydra.utils.instantiate(cfg)          # -> _DummyConfig(config=<DictConfig>)
#   manager = hydra.utils.instantiate(dummy.config)  # -> MyManager(config=<DictConfig>)
#
# Result:
#   MyManager receives the entire DictConfig cleanly.
class _DummyConfig:
    def __init__(self, config: DictConfig) -> None:
        self.config = config


class AgentLoopBase(ABC):
    """An agent loop takes a input message, chat with OpenAI compatible LLM server and interact with various
    environments."""

    _class_initialized = False

    def __init__(
        self,
        trainer_config: _DummyConfig,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        **kwargs,
    ):
        """Initialize agent loop, each sample will have its own loop instance for async multiturn tool interaction, etc.

        Args:
            trainer_config (_DummyConfig): trainer config.
            server_manager (AsyncLLMServerManager): OpenAI compatible LLM server manager.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
            processor (AutoProcessor): Processor for process messages.
        """
        self.init_class(config=trainer_config.config, tokenizer=tokenizer, processor=processor, **kwargs)
        self.config = trainer_config.config
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.processor = processor
        # self.loop = asyncio.get_running_loop() stores the active event loop so the object
        # can schedule or manage async tasks consistently within the same coroutine context.
        # Super important
        self.loop = asyncio.get_running_loop()

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer, processor: AutoProcessor, **kwargs):
        """This is used to do heavy initialization work that should shared across all instances. It's only called once.

        Args:
            config (DictConfig): trainer config.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
            processor (AutoProcessor): Processor for process multi_modal data.
            **kwargs: extra kwargs from config file passed in by `hydra.utils.instantiate`.
        """
        if cls._class_initialized:
            return
        cls._class_initialized = True

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run agent loop to interact with LLM server and environment.

        Args:
            sampling_params (Dict[str, Any]): LLM sampling params.
            **kwargs: dataset fields from `verl.utils.dataset.RLHFDataset`.

        Returns:
            AgentLoopOutput: Agent loop output.
        """
        raise NotImplementedError


"""Agent loop registry: key is agent_name, value is a dict of agent loop config
used by hydra.utils.instantiate to initialize agent loop instance.

https://hydra.cc/docs/advanced/instantiate_objects/overview/

This is outside of all classes. it is global
"""
_agent_loop_registry: dict[str, dict] = {}


def register(agent_name: str):
    """
    Agent loop registry and decorator for Hydra-compatible instantiation.
    This registry maps each `agent_name` (e.g. "async", "sync") to a minimal Hydra config:
        {
            "agent_name": {"_target_": "<module_path>.<class_name>"}
        }

    The `register(agent_name)` decorator should be applied to subclasses of `AgentLoopBase`.
    When the decorated class is defined, its fully qualified import path is automatically
    recorded in `_agent_loop_registry`. This allows Hydra's `hydra.utils.instantiate()`
    to later create the appropriate agent loop instance dynamically.

    Example
    -------
    >>> @register("async")
    ... class AsyncAgentLoop(AgentLoopBase):
    ...     pass
    >>> _agent_loop_registry
    {'async': {'_target_': 'my_project.agent_loops.AsyncAgentLoop'}}

    >>> from hydra.utils import instantiate
    >>> loop = instantiate(_agent_loop_registry["async"])
    >>> isinstance(loop, AsyncAgentLoop)
    True

    Reference
    ---------
    https://hydra.cc/docs/advanced/instantiate_objects/overview/

    A decorator is just a callable that takes another callable and returns a callable.
    It wraps the function in another callable, meaning it can:
    - Preprocess or alter the inputs before calling the original function,
    - Skip the original call altogether,
    - Replace the result or wrap it with extra data,
    - Handle errors, logging, or caching around it.
    """

    def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        _agent_loop_registry[agent_name] = {"_target_": fqdn}
        return subclass

    return decorator


class AgentLoopWorkerBase:
    """Agent loop worker takes a batch of messages and run each message in an agent loop."""

    def __init__(
        self,
        config: DictConfig,
        server_handles: list[ray.actor.ActorHandle],
        reward_router_address: str = None,
    ):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
        """
        self.config = config

        # for recipe to change
        if not hasattr(self, "server_manager"):
            self.server_manager = AsyncLLMServerManager(config, server_handles)

        self.reward_router_address = reward_router_address

        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        # download huggingface model to local and return the cache directory
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.processor = hf_processor(local_path, trust_remote_code=True)

        agent_loop_config_path = config.actor_rollout_ref.rollout.agent.agent_loop_config_path
        # NOTE:
        # At import time, the @register(agent_name) decorator populates `_agent_loop_registry`
        # with minimal Hydra blueprints, e.g. {"async": {"_target_": "pkg.module.AsyncAgentLoop"}}.
        # These entries only record the class path for Hydra's dynamic instantiation.
        #
        # At runtime, once the composed Hydra config (`agent_loop_config`) is available,
        # we overwrite the placeholder entry with the full config node.
        # This "promotion" replaces the static class reference with the user-specified
        # Hydra configuration that includes both `_target_` and runtime parameters
        # (e.g. rollout_batch_size, timeout, etc.).
        #
        # In short: @register seeds the registry with class mappings,
        # and this line finalizes it with the active, parameterized config.
        if agent_loop_config_path:
            agent_loop_configs = OmegaConf.load(agent_loop_config_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config
        if self.config.actor_rollout_ref.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.actor_rollout_ref.model.custom_chat_template
            self.tokenizer.chat_template = self.config.actor_rollout_ref.model.custom_chat_template

        # Launch a RewardManagerWorker Ray actor with node affinity.
        #
        # The RewardManagerWorker is a long-lived remote process responsible for reward computation
        # and routing. We explicitly use Ray's NodeAffinitySchedulingStrategy to ensure the actor
        # runs on the same physical node as the current process. This guarantees low-latency
        # communication with any local model servers, ZeroMQ sockets, or shared GPU resources.
        #
        # Breakdown:
        #   - `RewardManagerWorker.options(...)` customizes how the actor is launched.
        #   - `NodeAffinitySchedulingStrategy(node_id=..., soft=False)` pins the actor to a specific
        #     node (here, the one running this code). `soft=False` means it is a hard constraint:
        #     Ray will fail to schedule if that node is unavailable.
        #   - `.remote(self.config, self.reward_router_address)` actually instantiates the actor,
        #     passing initialization arguments (config + reward router address).
        #   - The return value is a `ray.actor.ActorHandle`: a local proxy to the remote process,
        #     allowing non-blocking async calls such as:
        #         self.reward_manager_worker.compute_reward.remote(batch)
        #
        # In short, this line creates a remotely instantiated RewardManagerWorker pinned to the
        # current node — effectively a "remote class instance ready for asynchronous calls".

        # Current node (controller)
        # │
        # ├── get node_id of this node
        # │
        # ├── tell Ray: "launch RewardManagerWorker on this same node"
        # │
        # └── get back ActorHandle to the remote process
        #       ↓
        #    RewardManagerWorker (remote actor)
        #    ├── runs independently on same node
        #    ├── has access to same GPU / local router
        #    └── receives (config, reward_router_address)
        self.reward_manager_worker = RewardManagerWorker.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False,
            ),
        ).remote(self.config, self.reward_router_address)

        trace_config = self.config.actor_rollout_ref.rollout.get("trace", {})
        RolloutTraceConfig.init(
            self.config.trainer.project_name,
            self.config.trainer.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
        )

    # This decorator automatically handles conversions between `BatchMeta` and
    # `DataProto` in function parameters, and decides whether to sync function
    # output back to `BatchMeta` based on configuration(`put_data`). It supports
    # both synchronous and asynchronous functions (async def), and can control
    # whether to enable enhanced logic via the global `HAS_TQ` variable (when disabled,
    # simply calls the original function as-is).
    @tqbridge()
    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        # Build per-sample trajectory metadata from a flat list of sample indices.

        # `index` is a list where consecutive elements may refer to the same trajectory
        # (e.g., [0,0,0,1,1,2,2,2]). This function reconstructs how long each sample
        # has remained within its current trajectory segment by computing `rollout_n`:

        #     - If index[i] == index[i-1], we are continuing the same trajectory and
        #       increment the within-trajectory counter (`rollout_n += 1`).
        #     - Otherwise (first element or trajectory boundary), we reset `rollout_n` to 0.

        # This provides a lightweight way to recover episode/trajectory continuity from
        # a flattened dataset without explicitly storing full trajectories. The result is
        # a list of dictionaries, each containing:
        #     • step:        current global training step
        #     • sample_index: trajectory/episode ID for this sample
        #     • rollout_n:    how deep we are in this trajectory segment
        #     • validate:     whether this batch is from a validation rollout

        # Returns:
        #     list[dict]: Per-sample trajectory information aligned with `index`.
        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        # event loop for creating non blocking tasks (rollouts of samples)
        tasks = []
        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            tasks.append(asyncio.create_task(self._run_agent_loop(sampling_params, trajectory_info[i], **kwargs)))
        outputs = await asyncio.gather(*tasks)

        output = self._postprocess(outputs)
        return output

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        **kwargs,
    ) -> _InternalAgentLoopOutput:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=_DummyConfig(config=self.config),
                server_manager=self.server_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
            )
            output: AgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)

            # Some AgentLoop may have already computed the reward score, e.g SWE-agent.

            # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
            # prompt_ids: left padded with zeros (e.g., [0,0,0,0,1,2,3,4])
            # response_ids: right padded with zeros (e.g., [5,6,7,8,0,0,0,0])
            # input_ids: concatenation of prompt + response
            # Mask:
            # For example, if the prompt is [1,2,3,4] and the response is [5,6,7,(tool start)8,9(tool end),10,11,12]
            # - prompt_attention_mask: 0s for padding, 1s for tokens
            #   e.g., [0,0,0,0,1,1,1,1]
            # - response_attention_mask: 0s for padding, 1s for tokens
            #   e.g., [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]
            # attention_mask: concatenation of prompt_attention_mask and response_attention_mask
            #   e.g., [0,0,0,0,1,1,1,1(prompt),1,1,1,1,1,1,1,1,1,1,1,0,0,0,0(response)]
            # - response_mask: 1s for LLM generated tokens, 0 for tool response/padding tokens
            #   e.g., [1,1,1,1,1,1,1,(tool start),0,0(tool end),1,1,0,0,0,0]
            # - position_ids: sequential positions for tokens, starting at 0
            #   e.g., [0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,0,0,0,0]

            self.tokenizer.padding_side = "left"
            prompt_output = self.tokenizer.pad(
                {"input_ids": output.prompt_ids},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.prompt_length,
                return_tensors="pt",
                return_attention_mask=True,
            )

            # If input is a 1D sequence without batchsize
            if prompt_output["input_ids"].dim() == 1:
                prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
                prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

            self.tokenizer.padding_side = "right"
            response_output = self.tokenizer.pad(
                {"input_ids": output.response_ids},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.response_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            if response_output["input_ids"].dim() == 1:
                response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
                response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

            response_mask_output = self.tokenizer.pad(
                {"input_ids": output.response_mask},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.response_length,
                return_tensors="pt",
                return_attention_mask=False,
            )
            if response_mask_output["input_ids"].dim() == 1:
                response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

            # Set the log probs of response mask to 0
            response_logprobs = None
            if output.response_logprobs is not None:
                pad_size = self.config.actor_rollout_ref.rollout.response_length - len(output.response_logprobs)
                response_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)

            # Get the complete response_mask, attention_mask and token ids for the complete sequence
            # left padded concated with right padded
            response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
            attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
            input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

            # Handle multi-modal inputs and position_ids calculation
            # Only support Qwen2VLImageProcessor for multi-modal processing currently
            # TODO: support other multi-modal inputs
            multi_modal_inputs = None
            if (
                self.processor is not None
                and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
            ):
                from verl.models.transformers.qwen2_vl import get_rope_index

                images = getattr(output, "multi_modal_data", {}).get("image", None)
                # Decode the current text tokens into human-readable text.
                # This is needed because Qwen2-VL's processor takes raw text + images
                # and constructs a unified multimodal sequence (text tokens + image tokens),
                # including grid metadata required to compute RoPE indices for vision tokens.
                current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)

                # Run the Qwen2-VL processor to build multimodal inputs such as:
                #   - pixel_values / image embeddings
                #   - image_grid_thw (token grid: T/H/W)
                #   - video_grid_thw (for videos)
                #   - second_per_grid_ts (for temporal RoPE)
                # We request `return_tensors="pt"` so outputs are native PyTorch.
                multi_modal_inputs = self.processor(text=[current_text], images=images, return_tensors="pt")

                # We do not need the decode text converted again to input ids, we already have them
                # And this converted might not equal to the origin causing issue with RL later.
                multi_modal_inputs.pop("input_ids", None)
                multi_modal_inputs.pop("attention_mask", None)

                # We must use dict(multi_modal_inputs) to convert BatchFeature values to a new dict
                # because np.array() only keeps the keys for BatchFeature.
                multi_modal_inputs = dict(multi_modal_inputs)

                image_grid_thw = multi_modal_inputs.get("image_grid_thw")
                video_grid_thw = multi_modal_inputs.get("video_grid_thw")
                second_per_grid_ts = multi_modal_inputs.get("second_per_grid_ts")

                # Compute the *vision RoPE position indices* for Qwen2-VL.
                # get_rope_index() returns a 3-stream tensor of shape:
                #       (3, seq_len)
                # representing:
                #   stream 1: horizontal grid index
                #   stream 2: vertical grid index
                #   stream 3: temporal grid index (videos)
                #
                # These are NOT merged — Qwen2-VL expects 3 separate RoPE channels.
                vision_position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids.squeeze(0),
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask.squeeze(0),
                ).unsqueeze(0)  # (1, 3, seq_len)

                # Identify which tokens are non-padding so we can assign text positions only to real tokens.
                valid_mask = attention_mask[0].bool()

                # Build the text 1-D position IDs:
                #   shape: (1, seq_len)
                # Initialized with ones; then fill only valid positions with range(0, num_valid_tokens).
                text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
                text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
                text_position_ids = text_position_ids.unsqueeze(0)

                # Qwen2-VL multi-stream RoPE:
                #
                # The model expects:
                #     position_ids: (batch, num_streams, seq_len)
                #
                # where:
                #   stream 0 → text positions (1 stream)
                #   stream 1-3 → vision RoPE components (3 streams)
                #
                # So we concatenate:
                #   (1, 1, seq_len)  # text_position_ids
                #   (1, 3, seq_len)  # vision_position_ids
                #
                # Resulting in:
                #   (1, 4, seq_len)
                # which is exactly the format the Qwen2-VL transformer expects internally.

                # Qwen2-VL uses *separate* RoPE streams for text (1D) and vision (3-channel 2D/3D grids).
                # Text and image tokens do NOT share a unified positional timeline; their relative
                # positions are intentionally unspecified. Cross-modal alignment is learned through
                # attention (text ↔ vision) rather than positional offsets. Concatenating text and
                # vision RoPE streams gives the (1, 4, seq_len) format Qwen expects.
                position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)  # (1, 4, seq_length)
            else:
                position_ids = compute_position_id_with_mask(attention_mask)  # (1, seq_len)
            enable_async_reward = (
                self.reward_router_address is not None and self.config.reward_model.enable_resource_pool
            ) or not self.config.reward_model.enable

            # If the agent loop didn't compute a reward, send the completed prompt/response
            # sequence to the async RewardManagerWorker. We wrap tokens + metadata into a
            # DataProto, call the remote reward model, and store the returned reward score
            # and extra diagnostic info into the output.
            if output.reward_score is None and enable_async_reward:
                batch = TensorDict(
                    {
                        "prompts": prompt_output["input_ids"],  # [1, prompt_length]
                        "responses": response_output["input_ids"],  # [1, response_length]
                        "attention_mask": attention_mask,  # [1, prompt_length + response_length]
                        "input_ids": input_ids,  # [1, prompt_length + response_length]
                        "position_ids": position_ids,
                    },
                    batch_size=1,
                )
                non_tensor_batch = {
                    **{k: np.array([v]) for k, v in kwargs.items()},
                    "__num_turns__": np.array([output.num_turns]),
                    "tool_extra_fields": np.array([output.extra_fields], dtype=object),
                }

                data = DataProto(
                    batch=batch,
                    non_tensor_batch=non_tensor_batch,
                )
                result = await self.reward_manager_worker.compute_score.remote(data)
                output.reward_score = result["reward_score"]
                output.extra_fields["reward_extra_info"] = result["reward_extra_info"]

            return _InternalAgentLoopOutput(
                prompt_ids=prompt_output["input_ids"],
                response_ids=response_output["input_ids"],
                input_ids=input_ids,
                position_ids=position_ids,
                response_mask=response_mask,
                attention_mask=attention_mask,
                response_logprobs=response_logprobs,
                multi_modal_inputs=multi_modal_inputs,
                multi_modal_data=output.multi_modal_data,
                reward_score=output.reward_score,
                num_turns=output.num_turns,
                metrics=output.metrics,
                extra_fields=output.extra_fields,
            )

    def _postprocess(self, inputs: list[_InternalAgentLoopOutput]) -> DataProto:
        """Process the padded outputs from _run_agent_loop and combine them into a batch."""
        # Convert lists back to tensors and stack them to create a batch.
        prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
        response_ids = torch.cat([input.response_ids for input in inputs], dim=0)
        response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
        attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
        input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
        position_ids = torch.cat([input.position_ids for input in inputs], dim=0)
        optional_outputs = {}
        if inputs[0].response_logprobs is not None:
            optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)

        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                # position_ids: [bsz, 3, prompt_length + response_length] or [bsz, prompt_length + response_length]
                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=len(inputs),
        )

        scores = [input.reward_score for input in inputs]
        if all(score is not None for score in scores):
            prompt_length = prompt_ids.size(1)
            # Count valid response tokens, but subtract 1 to skip the final EOS token.
            # Reward is placed on the last meaningful generated token, not EOS.
            response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)
            # rm_scores is a [batch, response_mask.size(1)] matrix containing exactly one non-zero
            # reward per sample. We place the reward on the last *meaningful* generated token
            # (i.e., the final non-padding, non-EOS token). For example, if a response has
            # valid tokens [tok, tok, tok, tok, eos, pad], its valid count is 5 and the reward
            # is written at index 4. All other positions remain zero:
            #
            #   [
            #     [0, 0, 0, 0, 1.3, 0],   # reward for sample 0 at last real token
            #     [0, 0, 0, -0.7, 0, 0],  # reward for sample 1 at last real token
            #   ]
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
        }

        # add reward_extra_info to non_tensor_batch
        reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
        reward_extra_keys = list(reward_extra_infos[0].keys())
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

        # Add multi_modal_inputs to non_tensor_batch if any samples have them
        multi_modal_inputs_list = [input.multi_modal_inputs for input in inputs]
        if any(mmi is not None for mmi in multi_modal_inputs_list):
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        metrics = [input.metrics.model_dump() for input in inputs]
        # Collect extra fields from all inputs and convert them to np.ndarray
        extra_fields = {}
        all_keys = set(key for input_item in inputs for key in input_item.extra_fields)
        for key in all_keys:
            temp_arr = np.empty(len(inputs), dtype=object)
            temp_arr[:] = [input.extra_fields.get(key) for input in inputs]
            extra_fields[key] = temp_arr

        non_tensor_batch.update(extra_fields)
        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info={"metrics": metrics, "reward_extra_keys": reward_extra_keys},
        )

    def create_transferqueue_client(self, controller_infos, storage_infos, role):
        """Create a client for data system(transfer queue).
        Wrapper around the global transferqueue client factory. We generate a unique
        client_id for this worker (role_worker_xxxxxx) and forward it to the shared
        utility function. The names are the same but come from different namespaces:
        this is a convenience wrapper, not a duplicate definition.
        """
        from verl.single_controller.ray.base import get_random_string
        from verl.utils.transferqueue_utils import create_transferqueue_client

        client_name = get_random_string(length=6)
        create_transferqueue_client(
            client_id=f"{role}_worker_{client_name}",
            controller_infos=controller_infos,
            storage_infos=storage_infos,
        )


@ray.remote
class AgentLoopWorker(AgentLoopWorkerBase):
    """Agent loop worker takes a batch of messages and run each message in an agent loop."""

    def __init__(
        self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], reward_router_address: str = None
    ):
        """Initialize agent loop manager.
        Args:
            config (DictConfig): YAML config.
            server_handles returned by AsyncLLMServerManager (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            reward_router_address (str): reward router address.
        """
        super().__init__(config, server_handles, reward_router_address)


async def get_trajectory_info(step, index, validate):
    """Get trajectory info.

    Args:
        step (int): global steps in the trainer.
        index (list): from datastore extra_info.index column.
        validate (bool): whether is a validate step.

    Returns:
        list: trajectory.
    """
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n, "validate": validate})
    return trajectory_info


class AgentLoopManager:
    """Agent loop manager that manages a group of agent loop workers."""

    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup = None, rm_wg: RayWorkerGroup = None):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): trainer config.
            worker_group (RayWorkerGroup): ActorRolloutRef worker group for hybrid mode; None for standalone mode.
        """
        self.config = config
        self.worker_group = worker_group
        self.reward_model_manager = None
        self.reward_router_address = None
        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            from verl.experimental.reward import RewardModelManager

            self.reward_model_manager = RewardModelManager(config.reward_model, rm_wg)
            self.reward_router_address = self.reward_model_manager.get_router_address()

        # for recipe to change
        if not hasattr(self, "rollout_replica_class"):
            self.rollout_replica_class = get_rollout_replica_class(self.config.actor_rollout_ref.rollout.name)
        if not hasattr(self, "agent_loop_workers_class"):
            self.agent_loop_workers_class = AgentLoopWorker

        self._initialize_llm_servers()
        self._init_agent_loop_workers()

        # Initially we're in sleep mode.
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

    def _initialize_llm_servers(self):
        """
        Initialize all distributed LLM rollout servers used by VERL.

        This method determines how many GPUs a *single* model replica requires,
        how many GPUs are available in the cluster, how many full replicas can be
        launched in parallel, and then initializes those replicas as Ray actors
        running vLLM-backed OpenAI-compatible inference servers.

        - A Ray task is a stateless function executed remotely.
        - A Ray actor is a stateful class instance living on a remote worker process.

        ----------------------------------------------------------------------------
        1. Compute GPUs needed for one model replica
        One replica spans:
            tensor_model_parallel_size  (TP)
            x pipeline_model_parallel_size (PP)
            x data_parallel_size           (DP)
        GPUs. DP is intra-replica replication (synchronized copies *inside*
        a single model replica), not the number of rollout replicas.

        2. Determine total GPUs in the cluster
        - If hybrid training is enabled, use Ray WorkerGroup world_size.
        - Otherwise, compute from trainer config (n_gpus_per_node × nnodes).

        3. Determine how many replicas can be launched
            num_replicas = world_size // rollout_world_size
        This is *inter-model* parallelism: how many independent vLLM inference
        servers can run in parallel. It is distinct from DP.

        4. Instantiate rollout replica objects
        The replica class is obtained from get_rollout_replica_class(...).
        This class encapsulates all logic to configure and launch distributed
        vLLM (TPxPPxDP ranks, distributed groups, ports, cache settings).

        5. Launch vLLM servers (async initialization)
        Each rollout replica exposes async initialization methods:
            init_standalone() or init_hybrid()
        These are asynchronous because they internally bring up distributed
        vLLM executors, initialize process groups, and allocate GPU resources.

        Because these init functions return awaitables, VERL calls them via
        _run_all([...]) which executes them inside a temporary asyncio event
        loop to ensure correct async startup of all replicas.

        - Think of Evagelion's wunder start up sequence.
        - Multiple anti-gravity machines are started in a loop in a non-blocking fashion

        After initialization, each replica becomes a full vLLM-based
        OpenAI-compatible LLM server that handles:
            - token generation
            - KV-cache management
            - batching + scheduling
            - distributed execution over TPxPPxDP GPUs

        6. Store server handles and addresses
        Each replica exposes:
            _server_handle  → Ray actor handle for RPC
            _server_address → network endpoint for low-level communication
        These are gathered so AsyncLLMServerManager can route generation
        requests and manage sticky sessions.

        ----------------------------------------------------------------------------
        Interaction with AsyncLLMServerManager and RL rollouts
        ----------------------------------------------------------------------------
        - AsyncLLMServerManager performs async load balancing and sticky routing:
            new request_ids → least-loaded replica
            same request_id → same replica (for KV-cache reuse)
        - KV-cache locality is preserved across multi-turn episodes (tool-calling,
        ReAct, etc.).
        - Multiple vLLM replicas run independently to maximize rollout throughput.

        In summary:
        This method builds a distributed, multi-replica vLLM inference cluster,
        launched asynchronously through get_rollout_replica_class, and optimized
        for high-throughput, multi-turn, asynchronous RL rollouts.
        """
        rollout_world_size = (
            self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
            * self.config.actor_rollout_ref.rollout.data_parallel_size
            * self.config.actor_rollout_ref.rollout.pipeline_model_parallel_size
        )
        world_size = (
            self.worker_group.world_size
            if self.worker_group
            else self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        )
        # data_parallel_size (DP) is used *inside* one model replica (3-way replicated model),
        # while num_replicas is how many full replicas we can fit across the cluster.
        # Thus DP does NOT equal num_replicas. DP increases GPU cost per replica,
        # num_replicas = total_gpus // (TP * PP * DP).
        num_replicas = world_size // rollout_world_size

        rollout_config = self.config.actor_rollout_ref.rollout
        model_config = self.config.actor_rollout_ref.model
        self.rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=self.config.trainer.n_gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]
        if self.worker_group:
            self._run_all([server.init_hybrid(self.worker_group) for server in self.rollout_replicas])
        else:
            self._run_all([server.init_standalone() for server in self.rollout_replicas])
        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]

    def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            # All ray remote actors after decoration have the options class for specifying affinity
            # NOTE:
            # Calling `self.agent_loop_workers_class.options(...).remote(config, handles, ...)`
            # tells Ray to:
            #   1) Launch an AgentLoopWorker as a *remote Python process* on a selected node.
            #   2) Automatically set up all RPC sockets/channels used for communication.
            #   3) Initialize the actor’s dedicated asyncio event loop.
            #   4) Allow streaming of input batches via `worker.generate_sequences.remote(...)`.
            #   5) Schedule and run many internal async AgentLoop tasks concurrently.
            #   6) Return an ObjectRef for the async result once the actor finishes the call.
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                self.agent_loop_workers_class.options(
                    name=f"agent_loop_worker_{i}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(self.config, self.server_handles, self.reward_router_address)
            )

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """

        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.wake_up()
        if self.reward_model_manager and self.config.reward_model.rollout.free_cache_engine:
            self.reward_model_manager.wake_up()

        chunkes = prompts.chunk(len(self.agent_loop_workers))
        outputs = ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )
        output = DataProto.concat(outputs)
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()
        if self.reward_model_manager and self.config.reward_model.rollout.free_cache_engine:
            self.reward_model_manager.sleep()

        # calculate performance metrics
        metrics = [output.meta_info.pop("metrics") for output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)

        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output

    def _performance_metrics(self, metrics: list[list[dict[str, str]]], output: DataProto) -> dict[str, float]:
        timing = {}
        t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])
        t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])
        timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()
        timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
        timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
        timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()

        # batch sequence generation is bounded by the slowest sample
        slowest = np.argmax(t_generate_sequences + t_tool_calls)
        attention_mask = output.batch["attention_mask"][slowest]
        prompt_length = output.batch["prompts"].shape[1]
        timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
        timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()

        return timing

    def wake_up(self):
        """Wake up all rollout replica instances."""
        self._run_all([replica.wake_up() for replica in self.rollout_replicas])

    def sleep(self):
        """Sleep all rollout replica instances."""
        self._run_all([replica.sleep() for replica in self.rollout_replicas])

    def _run_all(self, tasks: list[asyncio.Task]):
        async def run_all():
            await asyncio.gather(*tasks)

        asyncio.run(run_all())
