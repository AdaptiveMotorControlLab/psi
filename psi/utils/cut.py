
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING, Generator

import torch
from torch import Tensor
from einops import rearrange, repeat

if TYPE_CHECKING:
    from transformers.models.auto.modeling_auto import _BaseModelWithGenerate

class EarlyStopForward(Exception):
    """An exception to stop the forward pass early."""
    def __init__(self, hidden: Tensor) -> None:
        super().__init__("Early stop forward pass")
        self.hidden = hidden

class ActivationCut(ABC):
    """An activation cut is a location (layer, position) in a model
    where we can capture and patch the activation vector of a transformer model.
    """
    name: str
    position: str

    def __init__(self, position: str) -> None:
        self.position = position

    @abstractmethod
    def capture(
        self,
        subject: "_BaseModelWithGenerate",
        input_ids: Tensor,
        positions: Tensor,
        attention_mask: Tensor | None = None,
        **kwargs
    ) -> Tensor:
        """Capture the model's activation at `positions`."""
        ...

    @abstractmethod
    @contextmanager
    def patch(
        self,
        subject: "_BaseModelWithGenerate",
        positions: Tensor | None,
        activation: Tensor | None,
    ) -> Generator[None, None, None]:
        '''Replace the native model's activation vector at `positions`
        with the provided `activation` and resume the forward pass.

        Args:
            subject: The transformer model to patch.
            positions: The positions in the sequence to patch, a tensor
                of shape (batch, num_positions,).
            activation: The activation tensor to use for patching, of shape
                (batch, num_positions * hidden_dim).
        '''
        ...

    @property
    @abstractmethod
    def fingerprint(self) -> str:
        """Return a unique fingerprint for the cut."""
        ...

class ResidualCut(ActivationCut):
    '''A vector cut that captures and patches the residual stream
    of a transformer model.'''

    name = 'residual'

    def __init__(self, layer: int, position: str) -> None:
        super().__init__(position=position)

        self.layer = layer

    @classmethod
    def from_config(cls, config: dict) -> "ResidualCut":
        """Create a ResidualCut from a configuration dictionary."""
        layer = config["layer"]
        position = config["position"]
        return cls(layer=layer, position=position)

    @property
    def fingerprint(self) -> str:
        return f"{self.name}_layer{self.layer}"

    def capture(
        self,
        subject: "_BaseModelWithGenerate",
        input_ids: Tensor,
        positions: Tensor,
        position_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        **kwargs
    ) -> Tensor:
        '''Record the model's activation at the cut point.

        Args:
            subject: The transformer model to capture from.
            input_ids: The input token IDs to feed into the model of shape (batch, seq_len).
            positions: A tensor of shape (batch, num_positions) indicating the positions
                indices in the sequence to capture.
            position_ids: An optional tensor of shape (batch, seq_len) indicating the
                position IDs to use for the input.
            attention_mask: An optional attention mask of shape (batch, seq_len).
            **kwargs: Additional keyword arguments to pass to the model's forward method.

        Returns:
            A tensor of shape (batch, num_positions * hidden_dim) containing the captured
                activation vectors at the specified layer and positions.
        '''

        def grab(_module, _inputs, output: Tensor | tuple[Tensor, ...]) -> None:
            hidden = output[0] if isinstance(output, tuple) else output

            b, s, h = hidden.shape
            extract = repeat(positions, "b n -> b n h", h=h)

            # Gather the requested token positions from the hidden state
            hidden = rearrange(
                hidden.gather(1, extract.to(hidden.device)),
                "b n h -> b (n h)",
            )

            raise EarlyStopForward(hidden)

        handle = subject.model.layers[self.layer].register_forward_hook(grab)

        try:
            # Trigger forward pass to capture the activation
            with torch.no_grad():
                subject(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    **kwargs
                )
        except EarlyStopForward as e:
            # Early stop is expected; we just want the activation
            return e.hidden
        finally:
            handle.remove()

        raise RuntimeError("Failed to capture activation; forward pass completed without triggering EarlyStopForward.")

    @contextmanager
    def patch(
        self,
        subject: "_BaseModelWithGenerate",
        positions: Tensor | None = None,
        activation: Tensor | None = None,
    ) -> Generator[None, None, None]:
        '''Replace the native model's activation vector at the cut point.

        Args:
            subject: The transformer model to patch.
            positions: The positions in the sequence to patch, a tensor
                of shape (num_positions,).
            activation: The activation tensor to use for patching, of shape
                (batch, num_positions * hidden_dim).
            **kwargs: Additional keyword arguments to pass to the model's forward method.
        '''
        # Check that positions is provided if activation is not None
        if activation is not None and positions is None:
            raise ValueError("Positions must be provided when activation is not None.")

        handle = None

        # Return a no-op context if activation is None
        if activation is None or positions is None:
            yield
            return

        def replace(_module, _inputs, output: Tensor | tuple[Tensor, ...]):
            hidden = output[0] if isinstance(output, tuple) else output
            edited = hidden.clone()

            b, s, h = hidden.shape
            p, n = positions.shape
            a, c = activation.shape

            assert b % a == 0, f"Batch size {b} must be divisible by activation batch size {a}"
            assert b % p == 0, f"Batch size {b} must be divisible by positions batch size {p}"

            assert c == n * h, (
                f"Activation width {c} must equal positions x hidden size "
                f"({n} x {h})."
            )

            pos = repeat(positions, "b n -> (b k) n h", k=b // p, h=h)
            act = repeat(activation, "b (n h) -> (b k) n h", k=b // a, h=h)
            edited.scatter_(1, pos.to(edited.device), act.to(edited))

            return (edited, *output[1:]) if isinstance(output, tuple) else edited

        handle = subject.model.layers[self.layer].register_forward_hook(replace)
        try:
            yield
        finally:
            if handle is not None:
                handle.remove()

class GPT2AttentionHeadCut(ActivationCut):
    '''A vector cut that captures and patches the attention head outputs
    of a GPT-2 transformer model.'''

    name = 'gpt2_attention_head'

    HEAD_DIM = 64  # The hidden dimension of each attention head in GPT-2
    NUM_HEADS = 12  # The number of attention heads in GPT-2

    def __init__(self, sites: tuple[tuple[int, int], ...], position: str) -> None:
        super().__init__(position=position)

        self.sites = sorted(sites)  # A tuple of (layer, head) pairs to capture/patch

    @classmethod
    def from_config(cls, config: dict) -> "GPT2AttentionHeadCut":
        """Create a GPT2AttentionHeadCut from a configuration dictionary."""
        sites = tuple((layer, head) for layer, head in config["sites"])
        position = config["position"]
        return cls(sites=sites, position=position)

    @property
    def fingerprint(self) -> str:
        """Return a unique fingerprint for the cut."""
        site_str = "_".join(f"l{layer}h{head}" for layer, head in self.sites)
        return f"{self.name}_{site_str}"

    def capture(
        self,
        subject: "_BaseModelWithGenerate",
        input_ids: Tensor,
        positions: Tensor,
        position_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        **kwargs
    ) -> Tensor:
        '''Record the model's activation at the cut point.

        Args:
            subject: The transformer model to capture from.
            input_ids: The input token IDs to feed into the model of shape (batch, seq_len).
            positions: A tensor of shape (batch, num_positions) indicating the positions
                indices in the sequence to capture.
            position_ids: An optional tensor of shape (batch, seq_len) indicating the
                position IDs to use for the input.
            attention_mask: An optional attention mask of shape (batch, seq_len).
            **kwargs: Additional keyword arguments to pass to the model's forward method.

        Returns:
            A tensor of shape (batch, num_positions * hidden_dim) containing the captured
                activation vectors at the specified layer and positions.
        '''

        b, p = positions.shape

        act_store = torch.empty((b, len(self.sites), p, self.HEAD_DIM), device=input_ids.device)

        # NOTE: We grab the input to the layer after the attention head.
        # The input has shape (batch, seq_len, hidden_dim = num_heads * head_dim),
        # where for gpt-2 num_heads = 12 and head_dim = 64, so hidden_dim = 768.
        def grab(_module, inputs: tuple[Tensor, ...], site: int, head: int) -> None:
            input = inputs[0] if isinstance(inputs, tuple) else inputs

            # Slice the requested head from the input tensor
            input = input[..., head * self.HEAD_DIM : (head + 1) * self.HEAD_DIM]

            b, s, h = input.shape # (h = len(heads) * head_dim)
            extract = repeat(positions, "b n -> b n h", h=h)

            # Gather the requested token positions from the hidden state
            heads_out = input.gather(1, extract.to(input.device)) # (b, p, h)
            act_store[:, site] = heads_out.detach()

        # Register forward pre-hooks on the layers after the attention heads
        handles = [
            subject.transformer.h[layer].attn.c_proj.register_forward_pre_hook( # type: ignore
                partial(grab, site=site, head=head)
            ) for site, (layer, head) in enumerate(self.sites)
        ]

        try:
            # Trigger forward pass to capture the activation
            with torch.no_grad():
                subject(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    **kwargs
                )
        finally:
            # Remove the hooks after capturing
            for handle in handles:
                handle.remove()

        return act_store.flatten(1)  # (b, len(sites) * p * head_dim)

    @contextmanager
    def patch(
        self,
        subject: "_BaseModelWithGenerate",
        positions: Tensor | None = None,
        activation: Tensor | None = None,
    ) -> Generator[None, None, None]:
        '''Replace the native model's activation vector at the cut point.

        Args:
            subject: The transformer model to patch.
            positions: The positions in the sequence to patch, a tensor
                of shape (batch, num_positions).
            activation: The activation tensor to use for patching, of shape
                (batch, num_positions * hidden_dim).
            **kwargs: Additional keyword arguments to pass to the model's forward method.
        '''
        # Check that positions is provided if activation is not None
        if activation is not None and positions is None:
            raise ValueError("Positions must be provided when activation is not None.")

        handles = None

        # Return a no-op context if activation is None
        if activation is None or positions is None:
            yield
            return

        b, p = positions.shape

        # Interpret activation as shape (batch, len(sites), num_positions, head_dim)
        activation = rearrange(
            activation, "b (s p h) -> b s p h",
            s=len(self.sites),
            p=p,
            h=self.HEAD_DIM
        )

        def replace(_module, inputs: tuple[Tensor, ...], site: int, head: int) -> Tensor | tuple[Tensor, ...]:
            input = inputs[0] if isinstance(inputs, tuple) else inputs

            # Select the appropriate site activation
            act = activation[:, site]  # (b, p, head_dim)

            mb, *_ = input.shape # (h = len(heads) * head_dim)
            pb, _ = positions.shape
            ab, *_ = act.shape

            assert mb % ab == 0, f"Model batch size {mb} must be divisible by activation batch size {ab}"
            assert mb % pb == 0, f"Model batch size {mb} must be divisible by positions batch size {pb}"

            start, stop = (head * self.HEAD_DIM, (head + 1) * self.HEAD_DIM)
            batch = torch.arange(mb, device=input.device)[:, None, None]
            shead = torch.arange(start, stop, device=input.device)[None, None, :]
            pos = repeat(positions, "b n -> (b k) n 1", k=mb // pb)
            act = repeat(act, "b n h -> (b k) n h", k=mb // ab)

            edited = input.clone()
            edited[batch, pos, shead] = act

            return (edited, *inputs[1:]) if isinstance(inputs, tuple) else edited

        handles = [
            subject.transformer.h[layer].attn.c_proj.register_forward_pre_hook( # type: ignore
                partial(replace, site=site, head=head)
            ) for site, (layer, head) in enumerate(self.sites)
        ]

        try:
            yield
        finally:
            if handles is not None:
                for handle in handles:
                    handle.remove()
