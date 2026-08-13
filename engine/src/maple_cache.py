"""Bounded recent-KV primitives for Maple sliding-attention layers."""


def store_sliding_prefill_(packed_cache, keys, values):
    """Replace the bounded cache with the most recent prefill K/V entries."""
    capacity = packed_cache.shape[1]
    keep = min(keys.shape[1], capacity)
    keys = keys[:, -keep:]
    values = values[:, -keep:]
    hidden_width = keys.shape[-1]
    packed_cache[:, :keep, :hidden_width].copy_(keys)
    packed_cache[:, :keep, hidden_width:].copy_(values)
    return keep


def append_sliding_decode_(packed_cache, keys, values, past_tokens):
    """Append one decode K/V entry, evicting the oldest entry when full."""
    if keys.shape[1] != 1 or values.shape[1] != 1:
        raise ValueError("sliding decode append requires exactly one K/V entry")
    capacity = packed_cache.shape[1]
    stored = min(int(past_tokens), capacity)
    if stored == capacity:
        packed_cache[:, :-1].copy_(packed_cache[:, 1:].clone())
        destination = packed_cache[:, -1:]
    else:
        destination = packed_cache[:, stored:stored + 1]
    hidden_width = keys.shape[-1]
    destination[..., :hidden_width].copy_(keys)
    destination[..., hidden_width:].copy_(values)
    return min(stored + 1, capacity)


def sliding_cache_view(packed_cache, past_tokens):
    """Return the ordered recent history visible to the next decode query."""
    stored = min(int(past_tokens), packed_cache.shape[1])
    return packed_cache[:, :stored]
