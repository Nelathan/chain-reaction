from __future__ import annotations

import ctypes

import torch


_OBS_DTYPE_MAP = {
    "ByteTensor": torch.uint8,
    "FloatTensor": torch.float32,
}

_TORCH_TO_TYPESTR = {
    torch.uint8: "|u1",
    torch.float32: "<f4",
}

_TORCH_TO_CTYPE = {
    torch.uint8: ctypes.c_uint8,
    torch.float32: ctypes.c_float,
}


class _CudaPtr:
    def __init__(self, ptr: int, shape: tuple[int, ...], dtype: torch.dtype):
        self.__cuda_array_interface__ = {
            "data": (ptr, False),
            "shape": shape,
            "typestr": _TORCH_TO_TYPESTR[dtype],
            "version": 2,
        }


def _cpu_tensor(ptr: int, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    ctype = _TORCH_TO_CTYPE[dtype]
    n = 1
    for size in shape:
        n *= size
    array = (ctype * n).from_address(ptr)
    return torch.frombuffer(array, dtype=dtype).reshape(shape)


class PufferVec:
    def __init__(self, args: dict):
        from pufferlib import _C

        self._C = _C
        self.gpu = bool(_C.gpu)
        self.device = torch.device("cuda" if self.gpu else "cpu")
        args["vec"]["num_buffers"] = 1
        self.vec = _C.create_vec(args, _C.gpu)
        self.total_agents = self.vec.total_agents
        self.obs_size = self.vec.obs_size
        self.num_atns = self.vec.num_atns
        if self.num_atns != 1:
            raise ValueError(f"expected one discrete action head, got {self.num_atns}")

        obs_dtype = _OBS_DTYPE_MAP.get(self.vec.obs_dtype, torch.float32)
        if self.gpu:
            self.observations = torch.as_tensor(_CudaPtr(
                self.vec.gpu_obs_ptr, (self.total_agents, self.obs_size), obs_dtype
            ))
            self.rewards = torch.as_tensor(_CudaPtr(
                self.vec.gpu_rewards_ptr, (self.total_agents,), torch.float32
            ))
            self.terminals = torch.as_tensor(_CudaPtr(
                self.vec.gpu_terminals_ptr, (self.total_agents,), torch.float32
            ))
        else:
            self.observations = _cpu_tensor(self.vec.obs_ptr, (self.total_agents, self.obs_size), obs_dtype)
            self.rewards = _cpu_tensor(self.vec.rewards_ptr, (self.total_agents,), torch.float32)
            self.terminals = _cpu_tensor(self.vec.terminals_ptr, (self.total_agents,), torch.float32)

        self.vec.reset()

    def step(self, actions: torch.Tensor) -> None:
        flat = actions.view(-1, 1).to(dtype=torch.float32, device=self.device).contiguous()
        if self.gpu:
            self.vec.gpu_step(flat.data_ptr())
            torch.cuda.synchronize()
        else:
            self.vec.cpu_step(flat.data_ptr())

    def log(self) -> dict:
        return self.vec.log()

    def close(self) -> None:
        self.vec.close()
