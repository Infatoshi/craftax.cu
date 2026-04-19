// PyTorch wrapper for opt5 (warp-per-env).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "craftax_opt5.cuh"

extern "C" __global__ void reset_kernel(EnvState* states, float* obs, int num_envs, uint64_t seed);
extern "C" __global__ void step_only_kernel(EnvState* states, const int32_t* actions, float* rewards, int8_t* dones, int num_envs);
extern "C" __global__ void autoreset_obs_kernel(EnvState* states, const int8_t* dones, float* obs, int num_envs, uint64_t reset_seed);
extern "C" __global__ void step_fused_kernel(EnvState* states, const int32_t* actions, float* rewards, int8_t* dones, float* obs, int num_envs, uint64_t reset_seed);
extern "C" __global__ void step_fused_multistep_kernel(EnvState* states, const int32_t* actions_ms, float* rewards_ms, int8_t* dones_ms, float* obs_ms, int num_envs, int K, uint64_t reset_seed_base);

#include "craftax_opt5.cu"

class CraftaxEnvOpt5 {
public:
    torch::Tensor states, obs, rewards, dones;
    int num_envs;
    uint64_t seed;
    uint64_t step_count;

    CraftaxEnvOpt5(int num_envs_, uint64_t seed_)
        : num_envs(num_envs_), seed(seed_), step_count(0) {
        auto opts_byte = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
        auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        auto opts_i8 = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA);
        int state_bytes = sizeof(EnvState) * num_envs;
        states = torch::zeros({state_bytes}, opts_byte);
        obs = torch::zeros({num_envs, OBS_DIM}, opts_f32);
        rewards = torch::zeros({num_envs}, opts_f32);
        dones = torch::zeros({num_envs}, opts_i8);
    }

    torch::Tensor reset() {
        int grid = (num_envs + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        auto stream = c10::cuda::getCurrentCUDAStream().stream();
        reset_kernel<<<grid, BLOCK_SIZE, 0, stream>>>((EnvState*)states.data_ptr(), obs.data_ptr<float>(), num_envs, seed);
        step_count = 0;
        return obs;
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> step(torch::Tensor actions) {
        TORCH_CHECK(actions.device().is_cuda(), "actions must be on CUDA");
        TORCH_CHECK(actions.dtype() == torch::kInt32, "actions must be int32");
        step_count++;
        int grid = (num_envs + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        uint64_t reset_seed = seed + step_count * 1000000ULL;
        auto stream = c10::cuda::getCurrentCUDAStream().stream();
        step_fused_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            (EnvState*)states.data_ptr(), actions.data_ptr<int32_t>(),
            rewards.data_ptr<float>(), dones.data_ptr<int8_t>(),
            obs.data_ptr<float>(), num_envs, reset_seed);
        return {obs, rewards, dones};
    }

    // Multi-step: take [K, NE] actions, write [K, NE, OBS_DIM] obs, [K, NE] rewards/dones.
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> step_n(torch::Tensor actions_ms) {
        TORCH_CHECK(actions_ms.device().is_cuda(), "actions must be on CUDA");
        TORCH_CHECK(actions_ms.dtype() == torch::kInt32, "actions must be int32");
        TORCH_CHECK(actions_ms.dim() == 2, "actions must be [K, NE]");
        int K = actions_ms.size(0);
        TORCH_CHECK(actions_ms.size(1) == num_envs, "actions shape mismatch");

        auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        auto opts_i8 = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA);
        auto obs_ms = torch::empty({K, num_envs, OBS_DIM}, opts_f32);
        auto rew_ms = torch::empty({K, num_envs}, opts_f32);
        auto dn_ms  = torch::empty({K, num_envs}, opts_i8);

        int grid = (num_envs + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        step_count += K;
        uint64_t reset_seed_base = seed + (step_count - K + 1) * 1000000ULL;
        auto stream = c10::cuda::getCurrentCUDAStream().stream();
        step_fused_multistep_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            (EnvState*)states.data_ptr(), actions_ms.data_ptr<int32_t>(),
            rew_ms.data_ptr<float>(), dn_ms.data_ptr<int8_t>(),
            obs_ms.data_ptr<float>(), num_envs, K, reset_seed_base);
        return {obs_ms, rew_ms, dn_ms};
    }

    int get_obs_dim() { return OBS_DIM; }
    int get_num_actions() { return NUM_ACTIONS; }
    int get_num_envs() { return num_envs; }
    int get_state_size() { return (int)sizeof(EnvState); }
    int get_block_size() { return BLOCK_SIZE; }
    int get_warps_per_block() { return WARPS_PER_BLOCK; }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<CraftaxEnvOpt5>(m, "CraftaxEnvOpt5")
        .def(pybind11::init<int, uint64_t>(), pybind11::arg("num_envs") = 4096, pybind11::arg("seed") = 42)
        .def("reset", &CraftaxEnvOpt5::reset)
        .def("step", &CraftaxEnvOpt5::step)
        .def("step_n", &CraftaxEnvOpt5::step_n)
        .def("get_obs_dim", &CraftaxEnvOpt5::get_obs_dim)
        .def("get_num_actions", &CraftaxEnvOpt5::get_num_actions)
        .def("get_num_envs", &CraftaxEnvOpt5::get_num_envs)
        .def("get_state_size", &CraftaxEnvOpt5::get_state_size)
        .def("get_block_size", &CraftaxEnvOpt5::get_block_size)
        .def("get_warps_per_block", &CraftaxEnvOpt5::get_warps_per_block)
        .def_readonly("obs", &CraftaxEnvOpt5::obs)
        .def_readonly("rewards", &CraftaxEnvOpt5::rewards)
        .def_readonly("dones", &CraftaxEnvOpt5::dones);
}
