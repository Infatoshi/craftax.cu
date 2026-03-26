// PyTorch C++ extension wrapper for Craftax CUDA kernels
#include <torch/extension.h>
#include "craftax_cuda.cuh"

// Forward declare kernels
extern "C" __global__ void reset_kernel(EnvState* states, float* obs, int num_envs, uint64_t seed);
extern "C" __global__ void step_kernel(EnvState* states, const int32_t* actions, float* obs, float* rewards, int8_t* dones, int num_envs, uint64_t reset_seed);
extern "C" __global__ void step_only_kernel(EnvState* states, const int32_t* actions, float* rewards, int8_t* dones, int num_envs);
extern "C" __global__ void autoreset_obs_kernel(EnvState* states, const int8_t* dones, float* obs, int num_envs, uint64_t reset_seed);

#include "craftax_cuda.cu"

static_assert(sizeof(EnvState) < 16384, "EnvState too large");

class CraftaxEnv {
public:
    torch::Tensor states;
    torch::Tensor obs;
    torch::Tensor rewards;
    torch::Tensor dones;
    int num_envs;
    uint64_t seed;
    uint64_t step_count;
    bool use_split_kernels;

    CraftaxEnv(int num_envs_, uint64_t seed_)
        : num_envs(num_envs_), seed(seed_), step_count(0), use_split_kernels(true) {

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
        int block = 256;
        int grid = (num_envs + block - 1) / block;
        reset_kernel<<<grid, block>>>(
            (EnvState*)states.data_ptr(),
            obs.data_ptr<float>(),
            num_envs, seed
        );
        step_count = 0;
        return obs;
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> step(torch::Tensor actions) {
        TORCH_CHECK(actions.device().is_cuda(), "actions must be on CUDA");
        TORCH_CHECK(actions.dtype() == torch::kInt32, "actions must be int32");

        step_count++;
        int block = 256;
        int grid = (num_envs + block - 1) / block;
        uint64_t reset_seed = seed + step_count * 1000000ULL;
        
        if (use_split_kernels) {
            // Split: step logic first (uniform work), then autoreset+obs (divergent)
            step_only_kernel<<<grid, block>>>(
                (EnvState*)states.data_ptr(),
                actions.data_ptr<int32_t>(),
                rewards.data_ptr<float>(),
                dones.data_ptr<int8_t>(),
                num_envs
            );
            autoreset_obs_kernel<<<grid, block>>>(
                (EnvState*)states.data_ptr(),
                dones.data_ptr<int8_t>(),
                obs.data_ptr<float>(),
                num_envs, reset_seed
            );
        } else {
            step_kernel<<<grid, block>>>(
                (EnvState*)states.data_ptr(),
                actions.data_ptr<int32_t>(),
                obs.data_ptr<float>(),
                rewards.data_ptr<float>(),
                dones.data_ptr<int8_t>(),
                num_envs, reset_seed
            );
        }
        return {obs, rewards, dones};
    }

    int get_obs_dim() { return OBS_DIM; }
    int get_num_actions() { return NUM_ACTIONS; }
    int get_num_envs() { return num_envs; }
    int get_state_size() { return (int)sizeof(EnvState); }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<CraftaxEnv>(m, "CraftaxEnv")
        .def(pybind11::init<int, uint64_t>(), pybind11::arg("num_envs") = 4096, pybind11::arg("seed") = 42)
        .def("reset", &CraftaxEnv::reset)
        .def("step", &CraftaxEnv::step)
        .def("get_obs_dim", &CraftaxEnv::get_obs_dim)
        .def("get_num_actions", &CraftaxEnv::get_num_actions)
        .def("get_num_envs", &CraftaxEnv::get_num_envs)
        .def("get_state_size", &CraftaxEnv::get_state_size)
        .def_readonly("obs", &CraftaxEnv::obs)
        .def_readonly("rewards", &CraftaxEnv::rewards)
        .def_readonly("dones", &CraftaxEnv::dones)
        .def_readwrite("use_split_kernels", &CraftaxEnv::use_split_kernels);
}
