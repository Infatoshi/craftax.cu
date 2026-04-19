// PyTorch extension wrapper for opt2. SoA map buffer; block=64 launches.
#include <torch/extension.h>
#include "craftax_opt2.cuh"

extern "C" __global__ void reset_kernel(EnvState* states, uint8_t* maps, float* obs, int num_envs, uint64_t seed);
extern "C" __global__ void step_only_kernel(EnvState* states, uint8_t* maps, const int32_t* actions, float* rewards, int8_t* dones, int num_envs);
extern "C" __global__ void autoreset_obs_kernel(EnvState* states, uint8_t* maps, const int8_t* dones, float* obs, int num_envs, uint64_t reset_seed);

#include "craftax_opt2.cu"

class CraftaxEnvOpt2 {
public:
    torch::Tensor states, maps, obs, rewards, dones;
    int num_envs;
    uint64_t seed;
    uint64_t step_count;
    int block;

    CraftaxEnvOpt2(int num_envs_, uint64_t seed_, int block_)
        : num_envs(num_envs_), seed(seed_), step_count(0), block(block_) {
        auto opts_byte = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
        auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        auto opts_i8 = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA);
        int state_bytes = sizeof(EnvState) * num_envs;
        states = torch::zeros({state_bytes}, opts_byte);
        maps = torch::zeros({MAP_PACKED_SIZE * num_envs}, opts_byte);
        obs = torch::zeros({num_envs, OBS_DIM}, opts_f32);
        rewards = torch::zeros({num_envs}, opts_f32);
        dones = torch::zeros({num_envs}, opts_i8);
    }

    torch::Tensor reset() {
        int grid = (num_envs + block - 1) / block;
        reset_kernel<<<grid, block>>>(
            (EnvState*)states.data_ptr(), maps.data_ptr<uint8_t>(),
            obs.data_ptr<float>(), num_envs, seed);
        step_count = 0;
        return obs;
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> step(torch::Tensor actions) {
        TORCH_CHECK(actions.device().is_cuda(), "actions must be on CUDA");
        TORCH_CHECK(actions.dtype() == torch::kInt32, "actions must be int32");
        step_count++;
        int grid = (num_envs + block - 1) / block;
        uint64_t reset_seed = seed + step_count * 1000000ULL;
        step_only_kernel<<<grid, block>>>(
            (EnvState*)states.data_ptr(), maps.data_ptr<uint8_t>(),
            actions.data_ptr<int32_t>(),
            rewards.data_ptr<float>(), dones.data_ptr<int8_t>(), num_envs);
        autoreset_obs_kernel<<<grid, block>>>(
            (EnvState*)states.data_ptr(), maps.data_ptr<uint8_t>(),
            dones.data_ptr<int8_t>(),
            obs.data_ptr<float>(), num_envs, reset_seed);
        return {obs, rewards, dones};
    }

    int get_obs_dim() { return OBS_DIM; }
    int get_num_actions() { return NUM_ACTIONS; }
    int get_num_envs() { return num_envs; }
    int get_state_size() { return (int)sizeof(EnvState); }
    int get_block_size() { return block; }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<CraftaxEnvOpt2>(m, "CraftaxEnvOpt2")
        .def(pybind11::init<int, uint64_t, int>(),
             pybind11::arg("num_envs") = 4096,
             pybind11::arg("seed") = 42,
             pybind11::arg("block") = 64)
        .def("reset", &CraftaxEnvOpt2::reset)
        .def("step", &CraftaxEnvOpt2::step)
        .def("get_obs_dim", &CraftaxEnvOpt2::get_obs_dim)
        .def("get_num_actions", &CraftaxEnvOpt2::get_num_actions)
        .def("get_num_envs", &CraftaxEnvOpt2::get_num_envs)
        .def("get_state_size", &CraftaxEnvOpt2::get_state_size)
        .def("get_block_size", &CraftaxEnvOpt2::get_block_size)
        .def_readonly("obs", &CraftaxEnvOpt2::obs)
        .def_readonly("rewards", &CraftaxEnvOpt2::rewards)
        .def_readonly("dones", &CraftaxEnvOpt2::dones);
}
