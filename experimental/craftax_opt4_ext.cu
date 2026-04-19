// opt4 = opt + L2 persisting-access window on the states tensor.
// sm_120 has 128 MB L2, 80 MB persist cap. At NE up to ~34k our state buffer
// fits entirely. Mark it persistent so the AoS scatter reads hit L2, not DRAM.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "craftax_opt4.cuh"

extern "C" __global__ void reset_kernel(EnvState* states, float* obs, int num_envs, uint64_t seed);
extern "C" __global__ void step_only_kernel(EnvState* states, const int32_t* actions, float* rewards, int8_t* dones, int num_envs);
extern "C" __global__ void autoreset_obs_kernel(EnvState* states, const int8_t* dones, float* obs, int num_envs, uint64_t reset_seed);

#include "craftax_opt4.cu"

class CraftaxEnvOpt4 {
public:
    torch::Tensor states, obs, rewards, dones;
    int num_envs;
    uint64_t seed;
    uint64_t step_count;

    CraftaxEnvOpt4(int num_envs_, uint64_t seed_)
        : num_envs(num_envs_), seed(seed_), step_count(0) {
        auto opts_byte = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
        auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        auto opts_i8 = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA);
        int state_bytes = sizeof(EnvState) * num_envs;
        states = torch::zeros({state_bytes}, opts_byte);
        obs = torch::zeros({num_envs, OBS_DIM}, opts_f32);
        rewards = torch::zeros({num_envs}, opts_f32);
        dones = torch::zeros({num_envs}, opts_i8);

        install_l2_persist();
    }

    void install_l2_persist() {
        size_t state_bytes = sizeof(EnvState) * (size_t)num_envs;
        // Cap to the device persisting-L2 limit (80 MB on sm_120).
        int max_persist = 0;
        cudaDeviceGetAttribute(&max_persist, cudaDevAttrMaxPersistingL2CacheSize, 0);
        size_t bytes = state_bytes < (size_t)max_persist ? state_bytes : (size_t)max_persist;
        // Advertise our persisting size so the driver reserves L2 ways for it.
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, bytes);

        cudaAccessPolicyWindow w{};
        w.base_ptr  = states.data_ptr();
        w.num_bytes = bytes;
        w.hitRatio  = 1.0f;
        w.hitProp   = cudaAccessPropertyPersisting;
        w.missProp  = cudaAccessPropertyNormal;

        auto stream = c10::cuda::getCurrentCUDAStream().stream();
        cudaStreamAttrValue av{};
        av.accessPolicyWindow = w;
        cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &av);
    }

    torch::Tensor reset() {
        int block = 256, grid = (num_envs + block - 1) / block;
        reset_kernel<<<grid, block>>>((EnvState*)states.data_ptr(), obs.data_ptr<float>(), num_envs, seed);
        step_count = 0;
        return obs;
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> step(torch::Tensor actions) {
        TORCH_CHECK(actions.device().is_cuda(), "actions must be on CUDA");
        TORCH_CHECK(actions.dtype() == torch::kInt32, "actions must be int32");
        step_count++;
        int block = 256, grid = (num_envs + block - 1) / block;
        uint64_t reset_seed = seed + step_count * 1000000ULL;
        step_only_kernel<<<grid, block>>>(
            (EnvState*)states.data_ptr(), actions.data_ptr<int32_t>(),
            rewards.data_ptr<float>(), dones.data_ptr<int8_t>(), num_envs);
        autoreset_obs_kernel<<<grid, block>>>(
            (EnvState*)states.data_ptr(), dones.data_ptr<int8_t>(),
            obs.data_ptr<float>(), num_envs, reset_seed);
        return {obs, rewards, dones};
    }

    int get_obs_dim() { return OBS_DIM; }
    int get_num_actions() { return NUM_ACTIONS; }
    int get_num_envs() { return num_envs; }
    int get_state_size() { return (int)sizeof(EnvState); }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<CraftaxEnvOpt4>(m, "CraftaxEnvOpt4")
        .def(pybind11::init<int, uint64_t>(), pybind11::arg("num_envs") = 4096, pybind11::arg("seed") = 42)
        .def("reset", &CraftaxEnvOpt4::reset)
        .def("step", &CraftaxEnvOpt4::step)
        .def("get_obs_dim", &CraftaxEnvOpt4::get_obs_dim)
        .def("get_num_actions", &CraftaxEnvOpt4::get_num_actions)
        .def("get_num_envs", &CraftaxEnvOpt4::get_num_envs)
        .def("get_state_size", &CraftaxEnvOpt4::get_state_size)
        .def_readonly("obs", &CraftaxEnvOpt4::obs)
        .def_readonly("rewards", &CraftaxEnvOpt4::rewards)
        .def_readonly("dones", &CraftaxEnvOpt4::dones);
}
