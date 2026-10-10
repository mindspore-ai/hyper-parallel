/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include <ATen/core/grad_mode.h>
#include <torch/library.h>

#include <array>
#include <cmath>
#include <mutex>
#include <tuple>

#include "op_plugin/include/npu_cpp_extension.h"

namespace {

void GetMegaGateOpApiFunctionsOnce(std::once_flag &resolve_once, const char *api_name, const char *workspace_api_name,
                                   void *&execute, void *&get_workspace_size) {
  std::call_once(resolve_once, [&]() { ::GetApiFunc(api_name, workspace_api_name, execute, get_workspace_size); });
}

// EXEC_NPU_CMD_EXT otherwise resolves both symbols on every invocation. Each
// V1/V2 expansion owns its once flag, preserving the macro's per-call-site
// function pointers without serializing independent Route calls.
#define GetApiFunc(api_name, workspace_api_name, execute, get_workspace_size)                               \
  do {                                                                                                      \
    static std::once_flag resolve_once;                                                                     \
    GetMegaGateOpApiFunctionsOnce(resolve_once, api_name, workspace_api_name, execute, get_workspace_size); \
  } while (false)

void CheckRuntimeResources(const at::Tensor &reference, const at::Tensor &runtime_config,
                           const at::Tensor &profile_buffer, const char *operation) {
  const std::array<const at::Tensor *, 2> resources{&runtime_config, &profile_buffer};
  for (const auto *resource : resources) {
    TORCH_CHECK(resource->dim() == 1 && resource->scalar_type() == at::kByte && resource->is_contiguous() &&
                  resource->device() == reference.device(),
                operation, " runtime resources must be contiguous UINT8 vectors on the input device");
  }
}

void CheckRouteInputs(const at::Tensor &logits, const at::Tensor &text_bias, const at::Tensor &vision_bias,
                      const at::Tensor &image_mask, const at::Tensor &runtime_config, const at::Tensor &profile_buffer,
                      int64_t top_k, double routed_scaling_factor, bool use_vision_bias) {
  TORCH_CHECK(logits.dim() == 2 && text_bias.dim() == 1 && vision_bias.dim() == 1,
              "HyperMegaGateRoute inputs must have logits [tokens, experts] and biases [experts]");
  const auto tokens = logits.size(0);
  const auto expert_count = logits.size(1);
  TORCH_CHECK(tokens > 0 && expert_count > 0, "HyperMegaGateRoute dimensions must be positive");
  TORCH_CHECK(text_bias.size(0) == expert_count && vision_bias.size(0) == expert_count,
              "HyperMegaGateRoute biases must match the expert dimension");
  TORCH_CHECK(image_mask.dim() == 1 && image_mask.scalar_type() == at::kBool && image_mask.is_contiguous() &&
                image_mask.size(0) == (use_vision_bias ? tokens : 1),
              "HyperMegaGateRoute image_mask must be BOOL [tokens] for vision routing or BOOL [1] otherwise");
  TORCH_CHECK(top_k > 0 && top_k <= expert_count, "HyperMegaGateRoute top_k must be in [1, expert_count]");
  TORCH_CHECK(logits.scalar_type() == at::kFloat && text_bias.scalar_type() == at::kFloat &&
                vision_bias.scalar_type() == at::kFloat,
              "HyperMegaGateRoute requires FP32 logits and biases");
  TORCH_CHECK(logits.is_contiguous() && text_bias.is_contiguous() && vision_bias.is_contiguous(),
              "HyperMegaGateRoute requires contiguous inputs");
  TORCH_CHECK(logits.device() == text_bias.device() && logits.device() == vision_bias.device() &&
                logits.device() == image_mask.device(),
              "HyperMegaGateRoute inputs must be on the same device");
  CheckRuntimeResources(logits, runtime_config, profile_buffer, "HyperMegaGateRoute");
  TORCH_CHECK(std::isfinite(routed_scaling_factor), "routed_scaling_factor must be finite");
  TORCH_CHECK(!at::GradMode::is_enabled() ||
                (!logits.requires_grad() && !text_bias.requires_grad() && !vision_bias.requires_grad()),
              "low-level HyperMegaGateRoute requires the module autograd bridge for gradients");
}

using RouteOutputs = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

RouteOutputs MegaGateRouteNpu(const at::Tensor &logits, const at::Tensor &text_bias, const at::Tensor &vision_bias,
                              const at::Tensor &image_mask, const at::Tensor &runtime_config,
                              const at::Tensor &profile_buffer, int64_t top_k, double routed_scaling_factor,
                              bool use_vision_bias) {
  CheckRouteInputs(logits, text_bias, vision_bias, image_mask, runtime_config, profile_buffer, top_k,
                   routed_scaling_factor, use_vision_bias);
  const auto tokens = logits.size(0);
  const auto expert_count = logits.size(1);
  auto routing_weights = at::empty({tokens, top_k}, logits.options());
  auto expert_indices = at::empty({tokens, top_k}, logits.options().dtype(at::kLong));
  auto route_scores = at::empty({tokens, expert_count}, logits.options());
  auto selected_scores = at::empty({tokens, top_k}, logits.options());
  auto normalization_denominator = at::empty({tokens, 1}, logits.options());
  EXEC_NPU_CMD_EXT(aclnnHyperMegaGateRoute, logits, text_bias, vision_bias, image_mask, runtime_config, profile_buffer,
                   routing_weights, expert_indices, route_scores, selected_scores, normalization_denominator, top_k,
                   routed_scaling_factor, use_vision_bias);
  return std::make_tuple(routing_weights, expert_indices, route_scores, selected_scores, normalization_denominator);
}

RouteOutputs MegaGateRouteMeta(const at::Tensor &logits, const at::Tensor &text_bias, const at::Tensor &vision_bias,
                               const at::Tensor &image_mask, const at::Tensor &runtime_config,
                               const at::Tensor &profile_buffer, int64_t top_k, double routed_scaling_factor,
                               bool use_vision_bias) {
  CheckRouteInputs(logits, text_bias, vision_bias, image_mask, runtime_config, profile_buffer, top_k,
                   routed_scaling_factor, use_vision_bias);
  const auto tokens = logits.size(0);
  const auto expert_count = logits.size(1);
  return std::make_tuple(at::empty({tokens, top_k}, logits.options()),
                         at::empty({tokens, top_k}, logits.options().dtype(at::kLong)),
                         at::empty({tokens, expert_count}, logits.options()),
                         at::empty({tokens, top_k}, logits.options()), at::empty({tokens, 1}, logits.options()));
}

void CheckRouteGradInputs(const at::Tensor &logits, const at::Tensor &route_scores, const at::Tensor &selected_scores,
                          const at::Tensor &normalization_denominator, const at::Tensor &expert_indices,
                          const at::Tensor &grad_routing_weights, const at::Tensor &runtime_config,
                          const at::Tensor &profile_buffer, int64_t top_k, double routed_scaling_factor) {
  TORCH_CHECK(logits.dim() == 2 && logits.size(0) > 0,
              "RouteGrad logits must have positive [tokens, experts] dimensions");
  const auto expert_count = logits.size(1);
  TORCH_CHECK(expert_count > 0 && top_k > 0 && top_k <= expert_count, "RouteGrad top_k must be in [1, expert_count]");
  TORCH_CHECK(logits.scalar_type() == at::kFloat && logits.is_contiguous(), "RouteGrad logits must be contiguous FP32");
  TORCH_CHECK(std::isfinite(routed_scaling_factor), "routed_scaling_factor must be finite");
  TORCH_CHECK(expert_indices.dim() == 2 && expert_indices.size(0) == logits.size(0) &&
                expert_indices.size(1) == top_k && grad_routing_weights.dim() == 2 &&
                grad_routing_weights.size(0) == logits.size(0) && grad_routing_weights.size(1) == top_k &&
                route_scores.sizes() == logits.sizes() && selected_scores.sizes() == grad_routing_weights.sizes() &&
                normalization_denominator.dim() == 2 && normalization_denominator.size(0) == logits.size(0) &&
                normalization_denominator.size(1) == 1,
              "RouteGrad inputs have incompatible shapes");
  TORCH_CHECK(route_scores.scalar_type() == at::kFloat && selected_scores.scalar_type() == at::kFloat &&
                normalization_denominator.scalar_type() == at::kFloat && expert_indices.scalar_type() == at::kLong &&
                grad_routing_weights.scalar_type() == at::kFloat,
              "RouteGrad requires INT64 indices and FP32 gradients");
  TORCH_CHECK(route_scores.is_contiguous() && selected_scores.is_contiguous() &&
                normalization_denominator.is_contiguous() && expert_indices.is_contiguous() &&
                grad_routing_weights.is_contiguous(),
              "RouteGrad requires contiguous inputs");
  TORCH_CHECK(logits.device() == route_scores.device() && logits.device() == selected_scores.device() &&
                logits.device() == normalization_denominator.device() && logits.device() == expert_indices.device() &&
                logits.device() == grad_routing_weights.device(),
              "RouteGrad inputs must be on the same device");
  CheckRuntimeResources(logits, runtime_config, profile_buffer, "HyperMegaGateRouteGrad");
}

at::Tensor MegaGateRouteGradNpu(const at::Tensor &logits, const at::Tensor &route_scores,
                                const at::Tensor &selected_scores, const at::Tensor &normalization_denominator,
                                const at::Tensor &expert_indices, const at::Tensor &grad_routing_weights,
                                const at::Tensor &runtime_config, const at::Tensor &profile_buffer, int64_t top_k,
                                double routed_scaling_factor) {
  CheckRouteGradInputs(logits, route_scores, selected_scores, normalization_denominator, expert_indices,
                       grad_routing_weights, runtime_config, profile_buffer, top_k, routed_scaling_factor);
  auto grad_logits = at::empty_like(logits);
  EXEC_NPU_CMD_EXT(aclnnHyperMegaGateRouteGrad, logits, route_scores, selected_scores, normalization_denominator,
                   expert_indices, grad_routing_weights, runtime_config, profile_buffer, grad_logits, top_k,
                   routed_scaling_factor);
  return grad_logits;
}

at::Tensor MegaGateRouteGradMeta(const at::Tensor &logits, const at::Tensor &route_scores,
                                 const at::Tensor &selected_scores, const at::Tensor &normalization_denominator,
                                 const at::Tensor &expert_indices, const at::Tensor &grad_routing_weights,
                                 const at::Tensor &runtime_config, const at::Tensor &profile_buffer, int64_t top_k,
                                 double routed_scaling_factor) {
  CheckRouteGradInputs(logits, route_scores, selected_scores, normalization_denominator, expert_indices,
                       grad_routing_weights, runtime_config, profile_buffer, top_k, routed_scaling_factor);
  return at::empty_like(logits);
}

}  // namespace

TORCH_LIBRARY_IMPL(hyper_parallel, PrivateUse1, m) {
  m.impl("mega_gate_route", &MegaGateRouteNpu);
  m.impl("mega_gate_route_grad", &MegaGateRouteGradNpu);
}

TORCH_LIBRARY_IMPL(hyper_parallel, Meta, m) {
  m.impl("mega_gate_route", &MegaGateRouteMeta);
  m.impl("mega_gate_route_grad", &MegaGateRouteGradMeta);
}

#undef GetApiFunc
