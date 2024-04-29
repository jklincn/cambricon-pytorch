#!/bin/bash

export PYTORCH_TESTING_DEVICE_ONLY_FOR='mlu'
TestList=(
# test_ao_sparsity.py
# test_autocast.py
# test_autograd.py
test_binary_ufuncs.py
# test_bundled_inputs.py
# test_comparison_utils.py
# test_complex.py
test_mlu.py
# test_mlu_primary_ctx.py
# test_mlu_sanitizer.py
# test_mlu_trace.py
# test_dataloader.py
# test_datapipe.py
# test_deploy.py
# test_determination.py
# test_dispatch.py
# test_dlpack.py
# test_dynamic_shapes.py
# test_expanded_weights.py
# test_fake_tensor.py
# test_function_schema.py
# test_functional_autograd_benchmark.py
# test_functional_optim.py
# test_functionalization.py
# test_futures.py
# test_fx.py
# test_fx_backends.py
# test_fx_experimental.py
# test_fx_passes.py
# test_fx_reinplace_pass.py
# test_hub.py
# test_import_stats.py
test_indexing.py
# test_itt.py
# test_license.py
test_linalg.py
# test_logging.py
# test_masked.py
# test_maskedtensor.py
test_meta.py
# test_mobile_optimizer.py
# test_model_dump.py
test_modules.py
# test_module_init.py
# test_monitor.py
# test_multiprocessing.py
# test_multiprocessing_spawn.py
# test_namedtensor.py
# test_namedtuple_return_api.py
# test_native_functions.py
# test_native_mha.py
test_nn.py
# test_nnapi.py
# test_numba_integration.py
# test_numpy_interop.py
# test_openmp.py
test_ops.py
# test_optim.py
test_overrides.py
# test_package.py
# test_per_overload_api.py
# test_proxy_tensor.py
# test_pruning_op.py
# test_public_bindings.py
# test_python_dispatch.py
# test_pytree.py
# test_quantization.py
test_reductions.py
# test_scatter_gather_ops.py
test_schema_check.py
# test_segment_reductions.py
# test_serialization.py
# test_set_default_mobile_cpu_allocator.py
test_shape_ops.py
# test_show_pickle.py
test_sort_and_select.py
# test_spectral_ops.py
# test_stateless.py
# test_static_runtime.py
# test_subclass.py
test_tensor_creation_ops.py
# test_tensorboard.py
# test_tensorexpr.py
# test_tensorexpr_pybind.py
# test_testing.py
# test_throughput_benchmark.py
test_torch.py
# test_transformers.py
# test_type_hints.py
# test_type_info.py
# test_type_promotion.py
# test_typing.py
test_unary_ufuncs.py
# test_utils.py
test_view_ops.py
test_expanded_weights.py
nn/test_embedding.py
nn/test_packed_sequence.py
# test_vmap.py
distributed/test_c10d_cncl.py
distributed/test_c10d_spawn_cncl.py
distributed/test_c10d_common.py
distributed/test_c10d_object_collectives.py
distributed/algorithms/test_join.py
distributed/algorithms/ddp_comm_hooks/test_ddp_hooks.py
)

for i in ${TestList[@]}
  do
  pytest $i
  done
