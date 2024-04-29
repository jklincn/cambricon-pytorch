#!/usr/bin/env python
#pylint: disable=C0301,W0613,W0611
from __future__ import print_function

import argparse
from datetime import datetime
import os
import shutil
import subprocess
import sys
import tempfile
import logging

import torch
import torch._six
from torch.utils import cpp_extension
from common_utils import shell, print_to_stderr, gen_err_message

# Note: torch_ops/ & custom_ops/ are dirs to be expanded in get_selected_tests
# items in NATIVE_CI_BLACKLIST are also added to TESTS below
TESTS = [
    'mlu/test_event',
    'mlu/test_stream',
    'mlu/test_tf32_ctrl',
    'mlu/test_lazy_init',
    'mlu/test_autograd',
    'torch_ops/',
    'custom_ops/',
    'torch/test_save_and_load',
    'torch/test_random',
    'distributed/test_distributed',
    'torch/test_pin_memory',
    'torch/test_complex',
    'torch/test_dataloader',
    'torch/test_set_default_type',
    'utils/test_cnnl_op_exception',
    'utils/test_counter',
    'utils/test_fakecase_print_log',
    'utils/test_utils',
    'optimizer/test_fused_adam',
    'optimizer/test_fused_lamb',
    'optimizer/test_fused_sgd',
    'distributions/test_distributions',
    'fallback/test_fallback',
    'view_chain/test_close_view_chain',
    'view_chain/test_close_view_chain_fuser',
    'view_chain/test_print_view_chain',
    'mlu/test_device',
    'mlu/test_mlu',
    'mlu/test_amp',
    'profiler/test_profiler',
    'mlu/test_caching_allocator',
    'cpp_extension/test_mlu_extension',
    'cnnl_gtest',
    'common_gtest',
    'codegen/test_gen_mlu_stubs',
    'storage/test_storage'
]

CNNL_BLACKLIST = [
    'torch_ops/',
]

NATIVE_CI_BLACKLIST = [
    # 'torch_native_ci/test_ao_sparsity',
    # 'torch_native_ci/test_autocast',
    # 'torch_native_ci/test_autograd',
    'torch_native_ci/test_binary_ufuncs',
    # 'torch_native_ci/test_bundled_inputs',
    # 'torch_native_ci/test_comparison_utils',
    # 'torch_native_ci/test_complex',
    'torch_native_ci/test_mlu',
    # 'torch_native_ci/test_mlu_primary_ctx',
    # 'torch_native_ci/test_mlu_sanitizer',
    # 'torch_native_ci/test_mlu_trace',
    # 'torch_native_ci/test_dataloader',
    # 'torch_native_ci/test_datapipe',
    # 'torch_native_ci/test_deploy',
    # 'torch_native_ci/test_determination',
    # 'torch_native_ci/test_dispatch',
    # 'torch_native_ci/test_dlpack',
    # 'torch_native_ci/test_dynamic_shapes',
    # 'torch_native_ci/test_expanded_weights',
    # 'torch_native_ci/test_fake_tensor',
    # 'torch_native_ci/test_function_schema',
    # 'torch_native_ci/test_functional_autograd_benchmark',
    # 'torch_native_ci/test_functional_optim',
    # 'torch_native_ci/test_functionalization',
    # 'torch_native_ci/test_futures',
    # 'torch_native_ci/test_fx',
    # 'torch_native_ci/test_fx_backends',
    # 'torch_native_ci/test_fx_experimental',
    # 'torch_native_ci/test_fx_passes',
    # 'torch_native_ci/test_fx_reinplace_pass',
    # 'torch_native_ci/test_hub',
    # 'torch_native_ci/test_import_stats',
    # 'torch_native_ci/test_indexing',
    # 'torch_native_ci/test_itt',
    # 'torch_native_ci/test_license',
    'torch_native_ci/test_linalg',
    # 'torch_native_ci/test_logging',
    # 'torch_native_ci/test_masked',
    # 'torch_native_ci/test_maskedtensor',
    'torch_native_ci/test_meta',
    # 'torch_native_ci/test_mobile_optimizer',
    # 'torch_native_ci/test_model_dump',
    # 'torch_native_ci/test_module_init',
    # 'torch_native_ci/test_monitor',
    # 'torch_native_ci/test_multiprocessing',
    # 'torch_native_ci/test_multiprocessing_spawn',
    # 'torch_native_ci/test_namedtensor',
    # 'torch_native_ci/test_namedtuple_return_api',
    # 'torch_native_ci/test_native_functions',
    # 'torch_native_ci/test_native_mha',
    'torch_native_ci/test_nn',
    # 'torch_native_ci/test_nnapi',
    # 'torch_native_ci/test_numba_integration',
    # 'torch_native_ci/test_numpy_interop',
    # 'torch_native_ci/test_openmp',
    'torch_native_ci/test_ops',
    # 'torch_native_ci/test_optim',
    'torch_native_ci/test_overrides',
    # 'torch_native_ci/test_package',
    # 'torch_native_ci/test_per_overload_api',
    # 'torch_native_ci/test_proxy_tensor',
    # 'torch_native_ci/test_pruning_op',
    # 'torch_native_ci/test_public_bindings',
    # 'torch_native_ci/test_python_dispatch',
    # 'torch_native_ci/test_pytree',
    # 'torch_native_ci/test_quantization',
    # 'torch_native_ci/test_reductions',
    # 'torch_native_ci/test_scatter_gather_ops',
    'torch_native_ci/test_schema_check',
    # 'torch_native_ci/test_segment_reductions',
    # 'torch_native_ci/test_serialization',
    # 'torch_native_ci/test_set_default_mobile_cpu_allocator',
    'torch_native_ci/test_shape_ops',
    # 'torch_native_ci/test_show_pickle',
    # 'torch_native_ci/test_sort_and_select',
    # 'torch_native_ci/test_spectral_ops',
    # 'torch_native_ci/test_stateless',
    # 'torch_native_ci/test_static_runtime',
    # 'torch_native_ci/test_subclass',
    'torch_native_ci/test_tensor_creation_ops',
    # 'torch_native_ci/test_tensorboard',
    # 'torch_native_ci/test_tensorexpr',
    # 'torch_native_ci/test_tensorexpr_pybind',
    # 'torch_native_ci/test_testing',
    # 'torch_native_ci/test_throughput_benchmark',
    'torch_native_ci/test_torch',
    # 'torch_native_ci/test_transformers',
    # 'torch_native_ci/test_type_hints',
    # 'torch_native_ci/test_type_info',
    # 'torch_native_ci/test_type_promotion',
    # 'torch_native_ci/test_typing',
    'torch_native_ci/test_unary_ufuncs',
    # 'torch_native_ci/test_utils',
    'torch_native_ci/test_view_ops',
    # 'torch_native_ci/test_vmap',
    'torch_native_ci/distributed/test_c10d_cncl',
    'torch_native_ci/distributed/test_c10d_spawn_cncl',
    'torch_native_ci/distributed/test_c10d_common',
    'torch_native_ci/distributed/test_c10d_object_collectives',
    'torch_native_ci/distributed/algorithms/test_join',
    'torch_native_ci/distributed/algorithms/ddp_comm_hooks/test_ddp_hooks'
]
TESTS += NATIVE_CI_BLACKLIST

# Case used to generate .xml log, see description inside this file
FAKECASE = 'utils/test_fakecase_print_log'

def run_test(test_module, test_directory, options, *extra_unittest_args):
    executable = get_executable_command(options, allow_pytest=True)
    unittest_args = options.additional_unittest_args.copy()
    if options.verbose:
        unittest_args.append(f'-{"v"*options.verbose}')

    log_base = ''
    # If using pytest, replace -f with equivalent -x
    if options.pytest:
        unittest_args = [arg if arg != '-f' else '-x' for arg in unittest_args]
        # Note: native ci cases produce too many pytest marker warnings, suppress warnings
        if 'torch_native_ci' in test_module:
            unittest_args += ['--disable-warnings']
        if options.result_dir != '' and os.path.isdir(options.result_dir):
            log_base = os.path.join(options.result_dir,
                                    test_module.replace('/', '_'))
            unittest_args += [f'--junitxml={log_base}.xml']
    # Can't call `python -m unittest test_*` here because it doesn't run code
    # in `if __name__ == '__main__': `. So call `python test_*.py` instead.
    argv = [test_module + '.py'] + unittest_args + list(extra_unittest_args)

    command = executable + argv
    run_env = os.environ.copy()
    # enable fallback to cpu for native ci cases
    if 'torch_native_ci' in test_module:
        run_env['PYTORCH_TESTING_DEVICE_ONLY_FOR'] = 'mlu'
        run_env['ENABLE_FALLBACK_TO_CPU'] = '1'
    else:
        run_env['ENABLE_FALLBACK_TO_CPU'] = '0'
    if options.large:
        run_env['TEST_LARGETENSOR'] = 'TRUE'
    if log_base:
        ret_code = shell(command, test_directory, run_env, log_base + '.log')
        if not os.path.exists(log_base + '.xml'):
            run_env['FAILED_TEST_MODULE'] = test_module
            run_env['FAILED_LOG_FILE'] = log_base + '.log'
            shell(executable + [FAKECASE + '.py', f'--junitxml={log_base}_fakecase.xml'],
                  test_directory, run_env)
        return ret_code
    else:
        return shell(command, test_directory, run_env)

def get_backend_type(test_module):
    if 'cnnl' in test_module:
        return 'cnnl'
    else:
        raise RuntimeError("unsupported backend type, currently only support cnnl.")

def test_executable_file(test_module, test_directory, options):
    gtest_dir = os.path.join(test_directory,'../build/bin')
    if 'cpp_op_gtest' in test_module:
        gtest_dir = os.path.join(test_directory,'cpp/build/bin')
        gtest_dir = os.path.join(gtest_dir,'op_test')
    elif 'cnnl_gtest' in test_module:
        gtest_dir = os.path.join(gtest_dir,'cnnl')
    else:
        gtest_dir = os.path.join(gtest_dir,'common')

    total_error_info = []
    total_failed_commands = []
    if os.path.exists(gtest_dir):
        commands = (os.path.join(gtest_dir, filename) for filename in os.listdir(gtest_dir))
        for command in commands:
            command = [command]
            log_base = ''
            if options.result_dir != '' and os.path.isdir(options.result_dir):
                log_base = os.path.join(options.result_dir,
                                test_module.replace('/', '_') + '_' + command[0].split('/')[-1])
                command += [f'--gtest_output=xml:{log_base}.xml']
            if log_base:
                executable = get_executable_command(options, allow_pytest=True)
                run_env = os.environ.copy()
                return_code = shell(command, test_directory, run_env, log_base + '.log')
                if not os.path.exists(log_base + '.xml'):
                    run_env['FAILED_TEST_MODULE'] = test_module + '/' + command[0].split('/')[-1]
                    run_env['FAILED_LOG_FILE'] = log_base + '.log'
                    shell(executable + [FAKECASE + '.py', f'--junitxml={log_base}_fakecase.xml'],
                          test_directory, run_env)
            else:
                return_code = shell(command, test_directory)
            if options.failfast:
                gen_err_message(return_code, command[0], total_error_info)
            elif return_code != 0:
                total_failed_commands.append((command[0], return_code))

    # Print total error message
    print("*********** Gtest : Error Message Summaries **************")
    for err_message in total_error_info:
        logging.error("\033[31;1m {}\033[0m .".format(err_message))
    for cmd, ret in total_failed_commands:
        print(f"command {cmd} failed, return code {ret}")
    print("**********************************************************")

    return 1 if total_failed_commands or total_error_info else 0

CUSTOM_HANDLERS = {
    'cnnl_gtest': test_executable_file,
    'common_gtest': test_executable_file,
    'cpp_op_gtest': test_executable_file,
}

def parse_test_module(test):
    return test.split('.')[0]

class TestChoices(list):
    def __init__(self, *args, **kwargs):
        super(TestChoices, self).__init__(args[0])

    def __contains__(self, item):
        return list.__contains__(self, parse_test_module(item))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the PyTorch unit test suite',
        epilog='where TESTS is any of: {}'.format(', '.join(TESTS)))
    parser.add_argument(
        '-v',
        '--verbose',
        action='count',
        default=0,
        help='print verbose information and test-by-test results')
    parser.add_argument(
        '-i',
        '--include',
        nargs='+',
        choices=TestChoices(TESTS),
        default=TESTS,
        metavar='TESTS',
        help='select a set of tests to include (defaults to ALL tests).'
             ' tests can be specified with module name, module.TestClass'
             ' or module.TestClass.test_method')
    parser.add_argument(
        '-x',
        '--exclude',
        nargs='+',
        choices=TESTS,
        metavar='TESTS',
        default=[],
        help='select a set of tests to exclude')
    parser.add_argument(
        '-f',
        '--first',
        choices=TESTS,
        metavar='TESTS',
        help='select the test to start from (excludes previous tests)')
    parser.add_argument(
        '-l',
        '--last',
        choices=TESTS,
        metavar='TESTS',
        help='select the last test to run (excludes following tests)')
    parser.add_argument(
        '--bring-to-front',
        nargs='+',
        choices=TestChoices(TESTS),
        default=[],
        metavar='TESTS',
        help='select a set of tests to run first. This can be used in situations'
             ' where you want to run all tests, but care more about some set, '
             'e.g. after making a change to a specific component')
    parser.add_argument(
        '--ignore_cnnl_blacklist',
        action='store_true',
        help='always ignore blacklisted train tests')
    parser.add_argument(
        '--ignore_native_ci_blacklist',
        action='store_true',
        help='always ignore blacklisted torch native ci train tests')
    parser.add_argument(
        'additional_unittest_args',
        nargs='*',
        help='additional arguments passed through to unittest, e.g., '
             'python run_test.py -i sparse -- TestSparse.test_factory_size_check')
    parser.add_argument(
        '--large',
        action='store_true',
        help='whether to run test cases of large tensor')
    parser.add_argument(
        '--pytest', action='store_true',
        help='If true, use `pytest` to execute the tests.'
    )
    parser.add_argument(
        '--result_dir', default='',
        help='If result_dir is not empty, generate xml results to the specified directory. '
        'For .py files, xml report is generated only if pytest is enabled.'
    )
    parser.add_argument(
        '--failfast', action='store_true',
        help='If true, exits immediately upon the first failing testcase.'
        'Otherwise, all selected tests are executed regardless of failures.'
    )
    return parser.parse_args()


def get_executable_command(options, allow_pytest):
    executable = [sys.executable]
    if options.pytest:
        if allow_pytest:
            executable += ['-m', 'pytest']
        else:
            print_to_stderr('Pytest cannot be used for this test. Falling back to unittest.')
    return executable

def find_test_index(test, selected_tests, find_last_index=False):
    """Find the index of the first or last occurrence of a given test/test module in the list of selected tests.

    This function is used to determine the indices when slicing the list of selected tests when
    ``options.first``(:attr:`find_last_index`=False) and/or ``options.last``(:attr:`find_last_index`=True) are used.

    :attr:`selected_tests` can be a list that contains multiple consequent occurrences of tests
    as part of the same test module, e.g.:

    ```
    selected_tests = ['autograd', 'cuda', **'torch.TestTorch.test_acos',
                     'torch.TestTorch.test_tan', 'torch.TestTorch.test_add'**, 'utils']
    ```

    If :attr:`test`='torch' and :attr:`find_last_index`=False, result should be **2**.
    If :attr:`test`='torch' and :attr:`find_last_index`=True, result should be **4**.

    Arguments:
        test (str): Name of test to lookup
        selected_tests (list): List of tests
        find_last_index (bool, optional): should we lookup the index of first or last
            occurrence (first is default)

    Returns:
        index of the first or last occurance of the given test
    """
    idx = 0
    found_idx = -1
    for t in selected_tests:
        if t.startswith(test):
            found_idx = idx
            if not find_last_index:
                break
        idx += 1
    return found_idx


def exclude_tests(exclude_list, selected_tests, exclude_message=None):
    tests_copy = selected_tests[:]
    for exclude_test in exclude_list:
        for test in tests_copy:
            # Using full match to avoid the problem of
            # similar file names.
            if test == exclude_test:
                if exclude_message is not None:
                    print_to_stderr('Excluding {} {}'.format(test, exclude_message))
                selected_tests.remove(test)
    return selected_tests


def get_selected_tests(options):
    selected_tests = options.include

    if options.bring_to_front:
        to_front = set(options.bring_to_front)
        selected_tests = options.bring_to_front + list(filter(lambda name: name not in to_front,
                                                              selected_tests))

    if options.first:
        first_index = find_test_index(options.first, selected_tests)
        selected_tests = selected_tests[first_index:]

    if options.last:
        last_index = find_test_index(options.last, selected_tests, find_last_index=True)
        selected_tests = selected_tests[:last_index + 1]

    selected_tests = exclude_tests(options.exclude, selected_tests)

    print("=========options.ignore_cnnl_blacklist===========")
    if not options.ignore_cnnl_blacklist:
        selected_tests = exclude_tests(CNNL_BLACKLIST, selected_tests)
    print("=========options.ignore_cnnl_blacklist===========")

    print("=========options.ignore_native_ci_blacklist===========")
    if not options.ignore_native_ci_blacklist:
        selected_tests = exclude_tests(NATIVE_CI_BLACKLIST, selected_tests)
    print("=========options.ignore_native_ci_blacklist===========")

    selected_copy = selected_tests.copy()
    for selected in selected_copy:
        # TODO(fanshijie): test_distributed.py does not support pytest
        if selected == 'distributed/test_distributed' and options.pytest:
            selected_tests = exclude_tests([selected], selected_tests)
            selected_tests += ['distributed/test_distributed_wrapper']
            continue
        if selected not in ['torch_ops/', 'custom_ops/']:
            continue
        selected_tests += select_current_op_portion(selected)
        selected_tests = exclude_tests([selected], selected_tests)

    return selected_tests

"""
    * This function splits testcases in module [torch_ops/, custom_ops/] into
    * CI_PARALLEL_TOTAL number of portions, with currently selected portion
    * index being CI_PARALLEL_INDEX.
    * CI_PARALLEL_TOTAL and CI_PARALLEL_INDEX are env variables set by
    * jenkins pipeline when parallel is used.
    * By default, all testcases are selected.
"""
def select_current_op_portion(module):
    parallel_total = int(os.environ.get('CI_PARALLEL_TOTAL', 1))
    parallel_index = int(os.environ.get('CI_PARALLEL_INDEX', 0))
    op_test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                module)
    all_op_test = []
    for f in os.listdir(op_test_path):
        if f.startswith("test_") and f.endswith(".py"):
            all_op_test += [os.path.join(module, f).split('.')[0]]
    all_op_test = sorted(all_op_test)
    selected_op_test = []
    for index, op_module in enumerate(all_op_test):
        if parallel_index == index % parallel_total:
            selected_op_test.append(op_module)
    return selected_op_test

def main():
    options = parse_args()
    # build cpp gtest
    # TODO(kongweiguang): temporarily sheild this code for testing.
    #if options.ignore_cnnl_blacklist:
    #    subprocess.check_call('cpp/scripts/build_cpp_test.sh')

    test_directory = os.path.dirname(os.path.abspath(__file__))
    selected_tests = get_selected_tests(options)
    total_error_info = []
    return_code_failed = []

    if options.verbose:
        print_to_stderr('Selected tests: {}'.format(', '.join(selected_tests)))

    for test in selected_tests:
        test_module = parse_test_module(test)

        # Printing the date here can help diagnose which tests are slow
        print_to_stderr('Running {} ... [{}]'.format(test, datetime.now()))
        handler = CUSTOM_HANDLERS.get(test, run_test)
        return_code = handler(test_module, test_directory, options)
        if options.failfast:
            assert isinstance(return_code, int) and not isinstance(
                return_code, bool), 'Return code should be an integer'
            gen_err_message(return_code, test, total_error_info)
        else:
            if not isinstance(return_code, int) or return_code != 0:
                return_code_failed.append((test_module, return_code))

    # Print total error message
    print("***************** run_test.py : Error Message Summaries **********************")
    if total_error_info:
        print("***************** Total Error Info **********************")
        for err_message in total_error_info:
            logging.error("\033[31;1m {}\033[0m .".format(err_message))
        assert False

    if return_code_failed:
        print("***************** Return Code Failed **********************")
        for test_module, return_code in return_code_failed:
            print(f'test_module {test_module}, return code {return_code}')

    print("*******************************************************************************")


if __name__ == '__main__':
    main()
