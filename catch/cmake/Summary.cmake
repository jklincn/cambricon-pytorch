# Prints accumulated TorchMLU configuration summary
function(torch_mlu_print_configuration_summary)
  message(STATUS "")
  message(STATUS "******** Summary ********")
  message(STATUS "General:")
  message(STATUS "  CMake version         : ${CMAKE_VERSION}")
  message(STATUS "  CMake command         : ${CMAKE_COMMAND}")
  message(STATUS "  System                : ${CMAKE_SYSTEM_NAME}")
  message(STATUS "  C++ compiler          : ${CMAKE_CXX_COMPILER}")
  message(STATUS "  C++ compiler id       : ${CMAKE_CXX_COMPILER_ID}")
  message(STATUS "  C++ compiler version  : ${CMAKE_CXX_COMPILER_VERSION}")
  message(STATUS "  CXX flags             : ${CMAKE_CXX_FLAGS}")
  message(STATUS "  PYTORCH_SOURCE_PATH   : ${PYTORCH_SOURCE_PATH}")
  get_directory_property(tmp DIRECTORY ${PROJECT_SOURCE_DIR} COMPILE_DEFINITIONS)
  message(STATUS "  CMAKE_PREFIX_PATH     : ${CMAKE_PREFIX_PATH}")
  message(STATUS "  CMAKE_INSTALL_PREFIX  : ${CMAKE_INSTALL_PREFIX}")
  message(STATUS "")

  message(STATUS "  USE_PYTHON            : ${USE_PYTHON}")
  message(STATUS "  USE_CNCL              : ${USE_CNCL}")
  message(STATUS "  USE_BANG              : ${USE_BANG}")
  message(STATUS "  USE_MLUOP             : ${USE_MLUOP}")
  message(STATUS "  USE_PROFILE           : ${USE_PROFILE}")
  message(STATUS "  BUILD_TEST            : ${BUILD_TEST}")
  message(STATUS "")

  SET(NEUWARE_HOME_PATH $ENV{NEUWARE_HOME})
  message(STATUS "  NEUWARE_HOME          : ${NEUWARE_HOME_PATH}")
  if (CNNL_FOUND)
    message(STATUS "  CNNL_LIBRARIES        : ${CNNL_LIBRARIES}")
  endif()
  if (CNRT_FOUND)
    message(STATUS "  CNRT_LIBRARIES        : ${CNRT_LIBRARIES}")
  endif()
  if (CNCL_FOUND)
    message(STATUS "  CNCL_LIBRARIES        : ${CNCL_LIBRARIES}")
  endif()

  if (CNPAPI_FOUND)
    message(STATUS "  CNPAPI_LIBRARIES      : ${CNPAPI_LIBRARIES}")
  endif()

  if (MLUOP_FOUND)
    message(STATUS "  MLUOP_LIBRARIES       : ${MLUOP_LIBRARIES}")
  endif()

  message(STATUS "")
  message(STATUS "*************************")
endfunction()
