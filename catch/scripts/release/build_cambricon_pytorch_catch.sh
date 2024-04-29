#!/bin/bash
set -e

function checkPython() {
  python_version=$(python --version | awk '{print $2}')
  version_num=$(echo $python_version | sed -e 's/\(^[0-9]\+\)\.\([0-9]\+\)\.\([0-9]\+\)/\1\2/') # for python 3.10.8, use 310 for compare
  bool_condition=$(echo $version_num | awk '{if($1>=36) printf("true\n"); else printf("false\n")}') # since 3.10 < 3.6, use 36 for compare
  if [ "$bool_condition" == "true" ]
  then
    echo -e "\033[32m Python Version $python_version\033[0m"
  else
    echo -e "\033[31m ERROR: Required Python Version >= 3.6\033[0m"
    exit 1
  fi
}

function setEnv() {
  if [ -z "${NEUWARE_HOME}" ]
  then
    echo -e "\033[31m ERROR : NEUWARE_HOME is not set\033[0m"
    exit 1
  fi
  if [ -z "${CATCH_HOME}"  ]
  then
    echo -e "\033[31m ERROR : CATCH_HOME is not set\033[0m"
  exit 1
  fi
  if [ -z "${PYTORCH_HOME}"  ]
  then
    echo -e "\033[31m ERROR : PYTORCH_HOME is not set\033[0m"
    exit 1
  fi
  if [ -z "${VISION_HOME}"  ]
  then
    echo -e "\033[31m ERROR : VISION_HOME is not set\033[0m"
    exit 1
  fi
}

function checkNeuwareVersion() {
  pushd $CATCH_HOME/scripts/release/
    python json_parser.py
    neuware_lib_version=$(awk -F ':' '{if($1 !~ "cntoolkit") print $1, $2, $3}' dependency2.txt | awk '{printf "lib%s.so.%s\n", $1, $3}' | awk -F '-' '{printf "%s\n", $1}')
  popd
  for lib in $neuware_lib_version
  do
    if [ ! -f "$NEUWARE_HOME/lib64/$lib" ]
    then
      echo -e "\033[31m ERROR: No Exit $NEUWARE_HOME/lib64/$lib\033[0m"
      exit 1
    fi
  done
  echo -e "\033[32m CNNL NUEWARE LIB VERSION CHECK PASS\033[0m"
}

function check_bazel() {
  bazel_ver=$(bazel --version | awk '{print $2}')
  if [  "$bazel_ver" == "3.4.1" ]
  then
    echo -e "\033[32m bazel 3.4.1\033[0m"
  elif [ "bazel_ver" == "" ]
  then
    echo -e "\033[31m NO bazel, Please Install bazel 3.4.1\033[0m"
    exit 1
  else
    echo  -e "\033[31m ERROR: bazel 3.4.1 Required.\033[0m"
    exit 1
  fi
}

function apply_patch() {
  # use [command group](https://www.gnu.org/software/bash/manual/html_node/Command-Grouping.html) to avoid exit when enable "set -e".
  bool_patch=$( (grep -rn "MLU" $PYTORCH_HOME/c10/core/DeviceType.h) || bool_patch="")
  if [ "$bool_patch" == "" ]
    then
    pushd $CATCH_HOME
      ./scripts/apply_patches_to_pytorch.sh
    popd
  fi 
  echo -e "\033[32m Apply Patch Success.\033[0m"
}

function build_pytorch() {
  # use [command group](https://www.gnu.org/software/bash/manual/html_node/Command-Grouping.html) to avoid exit when enable "set -e".
  torch_version=$( (pip show torch | grep "Version") || torch_version="")
  if [ "$torch_version" == "" ]
  then
    echo -e "\033[33mInstall torch now...\033[0m"
    pushd $PYTORCH_HOME
      pip install -r requirements.txt
      python setup.py clean; python setup.py install
    popd
  else
    echo -e "\033[32m pytorch Install Success, Version $torch_version\033[0m"
  fi 
}

function build_catch() {
  # use [command group](https://www.gnu.org/software/bash/manual/html_node/Command-Grouping.html) to avoid exit when enable "set -e".
  catch_version=$( (pip show torch_mlu | grep "Version") || catch_version="")
  if [ "$catch_version" == "" ]  
  then
    echo -e "\033[33mInstall torch_mlu now...\033[0m"
    pushd $CATCH_HOME
      pip install -r requirements.txt
      python setup.py clean; python setup.py install    
    popd
  else
    echo -e "\033[32m torch_mlu Install Success, $catch_version\033[0m"
  fi
}

function build_vision() {
  # use [command group](https://www.gnu.org/software/bash/manual/html_node/Command-Grouping.html) to avoid exit when enable "set -e".
  vision_version=$( (pip show torchvision | grep "Version") || vision_version="")
  if [ "$vision_version" == "" ]  
  then
    echo -e "\033[33mInstall torchvision now...\033[0m"
    pushd $VISION_HOME
      sed -i "s/has_ffmpeg = ffmpeg_exe is not None/has_ffmpeg = False/g" setup.py
      python setup.py bdist_wheel
      pip install dist/*.whl
    popd
  else
    echo -e "\033[32m torchvision Install Success, $catch_version\033[0m"
  fi
}

function check_is_installed() {
  torch_version=$( (python -c "import torch; print(torch.__version__)") || torch_version="")
  catch_version=$( (pip list | grep torch-mlu | awk '{print $2}') || catch_version="")
  vision_version=$( (python -c "import torchvision; print(torchvision.__version__)") || vision_version="")
  if [ "$torch_version" != "" ]
  then
    echo -e "\033[32m Install Pytorch Success.\033[0m"
  else
    echo -e "\033[31m Install Pytorch Failed.\033[0m"
  fi
  if [ "$catch_version" != "" ]
  then
    echo -e "\033[32m Install Catch Success.\033[0m"
  else
    echo -e "\033[31m Install Catch Failed.\033[0m"
  fi
  if [  "$vision_version" != "" ]
  then
    echo -e "\033[32m Install torchvision Success.\033[0m"
  else
    echo -e "\033[31m Install torchvision Failed.\033[0m"
  fi
}


checkPython
setEnv
checkNeuwareVersion
check_bazel
apply_patch
build_pytorch
build_catch
build_vision
check_is_installed


