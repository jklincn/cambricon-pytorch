#!/bin/bash

find ${PWD}/. -name '.git' | xargs rm -rf; &>/dev/null
find ${PWD}/. -name '.jenkins*' | xargs rm -rf; &>/dev/null

rm -rf catch/docs &>/dev/null
