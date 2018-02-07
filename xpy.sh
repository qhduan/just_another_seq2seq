#!/bin/bash

#pylint **/*.py
find . -iname "*.py" -not -path "./front/*" | xargs pylint 
