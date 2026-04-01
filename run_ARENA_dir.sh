#!/bin/bash

for file in $1/*.pdb; do    
    ./Arena/Arena "$file" "$file" 5 
done