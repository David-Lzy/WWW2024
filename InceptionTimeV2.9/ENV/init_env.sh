#!/bin/bash

# Define the environment name
environment_name="InceptionTime_torch"

# Check if the environment already exists
if conda env list | grep -q "$environment_name"; then
    echo "Conda environment $environment_name already exists."
else

    # Create a new environment using conda and install dependencies from freeze.yml
    conda env create -f ./freeze.yml
    echo "Conda environment $environment_name successfully created."
fi

conda activate "$environment_name"

# # Deactivate the environment
# conda deactivate
