
# HASP

This repository contains the wrapper script that creates Hubble
Advanced Spectral Products.

The script will create coadded spectral products for each target
in each visit.  In the future, it will be able to create products
for each target in each program.

Installing the package will instll the wrapper script: swrapper.

To run:

    swrapper -i . -o products -t threshold

```
    -i --input_directory

        The name of the directory containing the individual exposures to
        be coadded

    -o --output_directory

        The name of the directory containing the coadded spectral products

    -t --threshold

        Threshold for flux-based filtering.  Optional, default value is -50
```
