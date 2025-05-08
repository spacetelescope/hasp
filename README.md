
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
    -i INDIR, --input_directory INDIR

        The name of the directory containing the individual exposures to
        be coadded

    -o OUTDIR, --output_directory OUTDIR

        The name of the directory that will contain the coadded spectral products

    -t THRESHOLD, --threshold THRESHOLD

        Threshold for flux-based filtering.  Optional, default value is -50

    -c, --clobber

        If set, overwrite existing products

    -s SNRMAX, --snrmax SNRMAX
        Maximum SNR per pixel for flux-based filtering

    -k, --no_keyword_filtering
        Disable keyword-based filtering (except for STIS PRISM data, which is always filtered)

    -x, --cross_program
        Create cross-program (HSLA) products only

    -g GRATING_TABLE, --grating_table GRATING_TABLE
        Name of custom grating priority table
```
