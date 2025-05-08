
# HASP

This repository contains the wrapper script that creates Hubble
Advanced Spectral Products and Hubble Spectral Legacy Archive products.

The script will create coadded spectral products for each target
in each visit (visit-level products) and each program (program-level products).
When used with the -x option, it will create products for all exposures in the
input directory, assuming they are all the same target (cross-program, or HSLA products).

Products include single grating products made from all exposures that use the same
grating, and abutted products that stitch together the single grating products
according to priorities and wavelength ranges set in the grating priorities table.

Installing the package will install the wrapper script: swrapper.

To run:

    swrapper -i . -o products

```
    -i INDIR, --input_directory INDIR

        The name of the directory containing the individual exposures to
        be coadded

    -o OUTDIR, --output_directory OUTDIR

        The name of the directory that will contain the coadded spectral products

    -t THRESHOLD, --threshold THRESHOLD

        Threshold for flux-based filtering.  Optional, default value is -50

    -c, --clobber

        If set, overwrite existing products.  Optional, if this keyword is not set,
        products will not be overwritten.

    -s SNRMAX, --snrmax SNRMAX
        Maximum SNR per pixel for flux-based filtering.  Optional, default value is 20.0

    -k, --no_keyword_filtering
        Disable keyword-based filtering (except for STIS PRISM data, which is always filtered)

    -x, --cross_program
        Create cross-program (HSLA) products only

    -g GRATING_TABLE, --grating_table GRATING_TABLE
        Name of custom grating priority table.  If a custom grating priority table is not
        selected, the default priorities will be set by the tables in hasp/grating_priority.py.
        Users can use the hasp/grating_priority_table.json as a template for creating their own
        custom grating priority table.
```
