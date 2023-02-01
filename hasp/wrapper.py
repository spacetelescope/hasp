import pandas as pd
from collections import defaultdict
import argparse
import os
import glob
import re
import datetime
from datetime import datetime as dt
import numpy as np

import astropy
from astropy.io import fits
from astropy.time import Time

from ullyses.coadd import COSSegmentList, STISSegmentList, FUSESegmentList, CCDSegmentList
from ullyses.coadd import abut, SegmentList, Segment

CAL_VER = 0.1
VERSION = 'dr1'
BAD_SEGMENTS = ['NUVC']
STIS_NON_CCD_DETECTORS = ['FUV-MAMA', 'NUV-MAMA']

'''
This wrapper goes through each target folder in the ullyses data directory and find
the data and which gratings are present. This info is then fed into coadd.py.
'''

class HASP_SegmentList(SegmentList):
    """This class is a mixin to add the project-specific write function and functions
    to populate the target name and coordinates

    """

    def import_data(self, file_list):
        """This is used if the __init__ function is called with instrument
        and grating set to None.  This separates the selection of which files
        to work on from the constructor

        """

        alldata = []
        allhdr0 = []
        allhdr1 = []
        self.datasets = []

        for datafile in file_list:
            print('Processing file {}'.format(datafile))
            with fits.open(datafile) as f1:
                prihdr = f1[0].header
                hdr1 = f1[1].header
                data = f1[1].data
                alldata.append(data)
                allhdr0.append(prihdr)
                allhdr1.append(hdr1)

            if len(data) > 0:
                self.instrument = prihdr['INSTRUME'].upper()
                self.grating = prihdr['OPT_ELEM'].upper()
                self.propid = prihdr['PROPOSID']
                self.datasets.append(datafile)
                self.rootname = prihdr['ROOTNAME']
                self.detector = prihdr['DETECTOR']
                target = prihdr['TARGNAME'].strip()
                if target not in self.targnames:
                    self.targnames.append(target)
                try:
                    if prihdr['HLSP_LVL'] == 0:
                        self.level0 = True
                except:
                    pass
            else:
                print('{} has no data'.format(datafile))

        self.members = []
        self.primary_headers = []
        self.first_headers = []

        if len(alldata) > 0:
            for i in range(len(alldata)):
                data = alldata[i]
                hdr0 = allhdr0[i]
                hdr1 = allhdr1[i]
                if len(data) > 0:
                    self.primary_headers.append(hdr0)
                    self.first_headers.append(hdr1)
                    sdqflags = hdr1['SDQFLAGS']
                    if self.instrument == "STIS" and (sdqflags&16) == 16 and \
                            hdr0['DETECTOR'] in STIS_NON_CCD_DETECTORS:
                        sdqflags -= 16
                    exptime = hdr1['EXPTIME']
                    for row in data:
                        if self.instrument == 'COS' and row['SEGMENT'] in BAD_SEGMENTS:
                            continue

                        segment = Segment()
                        segment.data = row
                        segment.sdqflags = sdqflags
                        segment.exptime = exptime
                        if self.instrument == 'COS':
                            cenwave = hdr0['CENWAVE']
                            fppos = hdr0['FPPOS']
                        self.members.append(segment)
        return

    def write(self, filename, overwrite=False, level="", version=""):

        self.target = self.get_targname()
        self.targ_ra, self.targ_dec = self.get_coords()

        # Table 1 - HLSP data

        # set up the header
        hdr1 = fits.Header()
        hdr1['EXTNAME'] = ('SCIENCE', 'Spectrum science arrays')
        hdr1['TIMESYS'] = ('UTC', 'Time system in use')
        hdr1['TIMEUNIT'] = ('s', 'Time unit for durations')
        hdr1['TREFPOS'] = ('GEOCENTER', 'Time reference position')

        mjd_beg = self.combine_keys("expstart", "min")
        mjd_end = self.combine_keys("expend", "max")
        dt_beg = Time(mjd_beg, format="mjd").datetime
        dt_end = Time(mjd_end, format="mjd").datetime
        hdr1['DATE-BEG'] = (dt.strftime(dt_beg, "%Y-%m-%dT%H:%M:%S"), 'Date-time of first observation start')
        hdr1.add_blank('', after='TREFPOS')
        hdr1.add_blank('              / FITS TIME COORDINATE KEYWORDS', before='DATE-BEG')

        hdr1['DATE-END'] = (dt.strftime(dt_end, "%Y-%m-%dT%H:%M:%S"), 'Date-time of last observation end')
        hdr1['MJD-BEG'] = (mjd_beg, 'MJD of first exposure start')
        hdr1['MJD-END'] = (mjd_end, 'MJD of last exposure end')
        hdr1['XPOSURE'] = (self.combine_keys("exptime", "sum"), '[s] Sum of exposure durations')

        # set up the table columns
        nelements = len(self.output_wavelength)
        rpt = str(nelements)

        # Table with co-added spectrum
        cw = fits.Column(name='WAVELENGTH', format=rpt+'E', unit="Angstrom")
        cf = fits.Column(name='FLUX', format=rpt+'E', unit="erg /s /cm**2 /Angstrom")
        ce = fits.Column(name='ERROR', format=rpt+'E', unit="erg /s /cm**2 /Angstrom")
        cs = fits.Column(name='SNR', format=rpt+'E')
        ct = fits.Column(name='EFF_EXPTIME', format=rpt+'E', unit="s")
        cd = fits.ColDefs([cw, cf, ce, cs, ct])
        table1 = fits.BinTableHDU.from_columns(cd, nrows=1, header=hdr1)

        # populate the table
        table1.data['WAVELENGTH'] = self.output_wavelength.copy()
        table1.data['FLUX'] = self.output_flux.copy()
        table1.data['ERROR'] = self.output_errors.copy()
        table1.data['SNR'] = self.signal_to_noise.copy()
        table1.data['EFF_EXPTIME'] = self.output_exptime.copy()
        # HLSP primary header
        hdr0 = fits.Header()
        hdr0['EXTEND'] = ('T', 'FITS file may contain extensions')
        hdr0['NEXTEND'] = 3
        hdr0['FITS_VER'] = 'Definition of the Flexible Image Transport System (FITS) v4.0 https://fits.gsfc.nasa.gov/standard40/fits_standard40aa-le.pdf'
        hdr0['FITS_SW'] = ('astropy.io.fits v' + astropy.__version__, 'FITS file creation software')
        hdr0['ORIGIN'] = ('Space Telescope Science Institute', 'FITS file originator')
        hdr0['DATE'] = (str(datetime.date.today()), 'Date this file was written')
        hdr0['FILENAME'] = (os.path.basename(filename), 'Name of this file')
        hdr0['TELESCOP'] = (self.combine_keys("telescop", "multi"), 'Telescope used to acquire data')
        hdr0['INSTRUME'] = (self.combine_keys("instrume", "multi"), 'Instrument used to acquire data')
        hdr0.add_blank('', after='TELESCOP')
        hdr0.add_blank('              / SCIENCE INSTRUMENT CONFIGURATION', before='INSTRUME')
        hdr0['DETECTOR'] = (self.combine_keys("detector", "multi"), 'Detector or channel used to acquire data')
        hdr0['DISPERSR'] = (self.combine_keys("opt_elem", "multi"), 'Identifier of disperser')
        hdr0['CENWAVE'] = (self.combine_keys("cenwave", "multi"), 'Central wavelength setting for disperser')
        hdr0['APERTURE'] = (self.combine_keys("aperture", "multi"), 'Identifier of entrance aperture')
        hdr0['S_REGION'] = (self.obs_footprint(), 'Region footprint')
        hdr0['OBSMODE'] = (self.combine_keys("obsmode", "multi"), 'Instrument operating mode (ACCUM | TIME-TAG)')
        hdr0['TARGNAME'] = self.target
        hdr0.add_blank(after='OBSMODE')
        hdr0.add_blank('              / TARGET INFORMATION', before='TARGNAME')

        hdr0['RADESYS'] = ('ICRS ','World coordinate reference frame')
        hdr0['TARG_RA'] =  (self.targ_ra,  '[deg] Target right ascension')
        hdr0['TARG_DEC'] =  (self.targ_dec,  '[deg] Target declination')
        hdr0['PROPOSID'] = (self.combine_keys("proposid", "multi"), 'Program identifier')
        hdr0.add_blank(after='TARG_DEC')
        hdr0.add_blank('           / PROVENANCE INFORMATION', before='PROPOSID')
        hdr0['CAL_VER'] = (f'HSLA Cal {CAL_VER}', 'HLSP processing software version')
        hdr0['HLSPID'] = ('HSLA', 'Name ID of this HLSP collection')
        hdr0['HSLPNAME'] = ('Hubble Spectroscopic Legacy Archive',
                        'Name ID of this HLSP collection')
        hdr0['HLSPLEAD'] = ('John Debes', 'Full name of HLSP project lead')
        hdr0['HLSP_VER'] = (version,'HLSP data release version identifier')
        hdr0['HLSP_LVL'] = (level, 'HSLA HLSP Level')
        hdr0['LICENSE'] = ('CC BY 4.0', 'License for use of these data')
        hdr0['LICENURL'] = ('https://creativecommons.org/licenses/by/4.0/', 'Data license URL')
        hdr0['REFERENC'] = ('https://ui.adsabs.harvard.edu/abs/2020RNAAS...4..205R', 'Bibliographic ID of primary paper')

        hdr0['CENTRWV'] = (self.combine_keys("centrwv", "average"), 'Central wavelength of the data')
        hdr0.add_blank(after='REFERENC')
        hdr0.add_blank('           / ARCHIVE SEARCH KEYWORDS', before='CENTRWV')
        hdr0['MINWAVE'] = (self.combine_keys("minwave", "min"), 'Minimum wavelength in spectrum')
        hdr0['MAXWAVE'] = (self.combine_keys("maxwave", "max"), 'Maximum wavelength in spectrum')

        primary = fits.PrimaryHDU(header=hdr0)

        # Table 2 - individual product info

        # first set up header
        hdr2 = fits.Header()
        hdr2['EXTNAME'] = ('PROVENANCE', 'Metadata for contributing observations')
        # set up the table columns
        cfn = fits.Column(name='FILENAME', array=self.combine_keys("filename", "arr"), format='A64')
        cpid = fits.Column(name='PROPOSID', array=self.combine_keys("proposid", "arr"), format='A32')
        ctel = fits.Column(name='TELESCOPE', array=self.combine_keys("telescop", "arr"), format='A32')
        cins = fits.Column(name='INSTRUMENT', array=self.combine_keys("instrume", "arr"), format='A32')
        cdet = fits.Column(name='DETECTOR', array=self.combine_keys("detector", "arr"), format='A32')
        cdis = fits.Column(name='DISPERSER', array=self.combine_keys("opt_elem", "arr"), format='A32')
        ccen = fits.Column(name='CENWAVE', array=self.combine_keys("cenwave", "arr"), format='A32')
        cap = fits.Column(name='APERTURE', array=self.combine_keys("aperture", "arr"), format='A32')
        csr = fits.Column(name='SPECRES', array=self.combine_keys("specres", "arr"), format='F8.1')
        ccv = fits.Column(name='CAL_VER', array=self.combine_keys("cal_ver", "arr"), format='A32')
        mjd_begs = self.combine_keys("expstart", "arr")
        mjd_ends = self.combine_keys("expend", "arr")
        mjd_mids = (mjd_ends + mjd_begs) / 2.
        cdb = fits.Column(name='MJD_BEG', array=mjd_begs, format='F15.9', unit='d')
        cdm = fits.Column(name='MJD_MID', array=mjd_mids, format='F15.9', unit='d')
        cde = fits.Column(name='MJD_END', array=mjd_ends, format='F15.9', unit='d')
        cexp = fits.Column(name='XPOSURE', array=self.combine_keys("exptime", "arr"), format='F15.9', unit='s')
        cmin = fits.Column(name='MINWAVE', array=self.combine_keys("minwave", "arr"), format='F9.4', unit='Angstrom')
        cmax = fits.Column(name='MAXWAVE', array=self.combine_keys("maxwave", "arr"), format='F9.4', unit='Angstrom')

        cd2 = fits.ColDefs([cfn, cpid, ctel, cins, cdet, cdis, ccen, cap, csr, ccv, cdb, cdm, cde, cexp, cmin ,cmax])

        table2 = fits.BinTableHDU.from_columns(cd2, header=hdr2)

        # the HDUList:
        # 0 - empty data - 0th ext header
        # 1 - HLSP data - 1st ext header
        # 2 - individual product info - 2nd ext header

        hdul = fits.HDUList([primary, table1, table2])

        hdul.writeto(filename, overwrite=overwrite)

    def obs_footprint(self):
        # Not using WCS at the moment
        # This is a placeholder, need to figure out polygon
#        apertures = list(set([h["aperture"] for h in self.primary_headers]))
#        ras = list(set([h["ra_targ"] for h in self.primary_headers]))
#        ra_diff = max(np.abs(ras)) - min(np.abs(ras))
#        decs = list(set([h["dec_targ"] for h in self.primary_headers]))
#        dec_diff = max(np.abs(decs)) - min(np.abs(decs))
#        center_ra = np.average(ras)
#        center_dec = np.average(decs)
#        extent_ra = (2.5 / 2 / 3600) + ra_diff
#        extent_dec = (2.5 / 2 / 3600) + dec_diff
#        radius = max([extent_ra, extent_dec])
        self.targ_ra, self.targ_dec = self.get_coords()
        radius = (2.5 / 2 / 3600)
        center_ra = self.targ_ra
        center_dec = self.targ_dec

        s_region = f"CIRCLE {center_ra} {center_dec} {radius}"
        return s_region

    def combine_keys(self, key, method):
        keymap= {"HST": {"expstart": ("expstart", 1),
                         "expend": ("expend", 1),
                         "exptime": ("exptime", 1),
                         "telescop": ("telescop", 0),
                         "instrume": ("instrume", 0),
                         "detector": ("detector", 0),
                         "opt_elem": ("opt_elem", 0),
                         "cenwave": ("cenwave", 0),
                         "aperture": ("aperture", 0),
                         "obsmode": ("obsmode", 0),
                         "proposid": ("proposid", 0),
                         "centrwv": ("centrwv", 0),
                         "minwave": ("minwave", 0),
                         "maxwave": ("maxwave", 0),
                         "filename": ("filename", 0),
                         "specres": ("specres", 0),
                         "cal_ver": ("cal_ver", 0)},
                "FUSE": {"expstart": ("obsstart", 0),
                         "expend": ("obsend", 0),
                         "exptime": ("obstime", 0),
                         "telescop": ("telescop", 0),
                         "instrume": ("instrume", 0),
                         "detector": ("detector", 0),
                         "opt_elem": ("detector", 0),
                         "cenwave": ("centrwv", 0),
                         "aperture": ("aperture", 0),
                         "obsmode": ("instmode", 0),
                         "proposid": ("prgrm_id", 0),
                         "centrwv": ("centrwv", 0),
                         "minwave": ("wavemin", 0),
                         "maxwave": ("wavemax", 0),
                         "filename": ("filename", 0),
                         "specres": ("spec_rp", 1),
                         "cal_ver": ("cf_vers", 0)}}

        vals = []
        for i in range(len(self.primary_headers)):
            tel = self.primary_headers[i]["telescop"]
            actual_key = keymap[tel][key][0]
            hdrno = keymap[tel][key][1]
            if hdrno == 0:
                val = self.primary_headers[i][actual_key]
            else:
                val = self.first_headers[i][actual_key]
            vals.append(val)

        # Allowable methods are min, max, average, sum, multi, arr
        if method == "multi":
            keys_set = list(set(vals))
            if len(keys_set) > 1:
                return "MULTI"
            else:
                return keys_set[0]
        elif method == "min":
            return min(vals)
        elif method == "max":
            return max(vals)
        elif method == "average":
            return np.average(vals)
        elif method == "sum":
            return np.sum(vals)
        elif method == "arr":
            return np.array(vals)

class HASP_COSSegmentList(COSSegmentList, HASP_SegmentList):
    pass


class HASP_STISSegmentList(STISSegmentList, HASP_SegmentList):
    pass


class HASP_CCDSegmentList(CCDSegmentList, HASP_SegmentList):
    pass


def main(indir, outdir, version=VERSION, clobber=False):
    outdir_inplace = False
    if outdir is None:
        HLSP_DIR = os.getenv('HLSP_DIR')
        if HLSP_DIR is None:
            print("Environment variable HLSP_DIR must be defined if outdir is not specified")
            raise RuntimeError("Please set HLSP_DIR and restart")
        outdir_inplace = True

    # Find out which unique modes are present
    # For single visit products, a unique mode is
    # (target, instrument, grating, detector, visit)
    # For single program products, the mode is
    # (target, instrument, grating, detector)
    uniqmodes = []
    uniqvisitmodes = []
    uniqproposalmodes = []
    targetlist = []
    visitlist = []
    proposallist = []
    visitdict = {}
    proposaldict = {}
    spec1d = glob.glob(os.path.join(indir, '*_x1d.fits')) + glob.glob(os.path.join(indir, '*_sx1.fits'))
    spec1d.sort()
#
# Create the list of modes
    print('Creating list of unique modes from these files:')
    for myfile in spec1d:
        f1 = fits.open(myfile)
        prihdr = f1[0].header
        instrument = prihdr['INSTRUME']
        grating = prihdr['OPT_ELEM']
        detector = prihdr['DETECTOR']
        targname = prihdr['TARGNAME']
        visit = myfile[-14:-12]
        proposal = prihdr['PROPOSID']
        visit = (proposal, visit)
        obsmode = (instrument, grating, detector)
        visitmode = (instrument, grating, detector, targname, visit)
        proposalmode = (instrument, grating, detector, targname, proposal)
        if targname not in targetlist:
            targetlist.append(targname)
        if obsmode not in uniqmodes:
            uniqmodes.append(obsmode)
        if visitmode not in uniqvisitmodes:
            uniqvisitmodes.append(visitmode)
            visitdict[visitmode] = [myfile]
        else:
            visitdict[visitmode].append(myfile)
        if proposalmode not in uniqproposalmodes:
            uniqproposalmodes.append(proposalmode)
            proposaldict[proposalmode] = [myfile]
        else:
            proposaldict[proposalmode].append(myfile)
        if visit not in visitlist:
            visitlist.append(visit)
        if proposal not in proposallist:
            proposallist.append(proposal)
        print(myfile, targname, instrument, detector, grating, proposal, visit)
        f1.close()

    visitlist.sort()
    proposallist.sort()

    print('Looping over visits')
    producttype = 'visit'
    for visit in visitlist:
        print('Processing product {}'.format(visit))
        thisvisitkeys = []
        for visitspec in visitdict.keys():
            if visitspec[4] == visit:
                thisvisitkeys.append(visitspec)
        targetsinthisvisit = []
        for visitspec in thisvisitkeys:
            thistarget = visitspec[3]
            if thistarget not in targetsinthisvisit:
                targetsinthisvisit.append(thistarget)
        print('Targets in visit {}: {}'.format(visit, targetsinthisvisit))

        for target in targetsinthisvisit:
            print('Processing target {} in visit {}'.format(target, visit))
            thisvisitandtargetspec = []
            for visitspec in thisvisitkeys:
                thistarget = visitspec[3]
                if target == thistarget:
                    thisvisitandtargetspec.append(visitspec)

            # Create dictionary of all products, with each set to None by default
            products = defaultdict(lambda: None)

            level = 2
            uniqmodes = []
            for uniqmode in thisvisitandtargetspec:
                instrument = uniqmode[0]
                grating = uniqmode[1]
                detector = uniqmode[2]
                if (instrument, grating, detector) not in uniqmodes:
                    uniqmodes.append((instrument, grating, detector))
                files_to_import = visitdict[uniqmode]
                print('Importing files {}'.format(files_to_import))
                # this instantiates the class
                if instrument == 'COS':
                    prod = HASP_COSSegmentList(None, path=indir)
                elif instrument == 'STIS':
                    if detector == 'CCD':
                        prod = HASP_CCDSegmentList(None, path=indir)
                    else:
                        prod = HASP_STISSegmentList(None, path=indir)
                else:
                    print(f'Unknown mode [{instrument}, {grating}, {detector}]')
                    continue
                if prod is not None:
                    prod.import_data(files_to_import)
                prod.target = prod.get_targname()
                prod.targ_ra, prod.targ_dec = prod.get_coords()

                # these two calls perform the main functions
                if len(prod.members) > 0:
                    prod.create_output_wavelength_grid()
                    prod.coadd()
                    # this writes the output file
                    # If making HLSPs for a DR, put them in the official folder
                    prod.target = prod.get_targname()
                    prod.targ_ra, prod.targ_dec = prod.get_coords()
                    target = prod.target.lower()
                    if "." in target:
                        dir_target = rename_target(target)
                    else:
                        dir_target = target
                    if outdir_inplace is True:
                        outdir = os.path.join(HLSP_DIR, dir_target, version)
                    if not os.path.exists(outdir):
                        os.makedirs(outdir)
                    outname = create_output_file_name(prod, producttype, version, level=level)
                    outname = os.path.join(outdir, outname)
                    prod.write(outname, clobber, level=level, version=version)
                    print(f"   Wrote {outname}")
                    products[f'{instrument}/{grating}'] = prod
                else:
                    print(f"No valid data for grating {grating}")
                if prod.level0 is True:
                    print('In group block with prod.level0 is True')
                    prod.create_output_wavelength_grid()
                    prod.coadd()
                    # this writes the output file
                    # If making HLSPs for a DR, put them in the official folder
                    target = prod.target.lower()
                    if "." in target:
                        dir_target = rename_target(target)
                    else:
                        dir_target = target
                    if outdir_inplace is True:
                        outdir = os.path.join(HLSP_DIR, dir_target, version)
                    if not os.path.exists(outdir):
                        os.makedirs(outdir)
                    outname = create_output_file_name(prod, producttype, version, level=0)
                    outname = os.path.join(outdir, outname)
                    prod.write(outname, clobber, level=0, version=version)
                    print(f"   Wrote {outname}")
                    products[f'{instrument}/{grating}'] = prod
                products[f'{instrument}/{grating}'] = prod

            # Create level 3 products- abutted spectra for gratings of the same
            # resolution for each instrument.
            level = 3
            lvl3_modes = {"cos_fuv_m": ["COS/G130M", "COS/G160M", "COS/G185M"],
                            "stis_m": ["STIS/E140M", "STIS/E230M"],
                            "stis_h": ["STIS/E140H", "STIS/E230H"],
                            "stis_l": ["STIS/G140L", "STIS/G230L", "STIS/G230LB", "STIS/G430L", "STIS/G750L"]}
            for outprod,modes in lvl3_modes.items():
                abutted = None
                dowrite = False
                for mode in modes:
                    if products[mode] is not None:
                        if abutted is None:
                            abutted = products[mode]
                        else:
                            abutted = abut(abutted, products[mode])
                            abutted.propid = products[mode].propid
                            abutted.rootname = products[mode].rootname
                            dowrite = True
                if dowrite is True:
                    filename = create_output_file_name(abutted, producttype, version, level=level)
                    filename = os.path.join(outdir, filename)
                    abutted.write(filename, clobber, level=level, version=version)
                    print(f"   Wrote {filename}")

            # Determine which gratings should contribute to the final level 4 SED HLSP.
            # Starting with the bluest product and working redward, find which products,
            # if any, overlap with the bluer product. If more than one overlaps, use
            # the one that extends further. If none overlap, still abut them- there
            # will just be a region of flux=0 in between.
            level = 4
            gratings = []
            minwls = []
            maxwls = []
            ins = []
            for instrument, grating, detector in uniqmodes:
                ins.append(instrument)
                gratings.append(grating)
                minwls.append(products[instrument+"/"+grating].first_good_wavelength)
                maxwls.append(products[instrument+"/"+grating].last_good_wavelength)
            # Only go through this exercise if there is data for more than one instrument
            if len(set(ins)) != 1:
                df = pd.DataFrame({"gratings": gratings, "ins": ins, "minwls": minwls, "maxwls": maxwls})
                used = pd.DataFrame()
                # Start with the bluest product, and remove rows from the dataframe
                # until no rows remain. The only exception is if the bluest product is
                # STIS/echelle *and* G130M+G160M combo exists. Then use G130M+G160M as bluest
                # and ignore STIS/echelle
                lowind = df["minwls"].idxmin()
                if df.loc[lowind, "gratings"] in ["E140M", "E140H"]:
                    if "G130M" in gratings and "G160M" in gratings:
                        g130mind = df.loc[df["gratings"] == "G130M"].index.values
                        used = used.append(df.loc[g130mind])
                        shortestwl = df.loc[g130mind[0], "minwls"]
                        df = df.drop(index=g130mind)
                        g160mind = df.loc[df["gratings"] == "G160M"].index.values
                        used = used.append(df.loc[g160mind])
                        maxwl = df.loc[g160mind[0], "maxwls"]
                        df = df.drop(index=g160mind)
                        df = df.drop(index=lowind)
                    else:
                        shortestwl = df.loc[lowind, "minwls"]
                        used = used.append(df.loc[lowind])
                        maxwl = df.loc[lowind, "maxwls"]
                        df = df.drop(lowind)
                else:
                    shortestwl = df.loc[lowind, "minwls"]
                    used = used.append(df.loc[lowind])
                    maxwl = df.loc[lowind, "maxwls"]
                    df = df.drop(lowind)
                while len(df) > 0:
                    lowind = df.loc[(df["minwls"] < maxwl) & (df["maxwls"] > maxwl)].index.values
                    # If G130M and G160M both exist for a given target, *always*
                    # abut them together regardless of other available gratings.
                    # This captures the case where there is FUSE bluer than COS/FUV.
                    if "G130M" in used.gratings.values and "G160M" in gratings and "G160M" not in used.gratings.values:
                        lowind = df.loc[df["gratings"] == "G160M"].index.values
                        maxwl = df.loc[lowind[0], "maxwls"]
                        used = used.append(df.loc[lowind])
                        df = df.drop(index=lowind)
                    # Handle case where more than one grating overlaps with bluer data.
                    elif len(lowind) > 1:
                        df2 = df.loc[lowind]
                        ranges = df2.maxwls - df2.minwls
                        biggest = ranges.idxmax()
                        match_grating = df2.loc[biggest, "gratings"]
                        match_ind = df.loc[df["gratings"] == match_grating].index.values
                        used = used.append(df.loc[match_ind])
                        maxwl = df.loc[match_ind, "maxwls"].values[0]
                        df = df.drop(index=lowind)
                    # If none overlap, abut with the next closest product.
                    elif len(lowind) == 0:
                        lowind = df["minwls"].idxmin()
                        used = used.append(df.loc[lowind])
                        maxwl = df.loc[lowind, "maxwls"]
                        df = df.drop(lowind)
                    # This is the easy case- only one mode overlaps with the bluer data.
                    else:
                        maxwl = df.loc[lowind[0], "maxwls"]
                        used = used.append(df.loc[lowind])
                        df = df.drop(index=lowind)
                    # Check every time if there are any modes that overlap completely
                    # with what has been abutted so far.
                    badinds = df.loc[(df["minwls"] > shortestwl) & (df["maxwls"] < maxwl)].index.values
                    if len(badinds) > 0:
                        df = df.drop(index=badinds)
                # If more than one instrument was selected for abutting,
                # create level 4 product.
                if len(set(used["ins"].values)) > 1:
                    abut_gr = used.iloc[0]["ins"] + "/" + used.iloc[0]["gratings"]
                    abutted = products[abut_gr]
                    for i in range(1, len(used)):
                        abut_gr = used.iloc[i]["ins"] + "/" + used.iloc[i]["gratings"]
                        abutted = abut(abutted, products[abut_gr])
                        abutted.propid = products[abut_gr].propid
                        abutted.rootname = products[abut_gr].rootname
                    filename = create_output_file_name(abutted, producttype, version, level=level)
                    filename = os.path.join(outdir, filename)
                    abutted.write(filename, clobber, level=level, version=version)
                    print(f"   Wrote {filename}")

    print('Looping over proposals')
    producttype = 'proposal'
    for proposal in proposallist:
        print('Processing product {}'.format(proposal))
        thisproposalkeys = []
        for proposalspec in proposaldict.keys():
            if proposalspec[4] == proposal:
                thisproposalkeys.append(proposalspec)
        targetsinthisproposal = []
        for proposalspec in thisproposalkeys:
            thistarget = proposalspec[3]
            if thistarget not in targetsinthisproposal:
                targetsinthisproposal.append(thistarget)
        print('Targets in proposal {}: {}'.format(proposal, targetsinthisproposal))

        for target in targetsinthisproposal:
            print('Processing target {} in proposal {}'.format(target, proposal))
            thisproposalandtargetspec = []
            for proposalspec in thisproposalkeys:
                thistarget = proposalspec[3]
                if target == thistarget:
                    thisproposalandtargetspec.append(proposalspec)
                    
            # Create dictionary of all products, with each set to None by default
            products = defaultdict(lambda: None)

            level = 2
            uniqmodes = []
            for uniqmode in thisproposalandtargetspec:
                instrument = uniqmode[0]
                grating = uniqmode[1]
                detector = uniqmode[2]
                if (instrument, grating, detector) not in uniqmodes:
                    uniqmodes.append((instrument, grating, detector))
                files_to_import = proposaldict[uniqmode]
                print('Importing files {}'.format(files_to_import))
                # this instantiates the class
                if instrument == 'COS':
                    prod = HASP_COSSegmentList(None, path=indir)
                elif instrument == 'STIS':
                    if detector == 'CCD':
                        prod = HASP_CCDSegmentList(None, path=indir)
                    else:
                        prod = HASP_STISSegmentList(None, path=indir)
                else:
                    print(f'Unknown mode [{instrument}, {grating}, {detector}]')
                    continue
                if prod is not None:
                    prod.import_data(files_to_import)
                prod.target = prod.get_targname()
                prod.targ_ra, prod.targ_dec = prod.get_coords()

                # these two calls perform the main functions
                if len(prod.members) > 0:
                    prod.create_output_wavelength_grid()
                    prod.coadd()
                    # this writes the output file
                    # If making HLSPs for a DR, put them in the official folder
                    prod.target = prod.get_targname()
                    prod.targ_ra, prod.targ_dec = prod.get_coords()
                    target = prod.target.lower()
                    if "." in target:
                        dir_target = rename_target(target)
                    else:
                        dir_target = target
                    if outdir_inplace is True:
                        outdir = os.path.join(HLSP_DIR, dir_target, version)
                    if not os.path.exists(outdir):
                        os.makedirs(outdir)
                    outname = create_output_file_name(prod, producttype, version, level=level)
                    outname = os.path.join(outdir, outname)
                    prod.write(outname, clobber, level=level, version=version)
                    print(f"   Wrote {outname}")
                    products[f'{instrument}/{grating}'] = prod
                else:
                    print(f"No valid data for grating {grating}")
                if prod.level0 is True:
                    print('In group block with prod.level0 is True')
                    prod.create_output_wavelength_grid()
                    prod.coadd()
                    # this writes the output file
                    # If making HLSPs for a DR, put them in the official folder
                    target = prod.target.lower()
                    if "." in target:
                        dir_target = rename_target(target)
                    else:
                        dir_target = target
                    if outdir_inplace is True:
                        outdir = os.path.join(HLSP_DIR, dir_target, version)
                    if not os.path.exists(outdir):
                        os.makedirs(outdir)
                    outname = create_output_file_name(prod, producttype, version, level=0)
                    outname = os.path.join(outdir, outname)
                    prod.write(outname, clobber, level=0, version=version)
                    print(f"   Wrote {outname}")
                    products[f'{instrument}/{grating}'] = prod
                products[f'{instrument}/{grating}'] = prod

            # Create level 3 products- abutted spectra for gratings of the same
            # resolution for each instrument.
            level = 3
            lvl3_modes = {"cos_fuv_m": ["COS/G130M", "COS/G160M", "COS/G185M"],
                            "stis_m": ["STIS/E140M", "STIS/E230M"],
                            "stis_h": ["STIS/E140H", "STIS/E230H"],
                            "stis_l": ["STIS/G140L", "STIS/G230L", "STIS/G230LB", "STIS/G430L", "STIS/G750L"]}
            for outprod,modes in lvl3_modes.items():
                abutted = None
                dowrite = False
                for mode in modes:
                    if products[mode] is not None:
                        if abutted is None:
                            abutted = products[mode]
                        else:
                            abutted = abut(abutted, products[mode])
                            abutted.propid = products[mode].propid
                            abutted.rootname = products[mode].rootname
                            dowrite = True
                if dowrite is True:
                    filename = create_output_file_name(abutted, producttype, version, level=level)
                    filename = os.path.join(outdir, filename)
                    abutted.write(filename, clobber, level=level, version=version)
                    print(f"   Wrote {filename}")

            # Determine which gratings should contribute to the final level 4 SED HLSP.
            # Starting with the bluest product and working redward, find which products,
            # if any, overlap with the bluer product. If more than one overlaps, use
            # the one that extends further. If none overlap, still abut them- there
            # will just be a region of flux=0 in between.
            level = 4
            gratings = []
            minwls = []
            maxwls = []
            ins = []
            for instrument, grating, detector in uniqmodes:
                ins.append(instrument)
                gratings.append(grating)
                minwls.append(products[instrument+"/"+grating].first_good_wavelength)
                maxwls.append(products[instrument+"/"+grating].last_good_wavelength)
            # Only go through this exercise if there is data for more than one instrument
            if len(set(ins)) != 1:
                df = pd.DataFrame({"gratings": gratings, "ins": ins, "minwls": minwls, "maxwls": maxwls})
                used = pd.DataFrame()
                # Start with the bluest product, and remove rows from the dataframe
                # until no rows remain. The only exception is if the bluest product is
                # STIS/echelle *and* G130M+G160M combo exists. Then use G130M+G160M as bluest
                # and ignore STIS/echelle
                lowind = df["minwls"].idxmin()
                if df.loc[lowind, "gratings"] in ["E140M", "E140H"]:
                    if "G130M" in gratings and "G160M" in gratings:
                        g130mind = df.loc[df["gratings"] == "G130M"].index.values
                        used = used.append(df.loc[g130mind])
                        shortestwl = df.loc[g130mind[0], "minwls"]
                        df = df.drop(index=g130mind)
                        g160mind = df.loc[df["gratings"] == "G160M"].index.values
                        used = used.append(df.loc[g160mind])
                        maxwl = df.loc[g160mind[0], "maxwls"]
                        df = df.drop(index=g160mind)
                        df = df.drop(index=lowind)
                    else:
                        shortestwl = df.loc[lowind, "minwls"]
                        used = used.append(df.loc[lowind])
                        maxwl = df.loc[lowind, "maxwls"]
                        df = df.drop(lowind)
                else:
                    shortestwl = df.loc[lowind, "minwls"]
                    used = used.append(df.loc[lowind])
                    maxwl = df.loc[lowind, "maxwls"]
                    df = df.drop(lowind)
                while len(df) > 0:
                    lowind = df.loc[(df["minwls"] < maxwl) & (df["maxwls"] > maxwl)].index.values
                    # If G130M and G160M both exist for a given target, *always*
                    # abut them together regardless of other available gratings.
                    # This captures the case where there is FUSE bluer than COS/FUV.
                    if "G130M" in used.gratings.values and "G160M" in gratings and "G160M" not in used.gratings.values:
                        lowind = df.loc[df["gratings"] == "G160M"].index.values
                        maxwl = df.loc[lowind[0], "maxwls"]
                        used = used.append(df.loc[lowind])
                        df = df.drop(index=lowind)
                    # Handle case where more than one grating overlaps with bluer data.
                    elif len(lowind) > 1:
                        df2 = df.loc[lowind]
                        ranges = df2.maxwls - df2.minwls
                        biggest = ranges.idxmax()
                        match_grating = df2.loc[biggest, "gratings"]
                        match_ind = df.loc[df["gratings"] == match_grating].index.values
                        used = used.append(df.loc[match_ind])
                        maxwl = df.loc[match_ind, "maxwls"].values[0]
                        df = df.drop(index=lowind)
                    # If none overlap, abut with the next closest product.
                    elif len(lowind) == 0:
                        lowind = df["minwls"].idxmin()
                        used = used.append(df.loc[lowind])
                        maxwl = df.loc[lowind, "maxwls"]
                        df = df.drop(lowind)
                    # This is the easy case- only one mode overlaps with the bluer data.
                    else:
                        maxwl = df.loc[lowind[0], "maxwls"]
                        used = used.append(df.loc[lowind])
                        df = df.drop(index=lowind)
                    # Check every time if there are any modes that overlap completely
                    # with what has been abutted so far.
                    badinds = df.loc[(df["minwls"] > shortestwl) & (df["maxwls"] < maxwl)].index.values
                    if len(badinds) > 0:
                        df = df.drop(index=badinds)
                # If more than one instrument was selected for abutting,
                # create level 4 product.
                if len(set(used["ins"].values)) > 1:
                    abut_gr = used.iloc[0]["ins"] + "/" + used.iloc[0]["gratings"]
                    abutted = products[abut_gr]
                    for i in range(1, len(used)):
                        abut_gr = used.iloc[i]["ins"] + "/" + used.iloc[i]["gratings"]
                        abutted = abut(abutted, products[abut_gr])
                        abutted.propid = products[abut_gr].propid
                        abutted.rootname = products[abut_gr].rootname
                    filename = create_output_file_name(abutted, producttype, version, level=level)
                    filename = os.path.join(outdir, filename)
                    abutted.write(filename, clobber, level=level, version=version)
                    print(f"   Wrote {filename}")

def create_output_file_name(prod, producttype, version=VERSION, level=3):
    instrument = prod.instrument.lower()   # will be either cos, stis, or fuse. If abbuted can be cos-stis or cos-stis-fuse
    grating = prod.grating.lower()
    target = prod.target.lower()
    version = version.lower()
    propid = str(prod.propid)
    ipppss = prod.rootname[:7]
    detector = prod.detector.lower()

    # Target names can't have a period in them or it breaks MAST
    if "." in target:
        target = rename_target(target)
    tel = 'hst'
    if level == 0:
        suffix = "spec"
    if level == 1:
        suffix = "mspec"
    elif level == 3 or level == 2:
        suffix = "cspec"
    elif level == 4:
        suffix = "preview-spec"
        if "G430L" in prod.grating or "G750L" in prod.grating:
            grating = "uv-opt"
        else:
            grating = "uv"

    # Need to add logic for uv-opt here
    if producttype == 'visit':
        name = f"hst_{propid}_{instrument}_{target}_{grating}_{ipppss}_{suffix}.fits"
    elif producttype == 'proposal':
        name = f"hst_{propid}_{instrument}_{target}_{grating}_{suffix}.fits"
    return name

def rename_target(target_name):
    new_target_name = target_name.replace('.', '_')
    return new_target_name

def call_main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--indir",
                        default="./",
                        help="Directory(ies) with data to combine")
    parser.add_argument("-o", "--outdir", default=None,
                        help="Directory for output HLSPs")
    parser.add_argument("-v", "--version", default=VERSION,
                        help="Version number of the HLSP")
    parser.add_argument("-c", "--clobber", default=False,
                        action="store_true",
                        help="If True, overwrite existing products")
    args = parser.parse_args()

    main(args.indir, args.outdir, args.version, args.clobber)


if __name__ == '__main__':
    call_main()
