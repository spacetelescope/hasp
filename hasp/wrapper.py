from collections import defaultdict
import argparse
import os
import re
import glob
import datetime
from datetime import datetime as dt
import json
import numpy as np

import astropy
from astropy.io import fits
from astropy.time import Time

import hasp

from ullyses.coadd import COSSegmentList, STISSegmentList, CCDSegmentList
from ullyses.coadd import SegmentList, Segment

from .grating_priority import create_level4_products, GRATING_PRIORITIES_HASP, GRATING_PRIORITIES_HSLA

CAL_VER = 0.1

EXCLUDED_TARGETS = ['ANY', 'WAVE']

PREFILTERS = ['EXPFLAG', 'ZEROEXPTIME', 'PLANNEDVSACTUAL', 'MOVINGTARGET', 'NOTFINELOCK',
              'POSTARG1', 'POSTARG2', 'DITHERPERPENDICULARTOSLIT', 'MOSAICPURPOSE', 'PRISM',
              'COSBOA']

BAD_SEGMENTS = {'COS/G230L': 'NUVC'}

COORD_EPOCH = 2016.0

'''
This wrapper goes through each file in the selected directory ('indir') and
creates visit-level and program-level lists and dictionaries to collect all exposures
with the same target and grating.  These are then coadded with flux-based filtering.
Finally, the grating coadds are abutted using the grating priority table

'''


class HASP_SegmentList(SegmentList):
    """This class is a mixin to add the project-specific write function and functions
    to populate the target name and coordinates

    """

    def __init__(self, instrument, grating, inpath='./', infiles=None):
        self.bad_echelle_orders = []
        super().__init__(instrument, grating, inpath, infiles=None)

    def import_data(self, file_list):
        """This is used if the __init__ function is called with instrument
        and grating set to None.  This separates the selection of which files
        to work on from the constructor

        Parameters
        ----------

        file_list : list of strings
            List of files from which to make a product

        Returns
        -------

        None

        Side Effects
        ------------

        Populates many attributes of the calling object

        """

        alldata = []
        allhdr0 = []
        allhdr1 = []
        self.datasets = []
        self.num_exp = 0
        self.gratinglist = []
        self.instrumentlist = []
        self.aperturelist = []
        for datafile in file_list:
            print('Processing file {}'.format(datafile))
            with fits.open(datafile) as f1:
                prihdr = f1[0].header
                for extension in f1[1:]:
                    if extension.header['EXTNAME'] == 'SCI':
                        hdr1 = extension.header
                        data = extension.data
                        alldata.append(data)
                        allhdr0.append(prihdr)
                        allhdr1.append(hdr1)
                        self.datasets.append(datafile)
                        self.num_exp += 1
                        extver = extension.header['extver']
                        if extver > 1:
                            expname = extension.header['EXPNAME']
                            print(f'Extension {extver} with expname {expname} included for file {datafile}')

            if len(data) > 0:
                self.instrument = prihdr['INSTRUME'].upper()
                self.grating = prihdr['OPT_ELEM'].upper()
                self.echelle = False
                if self.grating.startswith('E'):
                    self.echelle = True
                self.propid = prihdr['PROPOSID']
                self.rootname = prihdr['ROOTNAME']
                self.detector = prihdr['DETECTOR']
                if self.grating not in self.gratinglist:
                    self.gratinglist.append(self.grating)
                if self.instrument not in self.instrumentlist:
                    self.instrumentlist.append(self.instrument)
                if self.instrument == 'COS':
                    self.aperture = prihdr['APERTURE']
                elif self.instrument == 'STIS':
                    self.aperture = prihdr['PROPAPER']
                else:
                    print(f'Unexpected value for instrument: {self.instrument}')
                    self.aperture = prihdr['aperture']
                if self.aperture not in self.aperturelist:
                    self.aperturelist.append(self.aperture)
                target = prihdr['TARGNAME'].strip()
                if target not in self.targnames:
                    self.targnames.append(target)
                try:
                    if prihdr['HLSP_LVL'] == 0:
                        self.level0 = True
                except KeyError:
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
                    instrument = hdr0['INSTRUME']
                    grating = hdr0['OPT_ELEM']
                    setting = f'{instrument}/{grating}'
                    sdqflags = hdr1['SDQFLAGS']
                    lifetime_position = get_lifetime_position(hdr0)
                    # Remove 16 from SDQFLAGS for STIS data if it's present
                    if self.instrument == "STIS" and (sdqflags & 16) == 16:
                        sdqflags -= 16
                    exptime = hdr1['EXPTIME']
                    for rownum, row in enumerate(data):
                        # Filter out bad COS segments
                        if setting in BAD_SEGMENTS.keys():
                            if row['SEGMENT'] in BAD_SEGMENTS[setting]:
                                print(f"Segment {row['SEGMENT']} removed from product for setting {setting}")
                                continue
                        segment = Segment()
                        segment.data = row
                        segment.sdqflags = sdqflags
                        segment.exptime = exptime
                        segment.filename = self.datasets[i]
                        segment.rownum = rownum
                        segment.lifetime_position = lifetime_position
                        self.members.append(segment)
        return

    def write(self, filename, overwrite=False, level=""):
        """Write a product to disk

        Parameters
        ----------
        filename : str
            Name of file to be written to

        overwrite : bool
            If True, overwrite existingfile with same name

        level : str or int
            HLSP Level:
                1. Single grating visit product
                2. Abutted multi grating visit product
                3. Single grating program product
                4. Abutted multi grating program product
                5. Single grating cross-program product
                6. Abutted multi grating cross-program product

        """
        if level in [1, 2]:
            ipppssoo = self.rootname[:6]
        elif level in [3, 4]:
            ipppssoo = self.rootname[:4]
        elif level in [5, 6]:
            ipppssoo = 'N/A'
        else:
            print(f'Unexpected value for level: {level}')
            ipppssoo = self.rootname
        if overwrite is False:
            if os.path.exists(filename):
                print(f'{filename} already exists and overwrite=False, skipping write')
                return
        self.targ_ra, self.targ_dec, self.epoch = self.get_coords()

        # Table 1 - HLSP data

        # set up the header
        hdr1 = fits.Header()
        hdr1['EXTNAME'] = ('SCI', 'Spectrum science arrays')
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
        hdr1['S_REGION'] = (self.obs_footprint(), 'Region footprint')

        # set up the table columns
        nelements = len(self.output_wavelength)
        goodpixels = np.where(self.output_exptime != 0.0)
        if len(goodpixels) != 0:
            first = goodpixels[0][0]
            last = goodpixels[0][-1] + 1
            nelements = last - first
        else:
            first = 0
            last = nelements
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
        table1.data['WAVELENGTH'] = self.output_wavelength[first:last].copy()
        table1.data['FLUX'] = self.output_flux[first:last].copy()
        table1.data['ERROR'] = self.output_errors[first:last].copy()
        table1.data['SNR'] = self.signal_to_noise[first:last].copy()
        table1.data['EFF_EXPTIME'] = self.output_exptime[first:last].copy()
        # HLSP primary header
        hdr0 = fits.Header()
        hdr0['EXTEND'] = ('T', 'FITS file may contain extensions')
        hdr0['NEXTEND'] = 3
        hdr0['FITS_VER'] = 'Definition of the Flexible Image Transport System (FITS) v4.0 https://fits.gsfc.nasa.gov/standard40/fits_standard40aa-le.pdf'
        hdr0['FITS_SW'] = ('astropy.io.fits v' + astropy.__version__, 'FITS file creation software')
        hdr0['ORIGIN'] = ('Space Telescope Science Institute', 'FITS file originator')
        hdr0['DATE'] = (str(datetime.date.today()), 'Date this file was written')
        hdr0['FILENAME'] = (os.path.basename(filename), 'Name of this file')
        hdr0['IPPPSSOO'] = (ipppssoo, 'IPPP or IPPPSS for product')
        hdr0['TELESCOP'] = (self.combine_keys("telescop", "multi"), 'Telescope used to acquire data')
        hdr0['INSTRUME'] = (self.instrument, 'Instrument used to acquire data')
        hdr0.add_blank('', after='TELESCOP')
        hdr0.add_blank('              / SCIENCE INSTRUMENT CONFIGURATION', before='INSTRUME')
        hdr0['DETECTOR'] = (self.combine_keys("detector", "multi"), 'Detector or channel used to acquire data')
        hdr0['NUM_EXP'] = (self.num_exp, "Number of exposures combined")
        hdr0['GRATING'] = ('-'.join(self.gratinglist), 'Grating(s) used')
        hdr0['CENWAVE'] = (self.combine_keys("cenwave", "multi"), 'Central wavelength setting for disperser')
        hdr0['SPORDER'] = (1, 'Spectral order')
        hdr0['APERTURE'] = (','.join(self.aperturelist), 'Identifier(s) of entrance aperture')
        hdr0['OBSMODE'] = (self.combine_keys("obsmode", "multi"), 'Instrument operating mode (ACCUM | TIME-TAG)')
        hdr0['TARGNAME'] = self.target
        if level in [5, 6]:
            if self.instrument == 'STIS':
                hdr0['LIFE_ADJ'] = 'N/A'
            else:
                hdr0['LIFE_ADJ'] = self.lifetime_position
        hdr0.add_blank(after='OBSMODE')
        hdr0.add_blank('              / TARGET INFORMATION', before='TARGNAME')

        hdr0['RADESYS'] = ('ICRS ', 'World coordinate reference frame')
        hdr0['TARG_RA'] = (self.targ_ra, '[deg] Target right ascension')
        hdr0['TARG_DEC'] = (self.targ_dec, '[deg] Target declination')
        hdr0['PROPOSID'] = (self.combine_keys("proposid", "multi"), 'Program identifier')
        hdr0['PINAME'] = (self.combine_keys("pr_inv_l", "multi"), "Principal Investigator")
        hdr0['MTFLAG'] = (self.combine_keys("mtflag", "multi"), 'Moving Target Flag')
        hdr0['EXTENDED'] = (self.combine_keys("extended", "max"), 'Is target extended?')
        hdr0.add_blank(after='TARG_DEC')
        hdr0.add_blank('           / PROVENANCE INFORMATION', before='PROPOSID')
        hdr0['CAL_VER'] = (f'HSLA Cal {CAL_VER}', 'HLSP processing software version')
        hdr0['HLSPID'] = ('HSLA', 'Name ID of this HLSP collection')
        hdr0['HSLPNAME'] = ('Hubble Spectroscopic Legacy Archive',
                            'Name ID of this HLSP collection')
        hdr0['HLSPLEAD'] = ('John Debes', 'Full name of HLSP project lead')
        hdr0['HLSP_LVL'] = (level, 'HASP HLSP Level')
        hdr0['LICENSE'] = ('CC BY 4.0', 'License for use of these data')
        hdr0['LICENURL'] = ('https://creativecommons.org/licenses/by/4.0/', 'Data license URL')
        hdr0['REFERENC'] = ('https://ui.adsabs.harvard.edu/abs/2020RNAAS...4..205R', 'Bibliographic ID of primary paper')
        hdr0['MINWAVE'] = (self.combine_keys("minwave", "min"), 'Minimum wavelength in spectrum')
        hdr0.add_blank(after='REFERENC')
        hdr0.add_blank('           / ARCHIVE SEARCH KEYWORDS', before='MINWAVE')
        hdr0['MAXWAVE'] = (self.combine_keys("maxwave", "max"), 'Maximum wavelength in spectrum')
        hdr0['CENTRWV'] = (0.5*(hdr0['MAXWAVE'] + hdr0['MINWAVE']), 'Central wavelength of the data')
        hdr0['BANDWID'] = hdr0['MAXWAVE'] - hdr0['MINWAVE']

        primary = fits.PrimaryHDU(header=hdr0)

        # Table 2 - individual product info

        # first set up header
        hdr2 = fits.Header()
        hdr2['EXTNAME'] = ('PROVENANCE', 'Metadata for contributing observations')
        # set up the table columns
        cfn = fits.Column(name='FILENAME', array=self.combine_keys("filename", "arr"), format='A64')
        ce_n = fits.Column(name='EXPNAME', array=self.combine_keys("expname", "arr"), format='A32')
        cpid = fits.Column(name='PROPOSID', array=self.combine_keys("proposid", "arr"), format='A32')
        ctel = fits.Column(name='TELESCOPE', array=self.combine_keys("telescop", "arr"), format='A32')
        cins = fits.Column(name='INSTRUMENT', array=self.combine_keys("instrume", "arr"), format='A32')
        cdet = fits.Column(name='DETECTOR', array=self.combine_keys("detector", "arr"), format='A32')
        cdis = fits.Column(name='DISPERSER', array=self.combine_keys("opt_elem", "arr"), format='A32')
        ccen = fits.Column(name='CENWAVE', array=self.combine_keys("cenwave", "arr"), format='A32')
        cap = fits.Column(name='APERTURE', array=self.combine_keys("aperture", "arr"), format='A32')
        clp = fits.Column(name='LIFE_ADJ', array=self.combine_keys("life_adj", "arr"), format='A32')
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

        cd2 = fits.ColDefs([cfn, ce_n, cpid, ctel, cins, cdet, cdis, ccen, cap, clp, csr, ccv, cdb, cdm,
                                cde, cexp, cmin, cmax])

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
        self.targ_ra, self.targ_dec, self.epoch = self.get_coords()
        radius = (2.5 / 2 / 3600)
        center_ra = self.targ_ra
        center_dec = self.targ_dec

        s_region = f"CIRCLE {center_ra} {center_dec} {radius}"
        return s_region

    def combine_keys(self, key, method):
        keymap = {"HST": {"expstart": ("expstart", 1),
                          "expend": ("expend", 1),
                          "exptime": ("exptime", 1),
                          "telescop": ("telescop", 0),
                          "instrume": ("instrume", 0),
                          "detector": ("detector", 0),
                          "opt_elem": ("opt_elem", 0),
                          "cenwave": ("cenwave", 0),
                          "aperture": ("aperture", 0),
                          "life_adj": ("life_adj", 0),
                          "obsmode": ("obsmode", 0),
                          "proposid": ("proposid", 0),
                          "centrwv": ("centrwv", 0),
                          "minwave": ("minwave", 0),
                          "maxwave": ("maxwave", 0),
                          "filename": ("filename", 0),
                          "specres": ("specres", 0),
                          "cal_ver": ("cal_ver", 0),
                          "mtflag": ("mtflag", 0),
                          "extended": ("extended", 0),
                          "pr_inv_l": ("pr_inv_l", 0),
                          "expname": ("expname", 1)},
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
                try:
                    val = self.primary_headers[i][actual_key]
                except KeyError:
                    val = '?'
            else:
                try:
                    val = self.first_headers[i][actual_key]
                except KeyError:
                    val = '?'
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

    def calculate_statistics(self, verbose=False):
        """Calcuate statistics for each of the exposures that
        contribute to the coadded product

        """
        print(f'Using a maximum SNR of {self.snrmax} in flux-based filtering')
        nsegments = len(self.members)
        if nsegments == 1:
            segment = self.members[0]
            filename = segment.filename
            rownum = 0
            ndeviations = len(segment.data)
            a0 = np.zeros(1)
            return [filename], [rownum], [ndeviations], a0, a0, a0, a0
        mean_deviation = np.zeros(nsegments)
        median_deviation = np.zeros(nsegments)
        mean_squared_deviation = np.zeros(nsegments)
        median_squared_deviation = np.zeros(nsegments)
        ndeviations = np.zeros(nsegments, dtype='int')
        filename = []
        rownum = []
        for nseg, segment in enumerate(self.members):
            goodpixels = np.where((segment.data['dq'] & segment.sdqflags) == 0)
            if len(goodpixels[0]) == 0:
                print('No good pixels for segment #{}'.format(nseg))
                filename.append(segment.filename)
                rownum.append(segment.rownum)
                continue
            wavelength = segment.data['wavelength'][goodpixels]
            indices = self.wavelength_to_index(wavelength)
            npts = len(indices)
            flux = segment.data['flux'][goodpixels]
            error = segment.data['error'][goodpixels]
            deviation = np.zeros(npts)
            deviation_squared = np.zeros(npts)
            ndeviations[nseg] = 0
            for i in range(npts):
                if error[i] != 0.0 and segment.exptime != self.output_exptime[indices[i]]:
                    min_error = flux[i] / self.snrmax
                    deviation[i] = (flux[i] - self.output_flux[indices[i]])
                    deviation[i] = deviation[i] / max(error[i], min_error)
                    ndeviations[nseg] = ndeviations[nseg] + 1
                    deviation_squared[i] = deviation[i] * deviation[i]
            nonzero_deviations = np.where(deviation != 0.0)
            n_nonzero = len(nonzero_deviations[0])
            if n_nonzero > 0:
                sorted_nonzero_deviations = deviation[nonzero_deviations]
                sorted_nonzero_deviations.sort()
                sorted_nonzero_squared_deviations = deviation_squared[nonzero_deviations]
                mean_deviation[nseg] = deviation[nonzero_deviations].mean()
                mean_squared_deviation[nseg] = deviation_squared[nonzero_deviations].mean()
                median_deviation[nseg] = np.median(sorted_nonzero_deviations)
                median_squared_deviation[nseg] = np.median(sorted_nonzero_squared_deviations)
                filename.append(segment.filename)
                rownum.append(segment.rownum)
            else:
                mean_deviation[nseg] = 0.0
                median_deviation[nseg] = 0.0
                mean_squared_deviation[nseg] = 0.0
                median_squared_deviation[nseg] = 0.0
            if verbose:
                print(f'for segment {nseg}')
                print(f'{ndeviations[nseg]} non-zero deviations')
                print(f'Mean deviation = {mean_deviation[nseg]}')
                print(f'Mean squared deviation = {mean_squared_deviation[nseg]}')
                print(f'Median deviation = {median_deviation[nseg]}')
                print(f'Median squared deviation = {median_squared_deviation[nseg]}')

        return filename, rownum, ndeviations, mean_deviation, median_deviation, mean_squared_deviation, median_squared_deviation


class HASP_COSSegmentList(COSSegmentList, HASP_SegmentList):
    pass


class HASP_STISSegmentList(STISSegmentList, HASP_SegmentList):
    pass


class HASP_CCDSegmentList(CCDSegmentList, HASP_SegmentList):
    pass


def main(indir, outdir, clobber=False, threshold=-50, snrmax=20, no_keyword_filtering=False, cross_program=False,
         grating_table=None):
    # Find out which unique modes are present
    # For single visit products, a unique mode is
    # (target, instrument, grating, detector, visit)
    # For single program products, the mode is
    # (target, instrument, grating, detector)
    print(f'HASP version {hasp.__version__}')
    try:
        import ullyses
        print(f'Ullyses version {ullyses.__version__}')
    except:
        print('Ullyses version unavailable')
    if cross_program:
        grating_priorities = GRATING_PRIORITIES_HSLA
    else:
        grating_priorities = GRATING_PRIORITIES_HASP
    if grating_table is not None:
        fg = open(grating_table)
        grating_priorities = json.load(fg)
        fg.close() 
    uniqmodes = []
    uniqvisitmodes = []
    uniqproposalmodes = []
    uniqlpindependentmodes = []
    uniqsingletargetmodes = []
    targetlist = []
    visitlist = []
    proposallist = []
    visitdict = {}
    proposaldict = {}
    singletargetdict = {}
    lpindependentdict = {}
    spec1d = glob.glob(os.path.join(indir, '*_x1d.fits')) + glob.glob(os.path.join(indir, '*_sx1.fits'))
    spec1d.sort()
    if no_keyword_filtering:
        keyword_filters = ['PRISM']
    else:
        keyword_filters = PREFILTERS
    spec1d = prefilter(spec1d, filters=keyword_filters)
#
# Create the list of modes
    lifetime_positions = []
    print('Creating list of unique modes from these files:')
    for myfile in spec1d:
        f1 = fits.open(myfile)
        prihdr = f1[0].header
        instrument = prihdr['INSTRUME']
        grating = prihdr['OPT_ELEM']
        detector = prihdr['DETECTOR']
        lifetime_position = get_lifetime_position(prihdr)
        targname = prihdr['TARGNAME']
        visit = myfile[-14:-12]
        proposal = prihdr['PROPOSID']
        aperture = prihdr['PROPAPER']
        visit = (proposal, visit)
        obsmode = (instrument, grating, detector, lifetime_position)
        lpindependentmode = (instrument, grating, detector)
        visitmode = (instrument, grating, detector, targname, visit)
        proposalmode = (instrument, grating, detector, targname, proposal)
        if targname not in targetlist:
            targetlist.append(targname)
        if obsmode not in uniqmodes:
            uniqmodes.append(obsmode)
            singletargetdict[obsmode] = [myfile]
        else:
            singletargetdict[obsmode].append(myfile)
        if lpindependentmode not in uniqlpindependentmodes:
            uniqlpindependentmodes.append(lpindependentmode)
            lpindependentdict[lpindependentmode] = [myfile]
        else:
            lpindependentdict[lpindependentmode].append(myfile)
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
        print(myfile, targname, instrument, detector, grating, aperture, lifetime_position, proposal, visit)
        f1.close()

    visitlist.sort()
    proposallist.sort()

    
    if cross_program:
        _, _ = create_cross_program_products('cross-program', uniqmodes,singletargetdict,
                                             indir,snrmax, threshold, outdir, clobber,
                                             keyword_filters, write_products=True)
        productlist, productdict = create_cross_program_products('cross-program',
                                                                 uniqlpindependentmodes,
                                                                 lpindependentdict, indir,
                                                                 snrmax, threshold, outdir, clobber, 
                                                                 keyword_filters,
                                                                 write_products=False)
        if len(productlist) == 0:
            print("No products to abut for this target")
            return
        abutted_product = create_level4_products(productlist, productdict, grating_table=grating_priorities)
        if abutted_product is not None:
            level = 6
            filename = create_output_file_name(abutted_product, 'cross-program', level)
            filename = os.path.join(outdir, filename)
            abutted_product.write(filename, clobber, level=6)
            print(f"   Wrote {filename}")

    else:
        create_products('visit', visitlist, visitdict, indir, snrmax, threshold, outdir,
                        clobber, keyword_filters, grating_priorities)
        create_products('proposal', proposallist, proposaldict, indir, snrmax, threshold,
                        outdir, clobber, keyword_filters, grating_priorities)


def create_products(product_type, product_type_list, product_type_dict, indir, snrmax,
                    threshold, outdir, clobber, keyword_filters, grating_priorities):
    print(f'Looping over {product_type}s')
    for prod_type in product_type_list:
        print('Processing product {}'.format(prod_type))
        thisprodkeys = []
        for prodspec in product_type_dict.keys():
            if prodspec[4] == prod_type:
                thisprodkeys.append(prodspec)
        targetsinthisprod = []
        for prodspec in thisprodkeys:
            thistarget = prodspec[3]
            if thistarget not in targetsinthisprod:
                targetsinthisprod.append(thistarget)
        print('Targets in {} {}: {}'.format(product_type, prod_type, targetsinthisprod))

        for target in targetsinthisprod:
            print('Processing target {} in {} {}'.format(target, product_type, prod_type))
            if target in EXCLUDED_TARGETS:
                print('{} is an excluded target and will not be processed'.format(target))
                continue
            thisprodandtargetspec = []
            for prodspec in thisprodkeys:
                thistarget = prodspec[3]
                if target == thistarget:
                    thisprodandtargetspec.append(prodspec)

            # Create dictionary of all products, with each set to None by default
            products = defaultdict(lambda: None)
            productlist = []
            productdict = {}
            uniqmodes = []
            for uniqmode in thisprodandtargetspec:
                instrument = uniqmode[0]
                grating = uniqmode[1]
                detector = uniqmode[2]
                setting = instrument + '/' + grating
                print('Processing grating {}'.format(setting))
                if (instrument, grating, detector) not in uniqmodes:
                    uniqmodes.append((instrument, grating, detector))
                files_to_import = product_type_dict[uniqmode]
                if product_type == 'proposal':
                    if 'MOVINGTARGET' in keyword_filters:
                        files_to_import = check_for_moving_targets(files_to_import)
                # Flux based filtering loop
                no_good_data = False
                while True:
                    if len(files_to_import) == 0:
                        print('No good files')
                        no_good_data = True
                        break
                    print('Importing files {}'.format(files_to_import))
                    # this instantiates the class
                    if instrument == 'COS':
                        prod = HASP_COSSegmentList(None, inpath=indir)
                    elif instrument == 'STIS':
                        if detector == 'CCD':
                            prod = HASP_CCDSegmentList(None, inpath=indir)
                        else:
                            prod = HASP_STISSegmentList(None, inpath=indir)
                    else:
                        print(f'Unknown mode [{instrument}, {grating}, {detector}]')
                        continue
                    if prod is not None:
                        prod.import_data(files_to_import)
                    prod.target = prod.get_targname()
                    prod.targ_ra, prod.targ_dec, prod.epoch = prod.get_coords()
                    prod.snrmax = snrmax
                    prod.gratinglist = [grating]
                    prod.disambiguated_grating = grating.lower()
                    prod.lifetime_position = prod.members[0].lifetime_position
                    if grating in ['G230L', 'G140L']:
                        prod.disambiguated_grating = instrument.lower()[0] + grating.lower()
                    # these two calls perform the main functions
                    if len(prod.members) > 0:
                        prod.create_output_wavelength_grid()
                        prod.coadd()
                        if prod.first_good_wavelength is None:
                            print("No good data, skipping")
                            no_good_data = True
                            break
                        result = prod.calculate_statistics()
                        files_to_cull = analyse_result(result, threshold=threshold)
                        if files_to_cull == []:
                            break
                        else:
                            print("Removing files from list:")
                            print(files_to_cull)
                            cull_files(files_to_cull, files_to_import)
                    else:
                        print(f'No valid data for grating {grating}')
                # this writes the output file
                # If making HLSPs for a DR, put them in the official folder
                if no_good_data:
                    continue
                prod.target = prod.get_targname()
                prod.targ_ra, prod.targ_dec, prod.epoch = prod.get_coords()
                target = prod.target.lower()
                level = 1
                if product_type == 'proposal': level = 3
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                outname = create_output_file_name(prod, product_type, level)
                outname = os.path.join(outdir, outname)
                prod.write(outname, clobber, level=level)
                print(f"   Wrote {outname}")
                productlist.append(prod)
                products[setting] = prod
                productdict[setting] = prod
                prod.product_type = product_type
            if len(productlist) == 0:
                print("No products to abut for this setting")
                continue
            if len(productlist) == 1:
                print("Only 1 grating to abut, skipping abutment")
                continue
            else:
                abutted_product = create_level4_products(productlist, productdict, grating_table=grating_priorities)
            if abutted_product is not None:
                level = 2
                if product_type == 'proposal': level = 4
                filename = create_output_file_name(abutted_product, product_type, level)
                filename = os.path.join(outdir, filename)
                abutted_product.write(filename, clobber, level=level)
                print(f"   Wrote {filename}")

    return

def create_cross_program_products(product_type, thisprodandtargetspec, product_type_dict, indir, snrmax,
                                  threshold, outdir, clobber, keyword_filters, write_products=True):
    # For cross-program products, we need to make single grating products for each distinct Lifetime Position (LP)
    # for COS.  These should have names like hst_cos_targetname_grating_lpnn_cspec.fits.  If there is only 1 LP
    # product for a COS grating, or if the product is for STIS data, the filename should omit the _lpnn part.
    # For gratings with multiple LPs, we need to also create a product from all the LPs so that it can be used in
    # the abutment, but this LP-independent product should not be written.  The abutted product name should be of the
    # form hst_cos-stis_targetname_hyphenated-list-of-gratings_cspec.fits
    products = defaultdict(lambda: None)
    productlist = []
    productdict = {}
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    real_indir = os.path.realpath(indir)
    target_name = real_indir.split('/')[-1]
    modes_with_more_than_1_lp = []
    lp_independent_modes = []
    for mode in thisprodandtargetspec:
        lpindependentmode = mode[:3]
        if lpindependentmode in lp_independent_modes:
            if lpindependentmode not in modes_with_more_than_1_lp:
                modes_with_more_than_1_lp.append(lpindependentmode)
        else:
            lp_independent_modes.append(lpindependentmode)
    print(thisprodandtargetspec)
    print(modes_with_more_than_1_lp)
    print(f'Creating cross-program products for target {target_name}')
    for uniqmode in thisprodandtargetspec:
        files_to_import = product_type_dict[uniqmode]
        if 'MOVINGTARGET' in keyword_filters:
            files_to_import = check_for_moving_targets(files_to_import)
        if len(files_to_import) == 0:
            print(f"No files to import, exiting mode {uniqmode}")
            continue
        instrument = uniqmode[0]
        grating = uniqmode[1]
        detector = uniqmode[2]
        lifetime_position = 'N/A'
        setting = instrument + '/' + grating
        if len(uniqmode) == 4 and instrument == 'COS':
            lifetime_position = uniqmode[3]
            print(f'Processing grating {setting} at lifetime position {lifetime_position}')
        else:
            print(f'Processing grating {setting}')
        prod = processmode(uniqmode, files_to_import, indir, target_name, snrmax, threshold)
        if prod is None:
            continue
        prod.target = target_name
        prod.targ_ra, prod.targ_dec, prod.epoch = prod.get_coords()
        prod.snrmax = snrmax
        prod.gratinglist = [grating]
        prod.disambiguated_grating = grating.lower()
        if grating in ['G230L', 'G140L']:
            prod.disambiguated_grating = instrument.lower()[0] + grating.lower()
        prod.lifetime_position = lifetime_position
        prod.product_type = product_type
        prod.morethan1lp = False
        if uniqmode[:3] in modes_with_more_than_1_lp:
            prod.morethan1lp = True
        # this writes the output file
        # If making HLSPs for a DR, put them in the official folder
        if write_products:
            level = 5
            outname = create_output_file_name(prod, product_type, level)
            outname = os.path.join(outdir, outname)
            prod.write(outname, clobber, level=5)
            print(f"   Wrote {outname}")

        productlist.append(prod)
        products[setting] = prod
        productdict[setting] = prod
    return (productlist, productdict)

def processmode(uniqmode, files_to_import, indir, target_name, snrmax, threshold):
    # Flux based filtering loop
    instrument = uniqmode[0]
    grating = uniqmode[1]
    detector = uniqmode[2]
    while True:
        if len(files_to_import) == 0:
            print('No good files')
            no_good_data = True
            break
        print('Importing files {}'.format(files_to_import))
        # this instantiates the class
        if instrument == 'COS':
            prod = HASP_COSSegmentList(None, inpath=indir)
        elif instrument == 'STIS':
            if detector == 'CCD':
                prod = HASP_CCDSegmentList(None, inpath=indir)
            else:
                prod = HASP_STISSegmentList(None, inpath=indir)
        else:
            print(f'Unknown mode [{instrument}, {grating}, {detector}]')
            continue
        if prod is not None:
            prod.import_data(files_to_import)
        # these two calls perform the main functions
        if len(prod.members) > 0:
            prod.create_output_wavelength_grid()
            prod.coadd()
            if prod.first_good_wavelength is None:
                print("No good data, skipping")
                no_good_data = True
                return None
            prod.snrmax = snrmax
            result = prod.calculate_statistics()
            files_to_cull = analyse_result(result, threshold=threshold)
            if files_to_cull == []:
                break
            else:
                print("Removing files from list:")
                print(files_to_cull)
                cull_files(files_to_cull, files_to_import)
        else:
            print(f'No valid data for grating {grating}')
    return prod

def get_lifetime_position(prihdr):
    try:
        lifetime_position = prihdr['LIFE_ADJ']
        if lifetime_position not in [x + 1 for x in range(10)]:
            lifetime_position = 0
    except KeyError:
        lifetime_position = 0
    return lifetime_position


def analyse_result(results, threshold=-50):
    files_to_cull = []
    input_filenames = results[0]
    row_numbers = results[1]
    npts = results[2]
    median_deviation = results[4]
    scaledmedian = median_deviation * np.sqrt(npts)
    rows_to_cull = np.where(scaledmedian < threshold)
    for row in rows_to_cull[0]:
        filename = input_filenames[row]
        rownum = row_numbers[row]
        print(f'Segment #{rownum} from file {filename} has scaled median = {scaledmedian[row]}')
        if input_filenames[row] not in files_to_cull:
            print(f'Removing file {filename} from product')
            files_to_cull.append(input_filenames[row])
        else:
            print(f'File {filename} already selected for removal from product')
    return files_to_cull


def cull_files(files_to_cull, file_list):
    for thisfile in files_to_cull:
        if thisfile in file_list:
            file_list.remove(thisfile)

PREFILTERS = ['EXPFLAG', 'ZEROEXPTIME', 'PLANNEDVSACTUAL', 'MOVINGTARGET', 'NOTFINELOCK',
              'POSTARG1', 'POSTARG2', 'DITHERPERPENDICULARTOSLIT', 'MOSAICPURPOSE', 'PRISM',
              'COSBOA']

def prefilter(file_list, filters):
    """Pre-filter the input exposure filenames
    Possible filters to apply are:

    'EXPFLAG': Filter out datasets that EXPFLAG keyword anything other than 'NORMAL'

    'ZEROEXPTIME': Filter out datasets that have EXPTIME = 0.0

    'PLANNEDVSACTUAL': Filter out datasets where the actual exposure time is less than a certain
                       fraction of planned exposure time

    'MOVINGTARGET': Filter exposures of moving targets from program-level products

    'NOTFINELOCK': Filter exposures for which FGSLOCK is not 'FINE'

    'POSTARG1': Filter exposures that have POSTARG1 != 0.0

    'POSTARG2': Filter exposures POSTARG2 != 0.0 and P1_PURPS != 'DITHER'

    'DITHERPERPENDICULARTOSLIT': Filter exposures where PATTERN1 = STIS-PERP-TO-SLIT and P1_FRAME = POS-TARG

    'MOSAICPURPOSE': Filter exposures that have P1_PURPS = MOSAIC

    'PRISM': Filter STIS exposures with OPT_ELEM = PRISM

    'COSBOA': Filter COS exposures with APERTURE = BOA (Bright Object Aperture)

    """

    if 'EXPFLAG' in filters:
        goodfiles = []
        for fitsfile in file_list:
            try:
                value = fits.getval(fitsfile, 'EXPFLAG', ext=1)
            except KeyError:
                value = 'Not found'
            if value == 'NORMAL':
                goodfiles.append(fitsfile)
            else:
                print(f'File {fitsfile} removed from products because EXPFLAG = {value}')
        file_list = goodfiles

    if 'ZEROEXPTIME' in filters:
        goodfiles = []
        for fitsfile in file_list:
            try:
                value = fits.getval(fitsfile, 'EXPTIME', ext=1)
            except KeyError:
                value = 'Not found'
            if value > 0.0:
                goodfiles.append(fitsfile)
            else:
                print(f'File {fitsfile} removed from products because EXPTIME = {value}')
        file_list = goodfiles

    if 'PLANNEDVSACTUAL' in filters:
        goodfiles = []
        for fitsfile in file_list:
            try:
                actual = fits.getval(fitsfile, 'EXPTIME', ext=1)
                planned = fits.getval(fitsfile, 'PLANTIME', ext=1)
                if actual >= 0.8 * planned:
                    goodfiles.append(fitsfile)
                else:
                    print(f'File {fitsfile} removed from products because EXPTIME ({actual}) < 0.8*PLANTIME ({planned})')
            except KeyError:
                goodfiles.append(fitsfile)
        file_list = goodfiles

    if 'NOTFINELOCK' in filters:
        goodfiles = []
        for fitsfile in file_list:
            try:
                value = fits.getval(fitsfile, 'FGSLOCK', ext=1)
                if value == 'FINE':
                    goodfiles.append(fitsfile)
                else:
                    print(f'File {fitsfile} removed from products because FGSLOCK = {value}')
            except KeyError:
                goodfiles.append(fitsfile)
        file_list = goodfiles

    if 'POSTARG1' in filters:
        goodfiles = []
        for fitsfile in file_list:
            try:
                value = fits.getval(fitsfile, 'POSTARG1')
                if value == 0.0:
                    goodfiles.append(fitsfile)
                else:
                    print(f'File {fitsfile} removed from products because POSTARG1 = {value}')
            except KeyError:
                goodfiles.append(fitsfile)
        file_list = goodfiles

    if 'POSTARG2' in filters:
        goodfiles = []
        for fitsfile in file_list:
            try:
                value = fits.getval(fitsfile, 'POSTARG2')
                if value == 0.0:
                    goodfiles.append(fitsfile)
                else:
                    if fits.getval(fitsfile, 'INSTRUME') == 'STIS':
                        purpose = fits.getval(fitsfile, 'P1_PURPS')
                        if purpose == 'DITHER':
                            goodfiles.append(fitsfile)
                        else:
                            print(f'File {fitsfile} removed from products because POSTARG2 = {value} and P1_PURPS != DITHER')
            except KeyError:
                goodfiles.append(fitsfile)
        file_list = goodfiles

    if 'DITHERPERPENDICULARTOSLIT' in filters:
        goodfiles = []
        for fitsfile in file_list:
            try:
                pattern1 = fits.getval(fitsfile, 'PATTERN1')
                if pattern1 == 'STIS-PERP-TO-SLIT':
                    p1_frame = fits.getval(fitsfile, 'P1_FRAME')
                    if p1_frame != 'POS-TARG':
                        goodfiles.append(fitsfile)
                    else:
                        print(f'File {fitsfile} removed from products because PATTERN1 = STIS-PERP-TO-SLIT and P1_FRAME = POS-TARG')
                else:
                    goodfiles.append(fitsfile)
            except KeyError:
                goodfiles.append(fitsfile)
        file_list = goodfiles

    if 'MOSAICPURPOSE' in filters:
        goodfiles = []
        for fitsfile in file_list:
            try:
                value = fits.getval(fitsfile, 'P1_PURPS')
                if value != 'MOSAIC':
                    goodfiles.append(fitsfile)
                else:
                    print(f'File {fitsfile} removed from products because P1_PURPS = MOSAIC')
            except KeyError:
                goodfiles.append(fitsfile)
        file_list = goodfiles

    if 'PRISM' in filters:
        goodfiles = []
        for fitsfile in file_list:
            value = fits.getval(fitsfile, 'OPT_ELEM')
            if 'PRISM' not in value:
                goodfiles.append(fitsfile)
            else:
                print(f'File {fitsfile} removed from products because OPT_ELEM = {value}')
        file_list = goodfiles

    if 'COSBOA' in filters:
        goodfiles = []
        for fitsfile in file_list:
            instrument = fits.getval(fitsfile, 'INSTRUME')
            if instrument == 'COS':
                aperture = fits.getval(fitsfile, 'APERTURE')
                if aperture != 'BOA':
                    goodfiles.append(fitsfile)
                else:
                    print(f'File {fitsfile} removed from products because COS APERTURE = BOA')
            else:
                goodfiles.append(fitsfile)
        file_list = goodfiles

    return goodfiles


def check_for_moving_targets(files_to_import):
    """Program level products are not made for moving targets
    
    This function removes files with moving targets from the list of input files and returns
    a list without these files

    Moving target have primary header keyword MTFLAG set to 'T'
    
    Parameters
    ----------
    
    files_to_import : list
        List of files to search for moving targets
        
    Returns
    -------
    
    not_moving : list
        List of files with non-moving targets
    
    """
    not_moving = []
    for fitsfile in files_to_import:
        f1 = fits.open(fitsfile)
        prihdr = f1[0].header
        mtflag = prihdr['MTFLAG']
        if mtflag != 'T':
            not_moving.append(fitsfile)
        else:
            targname = prihdr['TARGNAME']
            print(f'File {fitsfile} removed from program products because {targname} is a moving target')
        f1.close()
    ngood = len(not_moving)
    if ngood != 0 and ngood != len(files_to_import):
        print('Some (but not all) files with this target name have MTFLAG="T"')
    return not_moving


def create_output_file_name(prod, producttype, level):
    instrument = prod.instrument.lower()   # will be either cos or stis. If abutted can be cos-stis
    grating = prod.disambiguated_grating.lower()
    target = prod.target.lower()
    propid = str(prod.propid)
    ipppss = prod.rootname[:6]
    ippp = prod.rootname[:4]

    target = sanitize_targname(target)

    if level in range(1, 6):
        suffix = "cspec"
    elif level == 6:
        suffix = "aspec"
    else:
        print(f"Unknown level {level}")
        suffix = "spec"
    if producttype == 'visit':
        name = f"hst_{propid}_{instrument}_{target}_{grating}_{ipppss}_{suffix}.fits"
    elif producttype == 'proposal':
        name = f"hst_{propid}_{instrument}_{target}_{grating}_{ippp}_{suffix}.fits"
    elif producttype == 'cross-program':
        if level == 6:
            name = f"hst_{target}_{suffix}.fits"
        else:
            if (instrument == 'cos'):
                lifetime_position_string = 'lp{:02d}'.format(prod.lifetime_position)
                name = f"hst_{instrument}_{target}_{grating}_{lifetime_position_string}_{suffix}.fits"
            else:
                name = f"hst_{instrument}_{target}_{grating}_{suffix}.fits"
    return name


def sanitize_targname(target_name):
    new_target_name = target_name.replace('.', 'd')
    new_target_name = new_target_name.replace('+', 'p')
    new_target_name = new_target_name.replace('_', '-')
    return new_target_name

def parse_epoch(epoch_string):
    epoch_list = re.findall(r"[-+]?(?:\d*\.*\d+)", epoch_string)
    if len(epoch_list) > 1:
        print(f'Epoch string {epoch_string} parses to more than 1 floating-point value')
        return None
    elif len(epoch_list) == 0:
        return None
    else:
        epoch_float = float(epoch_list[0])
        if epoch_float < 1850.0 or epoch_float > 2100.0:
            print(f'Unreasonable value for parsed epoch: {epoch_float}')
            return None
        else:
            return epoch_float


def call_main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--indir",
                        default="./",
                        help="Directory(ies) with data to combine")
    parser.add_argument("-o", "--outdir", default=None,
                        help="Directory for output HLSPs")
    parser.add_argument("-c", "--clobber", default=False,
                        action="store_true",
                        help="If True, overwrite existing products")
    parser.add_argument("-t", "--threshold",
                        default=-50, type=float,
                        help="Threshold for flux-based filtering")
    parser.add_argument("-s", "--snrmax",
                        default=20.0, type=float,
                        help="Maximum SNR per pixel for flux-based filtering")
    parser.add_argument("-k", "--no_keyword_filtering", default=False,
                        action="store_true",
                        help="Disable keyword based filtering (except for STIS PRISM data, which is always filtered)")
    parser.add_argument("-x", "--cross_program",
                        default=False, action="store_true",
                        help="Create cross-program products only")
    parser.add_argument("-g", "--grating_table",
                        default=None,
                        help="Name of grating priority file [None]")
    args = parser.parse_args()

    main(args.indir, args.outdir, args.clobber, args.threshold, args.snrmax, args.no_keyword_filtering,
         args.cross_program, args.grating_table)


if __name__ == '__main__':
    call_main()
