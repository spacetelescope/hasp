# Test the wrapper script
import os
import glob
import shutil
import urllib.request
import tarfile

from astropy.io.fits import FITSDiff

from hasp import wrapper

Program_10033 = {'name': '10033',
                 'url': 'https://stsci.box.com/shared/static/wnmma331eixblioo5jfnmjnzz2fjt01r.gz'}
Program_11839 = {'name': '11839',
                 'url': 'https://stsci.box.com/shared/static/a05mixac3hreg6g07f1kkogcpphdo26a.gz'}
Program_13471 = {'name': '13471',
                 'url': 'https://stsci.box.com/shared/static/jac75olate8hjalvc3tgor1fuvxupiqm.gz'}

HD104237E = {'name': 'hd-104237e',
             'url': 'https://stsci.box.com/shared/static/irelgbjs7zhdiljksao4su2c6tfhyntx.gz'}
V_HK_ORI = {'name': 'v-hk-ori',
            'url': 'https://stsci.box.com/shared/static/az3ytnwohj0t4wnc4m8oqbw9ziozwqx1.gz'}


class TestWrapper():

    def test_10033(self):
        program = Program_10033
        self.setup_tree(program)
        self.run_wrapper(program['name'])
        report = self.compare_outputs(program['name'])
        self.cleanup(program['name'])
        if report is not None:
            raise AssertionError(report)
        return

    def test_11839(self):
        program = Program_11839
        self.setup_tree(program)
        self.run_wrapper(program['name'])
        report = self.compare_outputs(program['name'])
        self.cleanup(program['name'])
        if report is not None:
            raise AssertionError(report)
        return
    
    def test_13471(self):
        program = Program_13471
        self.setup_tree(program)
        self.run_wrapper(program['name'])
        report = self.compare_outputs(program['name'])
        self.cleanup(program['name'])
        if report is not None:
            raise AssertionError(report)
        return

    def setup_tree(self, program):
        program_name = program['name']
        url = program['url']
        if os.path.isdir(program_name):
            shutil.rmtree(program_name)
        filename = program_name + '.tar.gz'
        _ = urllib.request.urlretrieve(url, filename)
        data_tarfile = tarfile.open(filename, mode='r|gz')
        data_tarfile.extractall(filter='data')
        return

    def run_wrapper(self, program):
        indir = program + '/input/'
        wrapper.main(indir, outdir=indir)
        return

    def compare_outputs(self, program):
        report = None
        # Outputs from current run are in ./, truth files to compare
        # with are in ./truth
        all_ok = True
        fitsdiff_report = ''
        keywords_to_ignore = ['DATE', 'FITS_SW', 'FILENAME',
                              'HLSP_VER', 'S_REGION', 'CAL_VER']
        new_hlsps = glob.glob(program + '/input/hst_*')
        for new_product in new_hlsps:
            truth_filename = self.get_truth_filename(program, new_product)
            fdiff = FITSDiff(new_product, truth_filename,
                             ignore_keywords=keywords_to_ignore,
                             rtol=3.0e-7)
            fitsdiff_report += fdiff.report()
            if not fdiff.identical and all_ok:
                all_ok = False
        if not all_ok:
            report = os.linesep + fitsdiff_report
            return report
        print(fitsdiff_report)
        return None

    def get_truth_filename(self, program, product):
        # Get the truth filename.  The data release might be different
        filename = os.path.basename(product)
        truth_filename = program + '/truth/' + filename
        return truth_filename

    def cleanup(self, program):
        shutil.rmtree(program)
        os.remove(program + '.tar.gz')
        return
