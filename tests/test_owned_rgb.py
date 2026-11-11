"""Owner-only RGB inversion matches full-site inversion on retained pixels."""
import ast
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from asispectralinversion import inversion
from asispectralinversion.ownership import invert_owned_rgb, combine_owned_products
from test_inversion_lookup import inversion_case


class TestOwnedRGB(unittest.TestCase):
    def test_merged_results_match_full_site_inversion_exactly(self):
        lookup, observations = inversion_case()
        owners = np.array([[1, 2, 1], [2, 0, 1]], dtype=np.uint8)
        codes = {'ARV': 1, 'VEE': 2}
        baseline, compact = {}, {}
        for site, code in codes.items():
            rgb = tuple(value * (1 if code == 1 else .9) for value in observations)
            q0, e0 = inversion.calculate_E0_Q_v2(*rgb, lookup, plot=False)
            with patch.object(inversion, 'calculate_E0_Q_v2', wraps=inversion.calculate_E0_Q_v2) as call:
                q, e, sp, sh = invert_owned_rgb(*rgb, lookup, owners == code, False)
            expected_count = np.count_nonzero((owners == code) & np.isfinite(rgb[0]))
            self.assertEqual(call.call_args.args[0].shape, (expected_count,))
            self.assertIsNone(sp)
            self.assertIsNone(sh)
            np.testing.assert_array_equal(q[owners == code], q0[owners == code])
            np.testing.assert_array_equal(e[owners == code], e0[owners == code])
            self.assertTrue(np.isnan(q[owners != code]).all())
            baseline[site] = dict(q=q0,e0=e0,valid=np.isfinite(q0)&np.isfinite(e0))
            compact[site] = dict(q=q,e0=e,valid=np.isfinite(q)&np.isfinite(e))
        old = combine_owned_products(baseline, owners, codes, False)
        new = combine_owned_products(compact, owners, codes, False)
        for key in ('q','e0','source_site'):
            np.testing.assert_array_equal(new[key],old[key])
        self.assertTrue((new['ownership_count'] <= 1).all())

    def test_empty_and_invalid_owned_pixels_do_not_call_inversion(self):
        rgb = (np.array([0., np.nan, -1., 2.]),)*3
        with patch.object(inversion, 'calculate_E0_Q_v2') as call, patch.object(inversion,'calculate_Sig') as sig:
            for mask in (np.zeros(4,bool), np.array([True,True,True,False])):
                outputs = invert_owned_rgb(*rgb, {}, mask)
                for output in outputs:self.assertTrue(np.isnan(output).all())
            call.assert_not_called()
            sig.assert_not_called()

    def test_conductance_receives_only_selected_results(self):
        rgb=(np.ones((2,2)),)*3
        mask=np.array([[True,False],[False,True]])
        with patch.object(inversion,'calculate_E0_Q_v2',return_value=(np.array([1.,2.]),np.array([300.,400.]))), patch.object(inversion,'calculate_Sig',return_value=(np.array([3.,4.]),np.array([5.,6.]))) as sig:
            q,e,sp,sh=invert_owned_rgb(*rgb,{},mask)
        self.assertEqual(sig.call_args.args[0].shape,(2,))
        np.testing.assert_array_equal(sp[mask],[3.,4.])
        np.testing.assert_array_equal(sh[mask],[5.,6.])
        self.assertTrue(np.isnan(sp[~mask]).all())

    def test_notebook_passes_fixed_owner_mask(self):
        n=json.loads((Path(__file__).parents[1]/'gneiss_rgb.ipynb').read_text())
        node=next(node for c in n['cells'] if c['cell_type']=='code'
                  for node in ast.parse(''.join(c['source'])).body
                  if isinstance(node,ast.FunctionDef) and node.name=='invert_site')
        lookup,rgb=inversion_case()
        owner=np.array([[1,2,1],[2,0,1]])
        ns=dict(np=np,process_channel=lambda site,channel,time:(dict(zip(('red','green','blue'),rgb))[channel],{}),
                decimation=1,lookup_by_site={'ARV':lookup},owner_site=owner,site_codes={'ARV':1},
                calculate_conductances=False,invert_owned_rgb=invert_owned_rgb)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<notebook>','exec'),ns)
        result=ns['invert_site']('ARV',None,True)
        self.assertTrue(np.isnan(result['q'][owner!=1]).all())
        self.assertEqual(result['inversion_pixel_selection'],'fixed_owner_only')

if __name__=='__main__':unittest.main()
