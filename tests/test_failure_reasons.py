import numpy as np
from asispectralinversion.inversion import e0_interp_general
from asispectralinversion.ownership import invert_owned_rgb, combine_owned_products
from test_inversion_lookup import inversion_case


def test_e0_reasons_preserve_results():
    table = np.array([[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]])
    q = np.array([1., 2., 3.]); energy = np.array([100., 200., 300.])
    ratios = np.array([1.5, 4., 1.5]); inferred = np.array([1.5, 1.5, .5])
    old = e0_interp_general(table, q, energy, ratios, inferred)
    new, codes = e0_interp_general(table, q, energy, ratios, inferred, return_reasons=True)
    np.testing.assert_array_equal(old, new)
    np.testing.assert_array_equal(codes, [0, 8, 7])
    _, codes = e0_interp_general(np.array([[1.,1.,1.],[3.,3.,3.],[1.,1.,1.],[3.,3.,3.]]),
                                 q, np.arange(4), np.array([2.]), np.array([1.5]), return_reasons=True)
    assert codes[0] == 9


def test_owned_reasons_preserve_values_and_failed_owner_codes():
    lookup, rgb = inversion_case()
    rgb = [a.copy() for a in rgb]
    rgb[0][0,0] = np.nan
    rgb[1][0,1] = -1
    rgb[2][0,2] = 1e20
    owner = np.ones(rgb[0].shape, dtype=bool)
    owner[1,0] = False
    old = invert_owned_rgb(*rgb, lookup, owner, False)
    new = invert_owned_rgb(*rgb, lookup, owner, False, return_reasons=True)
    for a,b in zip(old[:2],new[:2]):np.testing.assert_array_equal(a,b)
    assert new[4][0,0] == 2
    assert new[4][0,1] == 3
    assert new[4][0,2] == 5
    assert new[4][1,0] == 1
    p=dict(q=new[0],e0=new[1],valid=np.isfinite(new[0])|np.isfinite(new[1]),failure_reason=new[4])
    merged=combine_owned_products({'A':p},owner.astype('u1'),{'A':1},False)
    np.testing.assert_array_equal(merged['failure_reason'],new[4])


def test_notebook_writes_reason_before_completion(tmp_path):
    import ast
    import json
    import h5py
    from pathlib import Path
    book = Path(__file__).resolve().parents[1] / 'gneiss_rgb.ipynb'
    nodes = []
    for cell in json.loads(book.read_text())['cells']:
        if cell['cell_type']=='code':
            nodes.extend(node for node in ast.parse(''.join(cell['source'])).body
                         if isinstance(node, ast.FunctionDef) and node.name=='write_time_result')
    ns={'np':np,'calculate_conductances':False}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<writer>','exec'),ns)
    with h5py.File(tmp_path/'output.h5','w') as f:
        for key in ('q_mW_m2','e0_eV','source_site','failure_reason'):
            f.create_dataset('combined/'+key,shape=(1,1,2))
        f.create_dataset('completed',shape=(1,),dtype='?')
        combined=dict(q=np.array([[12.,np.nan]]),e0=np.array([[np.nan,np.nan]]),
                      source_site=np.array([[1,0]]),failure_reason=np.array([[8,5]],dtype='u1'))
        ns['write_time_result'](f,0,{'combined':combined})
        np.testing.assert_array_equal(f['combined/failure_reason'][0],[[8,5]])
        assert f['completed'][0]
