"""Keep the actual RGB notebook's validity gate consistent with partial results."""
import ast
import json
from pathlib import Path

import numpy as np

from asispectralinversion.ownership import combine_owned_products


def test_rgb_notebook_retains_q_when_e0_fails():
    notebook = Path(__file__).resolve().parents[1] / 'gneiss_rgb.ipynb'
    nodes = []
    for cell in json.loads(notebook.read_text())['cells']:
        if cell['cell_type'] == 'code':
            nodes.extend(node for node in ast.parse(''.join(cell['source'])).body
                         if isinstance(node, ast.FunctionDef) and node.name == 'invert_site')
    assert len(nodes) == 1
    q = np.array([[12., np.nan, np.nan]])
    e0 = np.array([[np.nan, 500., np.nan]])
    missing = np.full(q.shape, np.nan)
    owner = np.ones(q.shape, dtype=np.uint8)
    ns = dict(np=np, decimation=1, calculate_conductances=True,
              owner_site=owner, site_codes={'VEE': 1}, lookup_by_site={'VEE': {}},
              process_channel=lambda *args: (np.ones(q.shape), {}),
              invert_owned_rgb=lambda *args, **kwargs: (q, e0, missing, missing, np.zeros(q.shape, dtype="u1")))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(notebook), 'exec'), ns)
    product = ns['invert_site']('VEE', None)
    np.testing.assert_array_equal(product['valid'], [[True, True, False]])
    combined = combine_owned_products({'VEE': product}, owner, {'VEE': 1})
    np.testing.assert_array_equal(combined['q'], q)
    np.testing.assert_array_equal(combined['e0'], e0)
    np.testing.assert_array_equal(combined['source_site'], [[1, 1, 0]])
    assert np.isnan(combined['sigp']).all()
