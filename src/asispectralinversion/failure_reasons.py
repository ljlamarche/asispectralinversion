"""Version 1 RGB failure codes: first failed stage, with conductance checked last."""
FAILURE_REASONS = {
    0: 'Q/E0 and requested conductances finite',
    1: 'Outside fixed ownership coverage',
    2: 'Nonfinite RGB input',
    3: 'Nonpositive RGB input',
    4: 'Blue below lookup brightness range',
    5: 'Blue above lookup brightness range',
    6: 'Q lookup failed within brightness range',
    7: 'Q outside E0 interpolation support',
    8: 'Red/green ratio has no crossing at Q',
    9: 'E0 ambiguous: multiple crossings',
    10: 'E0 numerical or table failure',
    11: 'Conductance missing with finite Q/E0',
    255: 'Not recorded / unfinished',
}
