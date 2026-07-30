# GNEISS data layout

Paths may live anywhere; `config.toml` points to their roots. The current
notebook expects this organization beneath those roots:

```text
image_root/
├── green/{ARV,VEE,BVR}/...
└── red/6300/{ARV,VEE,BVR}/...

starmap_root/
├── green/{ARV,VEE,BVR}/...
├── red/6300/{ARV,VEE,BVR}/...
└── blue/VEE/...                 # optional

glow_root/
├── ARV/
│   └── airglow/
├── VEE/
│   └── airglow/
└── BVR/
    └── airglow/

trajectory_root/
└── GNEISS/
    ├── 36397_GPS_Time_Export_01.csv
    └── 36398_GPS_Time_Export_00.csv
```

The GLOW directory must contain the generated `I*.bin`, `eta*.bin`,
`edens3d`, `hall3d`, and `ped3d` products expected by
`load_lookup_tables_directory`.

Large mission data should be published in a versioned archive with a checksum
manifest. Do not commit full TIFF stacks, lookup tables, or inversion cubes to
ordinary Git.
