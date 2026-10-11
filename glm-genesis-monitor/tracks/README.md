Optional combined invest + best-track files, named `BEST_b<atcf id>.nc`
(for example `BEST_bal092021.nc`, from the paper's Dryad `Misc_files` folder).

When a file for an id is here, "GLM add archive case" uses its positions
instead of NHC's b-decks. Variables read: `initialization_time`, `track_lat`,
`track_lon`, `track_type`, and `track_maxwind` / `track_mslp` if present.
