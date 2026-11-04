# Vendored: LTAE2d (Lightweight Temporal Attention Encoder, 2d variant)

- Source: `VSainteuf/utae-paps` @ `987874e27a98bb399277f2d55eb010744d26f881`
  (latest master, "loosen reqs"), path `src/backbones/{ltae.py,positional_encoding.py}`.
- License: MIT (see `ltae/LICENSE`, Copyright (c) 2019 VSainteuf).
- Vendored on 2026-11-04 because the deployment cannot reach GitHub (3080 syncs
  via git bundle; Colab runs from a bundle), so a path-based reference like
  `anysat-src` cannot be resolved on every host. Files are copied unmodified
  except the single import line `src.backbones.positional_encoding` ->
  `lim_worker.vendor.ltae.positional_encoding` so the package is
  self-contained inside the lim worker.
- Do not edit the vendored modules; if the upstream changes, re-vendor with a
  new commit pin and update this file.
