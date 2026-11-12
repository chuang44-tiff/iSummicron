# tests/fixtures/real/

Cached real-data fixtures for offline real-data probes.

Empty on purpose. A maintainer may copy ONE small real frame into this
directory for offline re-running of a real-data probe.

## Rules

- **Read-only at runtime.** Tests and probes read from this directory;
  they do not write here.
- **Small frames only.** A single 2-3 MB JPG is enough for a probe. Do
  not mirror the whole corpus here.
- **Gitignored.** See `.gitignore` — real corpus frames must never reach
  GitHub.
- **Privacy scrub.** Reports must refer to fixture
  frames by index or hash, never by original filename.

## Related

- `.gitignore` — excludes `tests/fixtures/real/*.jpg`, `*.JPG`, `*.dng`,
  `*.DNG`, `*.heic`, `*.HEIC`
