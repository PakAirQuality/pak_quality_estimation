# Minting the Zenodo DOI

This repository is set up for the **GitHub → Zenodo** release integration. Creating a GitHub
release produces a citable DOI whose archive includes the code, the 2025 validation table, and
the gridded prediction product (`data/gridded_predictions_2024_2025/`). Deposit metadata is
controlled by `.zenodo.json`.

> **Order matters:** the repo must be **public** and the Zenodo toggle must be **ON _before_**
> you create the release, or Zenodo will not archive it.

## Steps

1. **Coauthor sign-off**, then make the repository public:
   ```sh
   gh repo edit PakAirQuality/pak_quality_estimation --visibility public
   ```
   (or GitHub → Settings → General → Danger Zone → Change visibility.)

2. Go to **https://zenodo.org → Log in → Log in with GitHub** and authorize Zenodo.
   Because the repo is owned by the **PakAirQuality** organization, an org owner may need to
   grant Zenodo OAuth access to the org (GitHub → Settings → Applications → Authorized OAuth
   Apps, or approve when prompted).

3. In Zenodo, open **Account → GitHub** (https://zenodo.org/account/settings/github/) and flip
   the **`pak_quality_estimation`** switch **ON**. (If it doesn't appear, click *Sync* / re-check
   after making the repo public.)

4. On GitHub, **draft a new release**:
   ```sh
   gh release create v1.0.0 --repo PakAirQuality/pak_quality_estimation \
     --title "v1.0.0 — code & data for the ACP submission" \
     --notes "Support-aware daily PM2.5 estimation pipeline, held-out 2025 validation table, and 2024–2025 daily gridded 0.1° PM2.5 fields. Accompanies the ACP manuscript."
   ```
   (or GitHub → Releases → Draft a new release; tag `v1.0.0`.)

5. Zenodo detects the release within ~a minute and mints **two DOIs**:
   - a **version DOI** (points to `v1.0.0` specifically), and
   - a **concept DOI** (always resolves to the latest version).
   **Use the concept DOI in the paper.** Find them at
   https://zenodo.org/account/settings/github/ (or your Zenodo uploads).

6. **Send me the concept DOI** (e.g. `10.5281/zenodo.XXXXXXX`). I will replace the `[TBD]`
   placeholders in the manuscript availability statement, `CITATION.cff`, `README.md`, and
   `DATA.md`, add the DOI badge, and rebuild the manuscript.

## Optional before releasing

- **ORCIDs:** add them to `.zenodo.json` creators for a richer record, e.g.
  `{"name": "Ahmad, Rehan", "affiliation": "...", "orcid": "0000-0002-1825-0097"}`.
  (Omitted by default — Zenodo validates ORCID format, so only add real ones.)
- **Reserve a DOI early:** if you need the DOI string *before* going public (e.g. for the
  preprint), you can instead create a manual upload on Zenodo, attach a snapshot, and click
  *Reserve DOI* — but then you manage the deposit by hand rather than via releases.
