# Data-source policy and provenance

## AAVSO

- Automated: yes
- Credentials: no
- Route: public VSX/AID light-curve request (`view=api.object`, CSV observation payload inside the XML object response)
- Target input: common name, designation or identifier accepted by AAVSO
- Recommended use: homogeneous passband, e.g. Johnson V
- Failure behavior: no observations / unrecognized response => explicit `TargetNotFound` or `RemoteServiceError`

## OGLE-IV OCVS

- Automated: yes
- Credentials: no
- Route: public static OCVS `phot/<band>/<OGLE-ID>.dat` files
- Target input: exact OGLE ID preferred; optional common-name coordinate resolution via CDS Sesame plus OCVS identification tables
- Native columns: HJD, magnitude, uncertainty
- Failure behavior: missing object/band file or no positional match => explicit `TargetNotFound`

## TESS / MAST

- Automated: yes
- Credentials: no for standard public products
- Client: Lightkurve (optional install extra: `pip install -e ".[tess]"`)
- Target input: common target name/coordinates understood by Lightkurve/MAST
- Photometry: flux converted to relative magnitude with propagated uncertainty
- Failure behavior: no products => explicit `TargetNotFound`

## ASAS-SN

- Automated in this project: no
- Reason: Sky Patrol V2 Python-client documentation describes username/token access; legacy public interface explicitly requests CAPTCHA and warns against scripted automation.
- Supported path: export/download according to ASAS-SN's allowed interface, then analyze using `--source local`.

## Local files

The local reader accepts numeric CSV/ASCII columns specified by the user. It does not infer unknown time systems or bandpasses. Record these explicitly with `--time-system` and `--band`.
