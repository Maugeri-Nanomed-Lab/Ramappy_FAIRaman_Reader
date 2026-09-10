"""Schema-oriented validation rules for FAIRaman files."""


import re
from dataclasses import dataclass
from typing import Any

import numpy as np

# ---------------------------------------------------------------- schema ---
PROJECT_FIELDS = [
    "project_name",
    "project_id",
    "funding",
    "governance_reference",
    "author",
    "author_id",
    "data_license",
    "accessibility",
    "keywords",
]

SAMPLE_INFO_FIELDS = [
    "sample_provenance",
    "sample_type",
    "detailed_sample_type",
    "sample_source",
    "anatomical_site",
    "anatomical_site_code",
    "anatomical_ontology",
    "storage_temperature",
    "processing_method",
    "sample_creation_date",
    "sample_notes",
]

SAMPLE_DONOR_FIELDS = [
    "donor_id",
    "donor_sex",
    "donor_age",
    "diagnosis_code",
    "diagnosis_ontology",
    "diagnosis_notes",
]

SAMPLE_EVENT_FIELDS = ["event_date", "event_description"]

ENTRY_FIELDS = ["title", "experiment_type", "run_type", "start_time", "data_type"]
MEASUREMENT_FIELDS = [
    "exposure_time",
    "exposure_time_units",
    "accumulation_count",
    "substrate",
]
LASER_FIELDS = [
    "wavelength",
    "wavelength_units",
    "power",
    "power_units",
    "filter",
]

NUMERIC_FIELDS = {
    "donor_age",
    "exposure_time",
    "accumulation_count",
    "wavelength",
    "power",
}

PLACEHOLDERS = {"nan", "none", "null", "n/a", "na", "-", "--", "nat"}

ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:?\d{2})?)?$")
ISO_DATE_STRICT = re.compile(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:?\d{2})?)?$")
ORCID = re.compile(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$")
UBERON = re.compile(r"^UBERON:\d{7}$")
ICD10 = re.compile(r"^[A-Z]\d{2}(\.\d{1,2})?$")

SPDX_HINTS = ("CC-BY", "CC BY", "CC0", "MIT", "GPL", "Apache", "http://", "https://")

# vocabolario MIABIS per storage_temperature
MIABIS_STORAGE = {
    "RT",
    "2 to 10",
    "-18 to -35",
    "-60 to -85",
    "LN",
    "Other",
}


@dataclass
class Issue:
    severity: str  # "error" | "warning" | "info"
    where: str
    message: str

    def __str__(self) -> str:
        tag = {"error": "ERRORE", "warning": "AVVISO", "info": "NOTA"}[self.severity]
        return f"[{tag}] {self.where}: {self.message}"


def _txt(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _is_placeholder(value: Any) -> bool:
    return _txt(value).lower() in PLACEHOLDERS


def _check_block(
    data: dict, fields: list[str], where: str, issues: list[Issue]
) -> None:
    for field in fields:
        if field not in data:
            issues.append(
                Issue("warning", f"{where}/{field}", "campo dello schema assente nel file")
            )
            continue
        value = data[field]
        if _is_placeholder(value):
            issues.append(
                Issue(
                    "error",
                    f"{where}/{field}",
                    f"contiene il segnaposto '{_txt(value)}' invece di una stringa vuota; "
                    "un valore mancante deve restare esplicitamente vuoto",
                )
            )
        elif _txt(value) == "":
            issues.append(Issue("info", f"{where}/{field}", "campo presente ma vuoto"))


def validate(ff) -> list[Issue]:
    """Esegue tutti i controlli su un FairamanFile."""
    issues: list[Issue] = []

    # ------------------------------------------------------------ PROJECT --
    project = ff.project
    _check_block(project, PROJECT_FIELDS, "PROJECT", issues)

    orcid = _txt(project.get("author_id"))
    if orcid and not ORCID.match(orcid):
        issues.append(
            Issue("warning", "PROJECT/author_id", f"'{orcid}' non ha il formato ORCID")
        )

    lic = _txt(project.get("data_license"))
    if lic and not any(h.lower() in lic.lower() for h in SPDX_HINTS):
        issues.append(
            Issue(
                "warning",
                "PROJECT/data_license",
                f"'{lic}' non e' un identificatore machine-readable; "
                "lo schema dichiara una licenza leggibile da macchina (SPDX o URI)",
            )
        )

    # ------------------------------------------------------------- SAMPLE --
    sample = ff.sample
    if "sample_id" not in sample:
        issues.append(Issue("warning", "SAMPLE/sample_id", "campo dello schema assente"))
    info = sample.get("SAMPLE_INFO", {})
    donor = sample.get("SAMPLE_DONOR", {})
    event = sample.get("SAMPLE_EVENT", {})

    _check_block(info, SAMPLE_INFO_FIELDS, "SAMPLE/SAMPLE_INFO", issues)
    _check_block(donor, SAMPLE_DONOR_FIELDS, "SAMPLE/SAMPLE_DONOR", issues)
    _check_block(event, SAMPLE_EVENT_FIELDS, "SAMPLE/SAMPLE_EVENT", issues)

    # date
    for where, value in (
        ("SAMPLE/SAMPLE_INFO/sample_creation_date", info.get("sample_creation_date")),
        ("SAMPLE/SAMPLE_EVENT/event_date", event.get("event_date")),
    ):
        text = _txt(value)
        if text and not _is_placeholder(text) and not ISO_DATE_STRICT.match(text):
            hint = ""
            if ISO_DATE.match(text):
                hint = " (usa 'T' come separatore, o solo YYYY-MM-DD)"
            issues.append(
                Issue("warning", where, f"'{text}' non e' ISO 8601{hint}")
            )

    start = _txt(ff.entry.get("start_time"))
    if start and not ISO_DATE_STRICT.match(start):
        issues.append(Issue("warning", "ENTRY/start_time", f"'{start}' non e' ISO 8601"))

    # ontologie
    code = _txt(info.get("anatomical_site_code"))
    onto = _txt(info.get("anatomical_ontology"))
    if code and onto.upper() == "UBERON" and not UBERON.match(code):
        issues.append(
            Issue(
                "warning",
                "SAMPLE/SAMPLE_INFO/anatomical_site_code",
                f"'{code}' non ha la forma UBERON:NNNNNNN",
            )
        )

    dcode = _txt(donor.get("diagnosis_code"))
    donto = _txt(donor.get("diagnosis_ontology"))
    if dcode and donto.upper().replace("-", "") == "ICD10" and not ICD10.match(dcode):
        issues.append(
            Issue(
                "warning",
                "SAMPLE/SAMPLE_DONOR/diagnosis_code",
                f"'{dcode}' non ha la forma di un codice ICD-10",
            )
        )

    temp = _txt(info.get("storage_temperature"))
    if temp and temp not in MIABIS_STORAGE:
        issues.append(
            Issue(
                "info",
                "SAMPLE/SAMPLE_INFO/storage_temperature",
                f"'{temp}' non appartiene al vocabolario MIABIS "
                f"({', '.join(sorted(MIABIS_STORAGE))})",
            )
        )

    # eta' elevate: rischio di re-identificazione
    age = _txt(donor.get("donor_age"))
    try:
        if age and int(float(age)) >= 90:
            issues.append(
                Issue(
                    "warning",
                    "SAMPLE/SAMPLE_DONOR/donor_age",
                    f"eta' {age}: valori >=90 sono potenzialmente identificanti, "
                    "valuta una fascia (>=90) nei file condivisi",
                )
            )
    except ValueError:
        issues.append(
            Issue("warning", "SAMPLE/SAMPLE_DONOR/donor_age", f"'{age}' non e' numerico")
        )

    # donor_id derivato da sample_id
    sid, did = _txt(sample.get("sample_id")), _txt(donor.get("donor_id"))
    if sid and did and len(sid) > 3 and len(did) > 3:
        common = 0
        for a, b in zip(sid, did):
            if a == b:
                common += 1
            else:
                break
        if common >= 4:
            issues.append(
                Issue(
                    "info",
                    "SAMPLE/SAMPLE_DONOR/donor_id",
                    f"'{did}' condivide il prefisso con sample_id '{sid}': "
                    "lo pseudonimo del donatore e' derivabile dall'identificativo del campione",
                )
            )

    # -------------------------------------------------------------- ENTRY --
    entry = ff.entry
    _check_block(entry, ENTRY_FIELDS, "ENTRY", issues)
    _check_block(entry.get("measurement", {}), MEASUREMENT_FIELDS, "ENTRY/measurement", issues)
    laser = entry.get("instrument", {}).get("laser", {})
    _check_block(laser, LASER_FIELDS, "ENTRY/instrument/laser", issues)
    if "name" not in entry.get("instrument", {}):
        issues.append(Issue("warning", "ENTRY/instrument/name", "campo dello schema assente"))

    # campi numerici salvati come stringa
    for where, block in (
        ("SAMPLE/SAMPLE_DONOR", donor),
        ("ENTRY/measurement", entry.get("measurement", {})),
        ("ENTRY/instrument/laser", laser),
    ):
        for field in NUMERIC_FIELDS & set(block):
            value = block[field]
            if isinstance(value, str) and value.strip():
                try:
                    float(value)
                except ValueError:
                    continue
                issues.append(
                    Issue(
                        "info",
                        f"{where}/{field}",
                        f"valore numerico '{value}' memorizzato come stringa",
                    )
                )

    if str(entry.get("@definition", "")).upper() == "NXRAMAN":
        issues.append(
            Issue(
                "info",
                "ENTRY/@definition",
                "il file dichiara conformita' formale a NXraman, ma SAMPLE e PROJECT "
                "stanno fuori da ENTRY e le unita' sono dataset separati anziche' "
                "attributi @units: un validatore NeXus fallirebbe",
            )
        )

    # ------------------------------------------------- canonical data model --
    dm = ff.data_model
    if "coordinate_mode" not in dm:
        issues.append(
            Issue("warning", "ENTRY/data/@coordinate_mode", "attributo assente")
        )

    declared = dm.get("spectral_count")
    if declared is not None:
        try:
            declared = int(declared)
            if declared != ff.n_spectra:
                issues.append(
                    Issue(
                        "error",
                        "ENTRY/data/@spectral_count",
                        f"dichiarati {declared} spettri, nel file ce ne sono {ff.n_spectra}",
                    )
                )
        except (TypeError, ValueError):
            pass

    nx, ny = dm.get("nx"), dm.get("ny")
    shape = ff.map_shape()
    if nx is not None and ny is not None and shape is not None:
        try:
            if (int(ny), int(nx)) != shape:
                issues.append(
                    Issue(
                        "error",
                        "ENTRY/data/@nx,@ny",
                        f"dichiarati ny={ny}, nx={nx} ma il cubo e' {shape[0]}x{shape[1]}",
                    )
                )
            if int(nx) * int(ny) != ff.n_spectra:
                issues.append(
                    Issue(
                        "error",
                        "ENTRY/data",
                        f"nx*ny = {int(nx) * int(ny)} non corrisponde a {ff.n_spectra} spettri",
                    )
                )
        except (TypeError, ValueError):
            pass

    nwn = dm.get("n_wavenumbers")
    if nwn is not None:
        try:
            if int(nwn) != len(ff.spectral_axis):
                issues.append(
                    Issue(
                        "error",
                        "ENTRY/data/@n_wavenumbers",
                        f"dichiarati {nwn} numeri d'onda, l'asse ne contiene "
                        f"{len(ff.spectral_axis)}",
                    )
                )
        except (TypeError, ValueError):
            pass

    validated = dm.get("coordinate_validated")
    source = _txt(dm.get("coordinate_source"))
    if validated is False or str(validated).lower() == "false":
        issues.append(
            Issue(
                "info",
                "ENTRY/data/@coordinate_validated",
                f"coordinate non validate (source: '{source or 'non dichiarata'}'): "
                "le posizioni sono ricostruite, non misurate",
            )
        )

    warn = _txt(dm.get("geometry_warning"))
    if warn:
        issues.append(Issue("warning", "ENTRY/data/@geometry_warning", warn))

    if str(dm.get("reshape_applied")).lower() == "true":
        issues.append(
            Issue(
                "info",
                "ENTRY/data/@reshape_applied",
                f"la geometria e' stata ricostruita (source: {_txt(dm.get('reshape_source'))})",
            )
        )

    # asse spettrale
    axis = ff.spectral_axis
    if axis.size > 1:
        raw_desc = ff.entry.get("_raw_axis_descending")
        if raw_desc:
            issues.append(
                Issue(
                    "info",
                    "ENTRY/data/raman_shift",
                    "asse memorizzato in ordine decrescente",
                )
            )
        steps = np.diff(axis)
        if steps.size and np.ptp(steps) / max(abs(np.median(steps)), 1e-12) > 0.05:
            issues.append(
                Issue(
                    "info",
                    "ENTRY/data/raman_shift",
                    "passo dell'asse spettrale non uniforme",
                )
            )

    # dati
    data = np.asarray(ff.obj.data)
    if not np.isfinite(data).all():
        n_bad = int((~np.isfinite(data)).sum())
        issues.append(
            Issue("error", "ENTRY/data/intensity", f"{n_bad} valori non finiti (NaN/inf)")
        )

    # immagini ausiliarie
    for name, img in ff.aux_images.items():
        if img.extent is None:
            issues.append(
                Issue(
                    "warning",
                    f"ENTRY/auxiliary/{name}",
                    "immagine priva di calibrazione spaziale: non puo' essere "
                    "sovrapposta alla mappa",
                )
            )

    order = {"error": 0, "warning": 1, "info": 2}
    return sorted(issues, key=lambda i: (order[i.severity], i.where))


def summary(issues: list[Issue]) -> str:
    counts = {"error": 0, "warning": 0, "info": 0}
    for issue in issues:
        counts[issue.severity] += 1
    return (
        f"{counts['error']} errori, {counts['warning']} avvisi, {counts['info']} note"
    )
