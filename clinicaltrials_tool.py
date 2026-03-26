"""
ClinicalTrials.gov API v2 client (no API key required).
Searches for clinical trials by condition, intervention, or keyword.
"""

import requests
from typing import Any


BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

# Fields we care about — keeps response payload small
FIELDS = [
    "NCTId",
    "BriefTitle",
    "OverallStatus",
    "Phase",
    "Condition",
    "InterventionName",
    "BriefSummary",
    "EligibilityCriteria",
    "MinimumAge",
    "MaximumAge",
    "Sex",
    "LeadSponsorName",
    "StartDate",
    "PrimaryCompletionDate",
    "LocationFacility",
    "LocationCity",
    "LocationCountry",
]


def search_clinical_trials(
    condition: str,
    status: str = "RECRUITING",
    max_results: int = 5,
    intervention: str = "",
) -> list[dict]:
    """
    Search ClinicalTrials.gov for studies matching a condition.

    Args:
        condition:    Disease or condition to search for (e.g. "lung cancer")
        status:       Trial status filter. One of: RECRUITING, ACTIVE_NOT_RECRUITING,
                      COMPLETED, NOT_YET_RECRUITING, or ALL (no filter).
        max_results:  Number of trials to return (max 20).
        intervention: Optional drug/intervention name to narrow results.

    Returns:
        List of trial dicts with key fields extracted.
    """
    condition = (condition or "").strip()
    if not condition:
        return []

    max_results = max(1, min(max_results, 20))

    params: dict[str, Any] = {
        "format": "json",
        "query.cond": condition,
        "pageSize": max_results,
        "countTotal": "true",
    }

    if intervention:
        params["query.intr"] = intervention.strip()

    if status and status.upper() != "ALL":
        params["filter.overallStatus"] = status.upper()

    try:
        r = requests.get(BASE_URL, params=params, timeout=15)
        r.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"ClinicalTrials.gov search failed (network): {e}") from e

    try:
        data = r.json()
    except Exception as e:
        raise RuntimeError(f"ClinicalTrials.gov search failed (parse): {e}") from e

    studies = data.get("studies") or []
    results = []
    for study in studies:
        parsed = _parse_study(study)
        if parsed:
            results.append(parsed)

    return results


def _get(d: Any, *keys, default: str = "") -> str:
    """Safely traverse nested dicts and return a string value."""
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key)
    if d is None:
        return default
    if isinstance(d, list):
        return "; ".join(str(i) for i in d if i)
    return str(d).strip()


def _parse_study(study: dict) -> dict | None:
    """Extract key fields from a ClinicalTrials.gov v2 study record."""
    try:
        proto = study.get("protocolSection") or {}
        id_mod = proto.get("identificationModule") or {}
        status_mod = proto.get("statusModule") or {}
        design_mod = proto.get("designModule") or {}
        desc_mod = proto.get("descriptionModule") or {}
        elig_mod = proto.get("eligibilityModule") or {}
        sponsor_mod = proto.get("sponsorCollaboratorsModule") or {}
        conditions_mod = proto.get("conditionsModule") or {}
        interventions_mod = proto.get("armsInterventionsModule") or {}
        contacts_mod = proto.get("contactsLocationsModule") or {}

        nct_id = _get(id_mod, "nctId")
        if not nct_id:
            return None

        # Interventions — flatten list of names
        interventions = interventions_mod.get("interventions") or []
        intervention_names = "; ".join(
            _get(i, "name") for i in interventions if isinstance(i, dict)
        )

        # Locations — first 3
        locations = contacts_mod.get("locations") or []
        location_strs = []
        for loc in locations[:3]:
            if not isinstance(loc, dict):
                continue
            parts = [
                _get(loc, "facility"),
                _get(loc, "city"),
                _get(loc, "country"),
            ]
            loc_str = ", ".join(p for p in parts if p)
            if loc_str:
                location_strs.append(loc_str)

        # Eligibility criteria — trim to 600 chars
        criteria = _get(elig_mod, "eligibilityCriteria")
        if len(criteria) > 600:
            criteria = criteria[:597] + "..."

        return {
            "nct_id": nct_id,
            "title": _get(id_mod, "briefTitle"),
            "status": _get(status_mod, "overallStatus"),
            "phase": "; ".join(design_mod.get("phases") or []),
            "conditions": "; ".join(conditions_mod.get("conditions") or []),
            "interventions": intervention_names,
            "brief_summary": _get(desc_mod, "briefSummary")[:400] + "..."
                if len(_get(desc_mod, "briefSummary")) > 400
                else _get(desc_mod, "briefSummary"),
            "eligibility_criteria": criteria,
            "min_age": _get(elig_mod, "minimumAge"),
            "max_age": _get(elig_mod, "maximumAge"),
            "sex": _get(elig_mod, "sex"),
            "sponsor": _get(sponsor_mod, "leadSponsor", "name"),
            "start_date": _get(status_mod, "startDateStruct", "date"),
            "primary_completion": _get(status_mod, "primaryCompletionDateStruct", "date"),
            "locations": location_strs,
            "url": f"https://clinicaltrials.gov/study/{nct_id}",
        }
    except (KeyError, TypeError, AttributeError):
        return None


def format_trials_for_claude(trials: list[dict]) -> str:
    """Format trial list into a string Claude can reason over."""
    if not trials:
        return "No clinical trials found matching those criteria."

    lines = []
    for i, t in enumerate(trials, 1):
        locations_str = "; ".join(t.get("locations") or []) or "N/A"
        block = [
            f"[Trial {i}]",
            f"NCT ID: {t.get('nct_id') or 'N/A'}",
            f"Title: {t.get('title') or 'N/A'}",
            f"Status: {t.get('status') or 'N/A'}",
            f"Phase: {t.get('phase') or 'N/A'}",
            f"Condition(s): {t.get('conditions') or 'N/A'}",
            f"Intervention(s): {t.get('interventions') or 'N/A'}",
            f"Sponsor: {t.get('sponsor') or 'N/A'}",
            f"Age Range: {t.get('min_age') or 'N/A'} – {t.get('max_age') or 'N/A'}",
            f"Sex: {t.get('sex') or 'N/A'}",
            f"Start Date: {t.get('start_date') or 'N/A'}",
            f"Primary Completion: {t.get('primary_completion') or 'N/A'}",
            f"Locations: {locations_str}",
            f"Summary: {t.get('brief_summary') or 'N/A'}",
            f"Eligibility Criteria: {t.get('eligibility_criteria') or 'N/A'}",
            f"URL: {t.get('url') or 'N/A'}",
            "",
        ]
        lines.append("\n".join(block))

    return "\n".join(lines).strip()


__all__ = ["search_clinical_trials", "format_trials_for_claude"]