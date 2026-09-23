"""Display names and display order shared by all figures and tables of the manuscript."""

# Study findings by organ, in the order used in every display.
FINDINGS = {
    "lung": {
        "aspiration_squames": "Aspiration of squames",
        "aspiration_meconium": "Aspiration of meconium",
        "bronchopneumonia": "Bronchopneumonia",
        "pneumonitis": "Pneumonitis",
        "hyaline_membranes": "Hyaline membranes",
    },
    "liver": {
        "steatosis": "Steatosis",
        "hemozoin_pigment": "Hemozoin pigment",
    },
}

# Candidate findings that are not study findings. The organ is given in its own column,
# so organ prefixes of the schema names are omitted.
OTHER_FINDINGS = {
    "aspiration_gi_contents": "Aspiration of gastrointestinal contents",
    "atelectasis": "Atelectasis",
    "atypical_cells_tumor": "Atypical cells or tumor",
    "bacterial_colonies": "Bacterial colonies",
    "bronchiolitis": "Bronchiolitis",
    "calcification": "Calcification",
    "cholestasis": "Cholestasis",
    "diffuse_alveolar_damage": "Diffuse alveolar damage",
    "ductular_proliferation": "Ductular proliferation",
    "edema": "Edema",
    "extramedullary_hematopoiesis": "Extramedullary hematopoiesis",
    "fibrin": "Fibrin",
    "fibrosis": "Fibrosis",
    "fungal": "Fungal organisms",
    "giant_cell_transformation": "Giant cell transformation",
    "granulomas": "Granulomas",
    "hemozoin_pigment": "Hemozoin pigment",
    "hepatic_inflammation_lobular": "Lobular hepatic inflammation",
    "hepatic_inflammation_portal": "Portal hepatic inflammation",
    "hepatic_inflammation_sinusoidal": "Sinusoidal hepatic inflammation",
    "hepatocellular_swelling": "Hepatocellular swelling",
    "increased_alveolar_macrophages": "Increased alveolar macrophages",
    "intraerythrocytic_parasites": "Intraerythrocytic parasites",
    "liver_congestion": "Congestion",
    "liver_hemosiderin": "Hemosiderin",
    "liver_necrosis": "Necrosis",
    "lung_congestion": "Congestion",
    "lung_hemorrhage": "Hemorrhage",
    "lung_hemosiderin": "Hemosiderin",
    "lung_intravascular_leukocytosis": "Intravascular leukocytosis",
    "lung_necrosis": "Necrosis",
    "organizing_pneumonia": "Organizing pneumonia",
    "pigment_unspecified": "Unspecified pigment",
    "pleuritis": "Pleuritis",
    "pneumocystis": "Pneumocystis",
    "pneumocyte_hyperplasia": "Pneumocyte hyperplasia",
    "preterm_lung": "Preterm lung",
    "sinusoidal_dilation": "Sinusoidal dilation",
    "thrombosis": "Thrombosis",
    "viral_inclusions": "Viral inclusions",
}

ORGANS = {"lung": "Lung", "liver": "Liver"}

SITES = {
    "BD": "Bangladesh",
    "ET": "Ethiopia",
    "KE": "Kenya",
    "ML": "Mali",
    "MZ": "Mozambique",
    "SL": "Sierra Leone",
    "ZA": "South Africa",
}

def site_code(value):
    """The code of a site written as a code or as its archive name (``Sierra_Leone``)."""
    if value in SITES:
        return value
    codes = {name: code for code, name in SITES.items()}
    name = str(value).replace("_", " ")
    if name not in codes:
        raise ValueError(f"unknown site: {value}")
    return codes[name]


SLIDE_SOURCES = {"SITE": "Site", "CPL": "Central laboratory"}

ENCODERS = {
    "virchow2": "Virchow2",
    "uni_v2": "UNI2-h",
    "conch_v15": "CONCH v1.5",
    "hoptimus0": "H-optimus-0",
    "hoptimus1": "H-optimus-1",
    "champs_mits": "In-domain encoder",
}

AGGREGATORS = {
    "meanpool": "Mean pooling",
    "abmil": "ABMIL",
    "clam_mb": "CLAM-MB",
    "acmil": "ACMIL",
    "transmil": "TransMIL",
    "mammil": "MamMIL",
}

NOT_APPLICABLE = "–"


def study_findings():
    """(organ, finding) of every study finding, lung before liver, in display order."""
    return [(organ, finding) for organ, findings in FINDINGS.items() for finding in findings]


def finding_name(finding, organ):
    """The display name of a finding in an organ group; unnamed findings raise ``KeyError``."""
    study_findings = FINDINGS.get(organ, {})
    if finding in study_findings:
        return study_findings[finding]
    if finding in OTHER_FINDINGS:
        return OTHER_FINDINGS[finding]
    raise KeyError(f"no display name for the finding {finding!r} ({organ})")
