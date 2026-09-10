"""Central registry for advanced-analysis plugins.

New algorithms should normally live in their own module and be registered here.
The UI consumes plugin metadata and never needs to know algorithm internals.
"""

from .models import AnalysisPlugin
from .mcr_als import run_mcr_als
from .nfindr import run_nfindr

ANALYSIS_PLUGINS: dict[str, AnalysisPlugin] = {
    "nfindr": AnalysisPlugin(
        key="nfindr",
        label="N-FINDR endmember extraction",
        description=(
            "Estrae pixel/spettri endmember dopo riduzione PCA. L'output principale sono "
            "gli endmember, non una mappa di abbondanza. Utile come inizializzazione di MCR-ALS."
        ),
        runner=run_nfindr,
        optional_packages=("sklearn", "pyspc_unmix"),
    ),
    "mcr_als": AnalysisPlugin(
        key="mcr_als",
        label="MCR-ALS (NNLS)",
        description=(
            "Risoluzione multivariata D = C·Sᵀ con regressione NNLS. Usa gli endmember "
            "N-FINDR come inizializzazione e restituisce spettri e mappe di abbondanza."
        ),
        runner=run_mcr_als,
        optional_packages=("pymcr",),
    ),
}



def register_plugin(plugin: AnalysisPlugin, *, replace: bool = False) -> None:
    """Register an analysis plugin.

    Third-party or experimental modules can call this at application startup.
    """
    if plugin.key in ANALYSIS_PLUGINS and not replace:
        raise KeyError(f"Analysis plugin already registered: {plugin.key}")
    ANALYSIS_PLUGINS[plugin.key] = plugin


def get_plugin(key: str) -> AnalysisPlugin:
    return ANALYSIS_PLUGINS[key]


def iter_plugins():
    return tuple(ANALYSIS_PLUGINS.values())
