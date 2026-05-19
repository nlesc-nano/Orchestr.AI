import re
import os

source_file = "/Users/ivaninfante/Documents/University/escience/Orchestr.AI/src/orchestr_ai/postprocessing/plotting.py"
with open(source_file, "r") as f:
    content = f.read()

# Determine boundaries
# utils.py: from start to just before generate_uq_plots
utils_start = content.find("def _ideal_colour(")
uq_start = content.find("def generate_uq_plots(")
al_start = content.find("def plot_histogram(")
mlff_start = content.find("def plot_mlff_stats(")

utils_code = """import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import halfnorm, norm, spearmanr
import matplotlib.colors as mcolors

""" + content[utils_start:uq_start].replace("import matplotlib.colors as mcolors\n", "")

uq_code = """import os
import traceback
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .utils import (
    _ideal_colour,
    plot_scalar_metrics,
    plot_coverage_curve,
    plot_sigma_density,
    _pick,
    plot_rmse_rmv_per_bin,
    compute_ence,
    _plot_reliability_gap,
    _plot_zscore_hist_qq_compare,
    plot_swapped_final_tight,
    plot_original_final_tight
)

""" + content[uq_start:al_start]
# Remove old imports from uq_code if they exist
uq_code = re.sub(r'# assume these come from your module.*?# \)', '', uq_code, flags=re.DOTALL)
uq_code = uq_code.replace("import os\nimport traceback\nfrom pathlib import Path\n\nimport numpy as np\nimport matplotlib.pyplot as plt\n", "", 1)


al_code = """import os
import traceback
import numpy as np
import matplotlib.pyplot as plt

""" + content[al_start:mlff_start]

mlff_code = """import os
import traceback
import numpy as np
import matplotlib.pyplot as plt

""" + content[mlff_start:]


plots_dir = "/Users/ivaninfante/Documents/University/escience/Orchestr.AI/src/orchestr_ai/postprocessing/plots"

with open(os.path.join(plots_dir, "utils.py"), "w") as f:
    f.write(utils_code)

with open(os.path.join(plots_dir, "uq.py"), "w") as f:
    f.write(uq_code)

with open(os.path.join(plots_dir, "active_learning.py"), "w") as f:
    f.write(al_code)

with open(os.path.join(plots_dir, "mlff.py"), "w") as f:
    f.write(mlff_code)

with open(os.path.join(plots_dir, "__init__.py"), "w") as f:
    f.write('''from .utils import (
    plot_scalar_metrics,
    plot_coverage_curve,
    plot_sigma_density,
    plot_rmse_rmv_per_bin,
    compute_ence,
    plot_swapped_final_tight,
    plot_original_final_tight,
)
from .uq import generate_uq_plots
from .active_learning import (
    plot_histogram,
    plot_overall_score_vs_clusters,
    plot_rmse_vs_sorted_overall_score,
    plot_error_vs_uncertainty,
    generate_al_influence_plots,
    generate_al_traditional_plots,
)
from .mlff import plot_mlff_stats

__all__ = [
    "plot_scalar_metrics",
    "plot_coverage_curve",
    "plot_sigma_density",
    "plot_rmse_rmv_per_bin",
    "compute_ence",
    "plot_swapped_final_tight",
    "plot_original_final_tight",
    "generate_uq_plots",
    "plot_histogram",
    "plot_overall_score_vs_clusters",
    "plot_rmse_vs_sorted_overall_score",
    "plot_error_vs_uncertainty",
    "generate_al_influence_plots",
    "generate_al_traditional_plots",
    "plot_mlff_stats",
]
''')

# Now overwrite plotting.py
plotting_code = '''"""Plotting utilities compatibility layer.

This module re-exports the plotting functions which have been moved to the `plots` sub-package.
"""
from .plots.utils import (
    plot_scalar_metrics,
    plot_coverage_curve,
    plot_sigma_density,
    plot_rmse_rmv_per_bin,
    compute_ence,
    plot_swapped_final_tight,
    plot_original_final_tight,
)
from .plots.uq import generate_uq_plots
from .plots.active_learning import (
    plot_histogram,
    plot_overall_score_vs_clusters,
    plot_rmse_vs_sorted_overall_score,
    plot_error_vs_uncertainty,
    generate_al_influence_plots,
    generate_al_traditional_plots,
)
from .plots.mlff import plot_mlff_stats

__all__ = [
    "plot_scalar_metrics",
    "plot_coverage_curve",
    "plot_sigma_density",
    "plot_rmse_rmv_per_bin",
    "compute_ence",
    "plot_swapped_final_tight",
    "plot_original_final_tight",
    "generate_uq_plots",
    "plot_histogram",
    "plot_overall_score_vs_clusters",
    "plot_rmse_vs_sorted_overall_score",
    "plot_error_vs_uncertainty",
    "generate_al_influence_plots",
    "generate_al_traditional_plots",
    "plot_mlff_stats",
]
'''
with open(source_file, "w") as f:
    f.write(plotting_code)

print("Split complete!")
