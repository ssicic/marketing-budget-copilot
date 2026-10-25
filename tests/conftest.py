import warnings

import mbc  # noqa: F401  (sets PyTensor compiler flags before pymc is imported)

warnings.filterwarnings("ignore", category=FutureWarning, module="arviz")
