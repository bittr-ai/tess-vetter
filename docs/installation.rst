Installation
============

Requirements
------------

- Python 3.11 or 3.12
- NumPy, SciPy, Astropy, Pydantic (installed with the base package)

Basic Installation
------------------

Using pip
^^^^^^^^^

Install the base package:

.. code-block:: bash

   pip install bittr-tess-vetter

Or install from source:

.. code-block:: bash

   git clone https://github.com/bittr-ai/bittr-tess-vetter.git
   cd bittr-tess-vetter
   pip install -e .

Using uv (recommended for development)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`uv <https://github.com/astral-sh/uv>`_ is a fast Python package manager that we recommend
for development:

.. code-block:: bash

   git clone https://github.com/bittr-ai/bittr-tess-vetter.git
   cd bittr-tess-vetter
   uv sync --all-extras --group dev

Optional Dependencies
---------------------

The package has several optional dependency groups for extended functionality:

TLS (Transit Least Squares)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For transit searches using Transit Least Squares:

.. code-block:: bash

   pip install "bittr-tess-vetter[tls]"

Fitting (MCMC)
^^^^^^^^^^^^^^

For MCMC fitting utilities (emcee + arviz):

.. code-block:: bash

   pip install "bittr-tess-vetter[fit]"

Batman (physical transit model)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For limb-darkened transit modeling using ``batman``:

.. code-block:: bash

   pip install "bittr-tess-vetter[batman]"

Wotan (detrending)
^^^^^^^^^^^^^^^^^^

For advanced detrending using the Wotan library:

.. code-block:: bash

   pip install "bittr-tess-vetter[wotan]"

LDTK (limb darkening)
^^^^^^^^^^^^^^^^^^^^^

For limb darkening coefficient estimation:

.. code-block:: bash

   pip install "bittr-tess-vetter[ldtk]"

.. note::

   LDTK is GPL-2.0 licensed. This optional dependency is kept separate to maintain
   BSD-3-Clause license compatibility for the core package.

MLX (Apple Silicon acceleration)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For GPU-accelerated detection on Apple Silicon (macOS):

.. code-block:: bash

   pip install "bittr-tess-vetter[mlx]"

TRICERATOPS (FPP)
^^^^^^^^^^^^^^^^^

For false positive probability estimation using TRICERATOPS+:

.. code-block:: bash

   pip install "bittr-tess-vetter[triceratops]"

Exovetter (ModShift/SWEET)
^^^^^^^^^^^^^^^^^^^^^^^^^^

For integration with the external ``exovetter`` package (V11-V12 checks):

.. code-block:: bash

   pip install "bittr-tess-vetter[exovetter]"

All extras
^^^^^^^^^^

To install all optional dependencies:

.. code-block:: bash

   pip install "bittr-tess-vetter[all]"

Development Installation
------------------------

For development, install with the dev dependency group:

Using pip:

.. code-block:: bash

   pip install -e ".[all]"
   pip install pytest ruff mypy

Using uv:

.. code-block:: bash

   uv sync --all-extras --group dev

Platform Support
----------------

- **macOS / Linux**: First-class support. All features work as expected.
- **Windows**: Best-effort support. Some platform-specific features may have limitations:

  - Cache file locking uses ``fcntl`` (Unix-only); graceful fallback on Windows.
  - Network timeouts use ``SIGALRM`` which may not work on all platforms.

Verifying Installation
----------------------

After installation, verify that the package is working:

.. code-block:: python

   import bittr_tess_vetter
   import bittr_tess_vetter.api as btv

   # Check version
   print(bittr_tess_vetter.__version__)

   # List available functions
   print(dir(btv))
