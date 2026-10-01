import sys

from .. import xla

xla.ensure_libdevice(verbose=False)       # before tensorflow is imported (see ecgr/xla.py)

from .cli import main                      # noqa: E402

sys.exit(main())
