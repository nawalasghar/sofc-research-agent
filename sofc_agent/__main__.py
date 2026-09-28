"""Allow ``python -m sofc_agent "question"``."""
import sys

from .cli import main

sys.exit(main())
