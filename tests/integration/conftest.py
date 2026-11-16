"""Import the SDKs once in the outer process so pytester's in-process runs reuse them.

pytester restores ``sys.modules`` after each inner run. Modules imported first inside an
inner run would otherwise be re-imported every time, which is slow and can break
C extensions.
"""

import anthropic  # noqa: F401
import openai  # noqa: F401
