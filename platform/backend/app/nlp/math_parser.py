"""Math expression parsing layer — detect and validate mathematical expressions."""

import re
import logging
from .pipeline import NLPContext

logger = logging.getLogger(__name__)

# ── Lazy imports ──
_sympy_parser = None
_sympy_latex = None
_sympy_simplify = None


def _get_sympy():
    """Return (parse_expr, latex, simplify) or (None, None, None)."""
    global _sympy_parser, _sympy_latex, _sympy_simplify
    if _sympy_parser is None:
        try:
            from sympy.parsing.sympy_parser import parse_expr
            from sympy import latex, simplify
            _sympy_parser = parse_expr
            _sympy_latex = latex
            _sympy_simplify = simplify
        except ImportError:
            logger.warning("sympy not available")
            _sympy_parser = False
            _sympy_latex = False
            _sympy_simplify = False
    if _sympy_parser is False:
        return None, None, None
    return _sympy_parser, _sympy_latex, _sympy_simplify


# ── Common ML formula patterns (informal notation) ──

_ML_FORMULA_PATTERNS = [
    # y = wx + b  (linear model)
    r'[yY]\s*=\s*[wW]\s*\*?\s*[xX]\s*[\+\-]\s*[bB]',
    # L = -log(p) or L = -sum(y*log(p))
    r'[lL]\s*=\s*-?\s*(sum\s*\()?\s*[yY]\s*\*?\s*log\s*\([pP]\)',
    # sigmoid(x) = 1/(1+e^-x)
    r'sigmoid\s*\(\s*[xX]\s*\)',
    # softmax
    r'softmax\s*\(',
    # general equation with = sign and math operators
    r'[a-zA-Z_]\w*\s*=\s*[^,\n]{3,}[\+\-\*/\^]',
]


def _normalize_math_expr(expr_str: str) -> str:
    """Clean up informal math notation for sympy parsing.

    Handles common student notation like:
    - e^-x  -> exp(-x)
    - 2x    -> 2*x
    - log() -> log()   (sympy understands this)
    """
    s = expr_str.strip()
    # Remove LaTeX-specific commands
    s = re.sub(r'\\(frac|sum|nabla|partial|left|right|cdot|times)', ' ', s)
    s = re.sub(r'[{}]', '', s)
    # Replace ^ with ** for exponentiation
    s = s.replace('^', '**')
    # e**(-x) is valid sympy
    s = re.sub(r'\be\*\*', 'exp(', s)
    # Handle implicit multiplication: 2x -> 2*x, but not inside function names
    s = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', s)
    return s


def parse_math_expressions(ctx: NLPContext) -> NLPContext:
    """Parse and validate mathematical expressions using sympy.

    If ctx.has_math is True and ctx.math_expressions has entries:
      - Try sympy.parse_expr() to validate each expression
      - Store simplified form and LaTeX representation
      - Keep unparseable expressions as-is

    Also detect common ML formulas from the message text.
    """
    parse_expr, latex_fn, simplify_fn = _get_sympy()
    if parse_expr is None:
        return ctx

    text = ctx.user_message

    # Step 1: Detect additional ML formula patterns not already caught by regex
    for pattern in _ML_FORMULA_PATTERNS:
        matches = re.findall(pattern, text)
        if matches:
            ctx.has_math = True
            for m in matches:
                expr_str = m if isinstance(m, str) else m[0] if m else ""
                if expr_str and expr_str not in ctx.math_expressions:
                    ctx.math_expressions.append(expr_str)

    # Step 2: Parse existing math_expressions with sympy
    if not ctx.has_math or not ctx.math_expressions:
        return ctx

    parsed_expressions = []
    for expr_str in ctx.math_expressions:
        normalized = _normalize_math_expr(expr_str)
        try:
            parsed = parse_expr(normalized, evaluate=False)
            simplified = simplify_fn(parsed)
            latex_repr = latex_fn(simplified)
            parsed_expressions.append(
                f"{expr_str} [valid: LaTeX={latex_repr}]"
            )
            logger.debug("parse_math: '%s' -> LaTeX: %s", expr_str, latex_repr)
        except Exception:
            # Not parseable by sympy — keep as-is (informal notation)
            parsed_expressions.append(expr_str)
            logger.debug("parse_math: '%s' not parseable — keeping as-is", expr_str)

    ctx.math_expressions = parsed_expressions
    return ctx
