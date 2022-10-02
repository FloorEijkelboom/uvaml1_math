
.. _Derivative Rules:

Derivative Rules
================

.. toctree::
   :maxdepth: 2
   :caption: Contents



Here we provide the basic derivative rules.
We separated them into (1) derivatives of specific functions, and (2) properties of the derivatives of combined functions.

Standard derivatives
....................

- :math:`f(x) = c \implies f'(x) = 0`,
- :math:`f(x) = x^n \implies f'(x) = nx^{n-1}`,
- :math:`f(x) = a^x \implies f'(x) = a^x \log a`, and hence :math:`f(x) = e^x \implies f'(x) = e^x`,
- :math:`f(x) = \log_b x \implies f'(x) = \frac{1}{\log (b) \cdot x}`, and hence :math:`f(x) = \log x \implies f'(x) = \frac{1}{x}`,
- :math:`f(x) = \sin x \implies f'(x) = \cos x`,
- :math:`f(x) = \cos x \implies f'(x) = - \sin x`.

Moreover, it is useful to remember special cases of the second rule, e.g. :math:`f(x) = x \implies f'(x) = 1`,
:math:`f(x) = ax \implies f'(x) = a`, and  :math:`f(x) = \sqrt{x} \implies f'(x) = \frac{1}{2\sqrt{x}}`.

Derivative of combined functions
................................

- :math:`(c \cdot f)'(x) = c \cdot f'(x)`,
- :math:`(f + g)'(x) = f'(x) + g'(x)`,
- :math:`(f \cdot g)'(x) = f'(x) \cdot g(x) + f(x) \cdot g'(x)`,
- :math:`(\frac{f}{g})'(x) = \frac{f'(x) g(x) - f(x)g'(x)}{g(x)^2}`,
- :math:`(f \circ g)'(x) = (f' \circ g)(x) \cdot g'(x)`.