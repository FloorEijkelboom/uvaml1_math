Linear Algebra
==============

Linear algebra serves as a core of most machine learning algorithms that you will encounter throughout the course, as
the majority of objects are represented as vectors and matrices (matrices are called arrays/tensors in `NumPy`/`PyTorch`).
For this reason, we will systematically revise all the essential
concepts such as vectors, matrices, linear operators, and determinants. To intuitively explain certain
concepts, we will use jargon which will be denoted in *italic* font.


.. toctree::
   :maxdepth: 2
   :caption: Contents





Vector spaces
-------------


In order to introduce vector spaces, which is a space where vectors live, we will first try to motivate
its formal definition which will follow later.

Informal definition
...................

Firstly, let’s denote a vector by a bold letter :math:`\vv`. The easiest way to visualize a vector is to associate
it with something familiar. For example, imagine you live on a flat Earth and you’re on a hike and
you wish to send your friends your location. You could, for example, represent your location as a
3D vector:

.. math::
    \vv = \matrix{ x_1 \\ y_1 \\ z_1 }

In this notation, :math:`x_1` and :math:`y_1` are your initial longitude/latitude offset from the bottom of the mountain
(the amount you moved west/east and south/north), while :math:`z_1` might represent your altitude. You
continue your hike, change your longitude/latitude by :math:`x_2` and :math:`y_2`, and climb up by :math:`z_2` to reach the
peak. Then, your new coordinates v are:

.. math::
    \vv' = \matrix{ x_1 + x_2 \\ y_1 + y_2 \\ z_1 + z_2 }

In other words, your new coordinates are simply a sum of the two offsets. Notice that the sum of
two independent offsets produced a new location :math:`\vv'` which also represents a valid location.
Now, imagine you’re going on the same hike, but this time the mountain grew in size by a factor of :math:`\lambda`,
and you wish to come to the same peak as last time.
Intuitively, we can deduce that the you will have to move further by a factor of :math:`\lambda` in each direction,
so the total offset :math:`\vw` will be given by:

.. math::
    \vw =
    \matrix{ \lambda x_1 + \lambda x_2 \\ \lambda y_1 + \lambda y_2 \\ \lambda z_1 + \lambda z_2 }
    = \lambda \matrix{ x_1 + x_2 \\ y_1 + y_2 \\ z_1 + z_2 } = \lambda \vv'

This tells us that even if we multiply our offsets by a number :math:`\lambda`, we can still represent a valid location.

This was a very specific example to aid the visualization of certain properties that define a vector
space, which we will soon define. If we think of a vector as an abstract object which doesn’t
correspond to anything visualizable, then the above-mentioned properties can be thought as the
following. First, we want the sum of two vectors to also be vector from the same space. Second, if
we scale a given vector, we wish that the scaled version is also a part of the same vector space.


Formal definition
.................

We shall now introduce a formal definition of a vector space.

**Definition 2.1**. A vector space over a field :math:`\F` is a set :math:`V` with two binary operations:
    1. Vector addition assigns to any two vectors :math:`\vv` and :math:`\vw` in :math:`V` a third vector in :math:`V` which is denoted by :math:`v + w`.
    2. Scalar multiplication assigns to any scalar :math:`\lambda` in :math:`\F` and any vector :math:`\vv` in :math:`V` a new vector in :math:`V`, which is denoted by :math:`\lambda \vv`.

Vector spaces also have to satisfy 8 axioms, which can be found `here <https://en.wikipedia.org/wiki/Vector\_space#Definition\_and\_basic\_properties>`_
(most of them are trivial and intuitive).
In the definition above, a field :math:`\F` is simply a structure from which we take scalars that we multiply
our vectors by. In most cases, the field will simply be real numbers :math:`\R`.
If we come back to the hiking example, we were dealing with the vectors from :math:`\R^3`, as we had 3 entries
of the vector, and each entry was a real number (coordinates are real numbers).

Summary
.......

Vectors are objects that live in a vector space. It is important to note that a vector space is a space
defined by only two operations with objects: how to add objects and how to scale them. If we know
how to do that, we call that space a vector space. In further sections, we will explore other ways to
utilize and transform vectors besides the addition of vectors and multiplication by a scalar.











Basis
-----



Dot product
-----------

Linear Operators
----------------

Change of Basis
---------------










