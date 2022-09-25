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

**Definition 2.1**.
A vector space over a field :math:`\F` is a set :math:`V` with two binary operations:

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

Similar to the previous section, we will first informally motivate the definition of a basis, and only
then formalize it.


Informal definition
...................

The basis of a vector space provides an organized way to represent any vector in that space. As a
simple example, let’s think about possible colors produced by a pixel on the screen you are reading
this on. Every pixel consists of 3 lighting elements: red, green, and blue, and every other color
can be reproduced by varying the intensities of each of these colors. Since the lighting elements are
independent, we can represent an arbitrary color :math:`\vc` as follows:

.. math::
    \vc = \matrix{ r_i \\ g_i \\ b_i}

where :math:`r_i / g_i / b_i` denote the intensities of the red/green/blue light. By tuning these three numbers,
we can represent any color reproducible by our monitor. Now, let’s rewrite this more suggestively:

.. math::
    \vc =
    \matrix{r_i \\ 0 \\ 0} + \matrix{0 \\ g_i \\ 0} + \matrix{0 \\ 0 \\ b_i} =
    r_i \cdot \matrix{1 \\ 0 \\ 0} + g_i \cdot \matrix{0 \\ 1 \\ 0} + b_i \cdot \matrix{0 \\ 0 \\ 1}


We can now also give unique names to the column vectors and write the previous expression as follows:

.. math::
    \vc = r_i \cdot \vr + g_i \cdot \vg + b_i \cdot \vb

where

.. math::
    \vr = \matrix{1 \\ 0 \\ 0}, \vg = \matrix{0 \\ 1 \\ 0}, \vb = \matrix{0 \\ 0 \\ 1}

The color vector :math:`\vc` has been written as a weighted sum of other vectors, and this is called a linear
combination. Since we can uniquely represent any vector (color) using these three vectors, we say
that vectors :math:`\vr`, :math:`\vg` and :math:`\vb` form a basis. A basis can be thought of as a set of independent vectors
whose linear combination can uniquely represent any vector. The basis of this form, where the n-th
basis vector has 1 as the n-th element and 0 otherwise, is called a canonical basis and is the most
simple form of basis.

It is worth investigating why we impose the condition that the basis vectors need to be independent,
and what independence means. For simplicity, imagine that a purple color :math:`\vp` can be expressed as a
linear combination of red and blue:

.. math::
    \vp = \frac{1}{\sqrt{2}} \vr + \frac{1}{\sqrt{2}} \vb = \frac{1}{\sqrt{2}} \matrix{1 \\ 0 \\ 1}

Now, let’s imagine that we add the color purple to our basis, so our basis now consists of :math:`\{\vr, \vg, \vb, \vp\}`
(you have 4 lights in your pixel now). Your friend told you about an imaginary color durple :math:`\vd`, and
they told you that they use the :math:`\{\vr, \vg, \vb\}` basis for their pixels. They represent the color durple as
follows:

.. math::
    \vd = \frac{1}{\sqrt{2}} \matrix{-1 \\ 0 \\1}

In order to reproduce this color, you start turning the 4 knobs of color intensities (one for each
different color in your pixel). First, you do not use your purple color, and you just stick with red
and blue. You find that the following combination reproduces durple:

.. math::
    \vd = -\frac{1}{\sqrt{2}} \vr + \frac{1}{\sqrt{2}} \vb
    = -\frac{1}{\sqrt{2}} \matrix{1 \\ 0 \\ 0} + \frac{1}{\sqrt{2}} \matrix{0 \\ 0 \\ 1}
    = \frac{1}{\sqrt{2}} \matrix{-1 \\ 0 \\1}

However, you start turning the purple knob, tune red and blue a bit, and you realize that also the
following combination produces durple:

.. math::
    \vd = -2 \cdot \vr + \vp
    = - 2 \matrix{1 \\ 0 \\ 0} + \frac{1}{\sqrt{2}} \matrix{1 \\ 0 \\ 1}
    = \frac{1}{\sqrt{2}} \matrix{-1 \\ 0 \\1}


This tells us that after adding the purple color to our basis, our representation of the durple color
was no longer unique, i.e. there were multiple ways to produce it. This stems from the fact that we
have added purple to our basis, which was not independent since we were able to write it as a linear
combination of already existing colors (red and blue). An equally valid choice of basis would have
been to remove the color red from our basis set, and simply have :math:`\{\vg, \vb, \vp\}` as our basis.

An intuitive way to think of a basis is as a set of vectors, of which none can be written as a linear
combination of the rest. Formally it can be shown that the number of basis vectors has to equal
the dimension of the vector space. For example, in the example above, the number of basis vectors
was 3, as we had 3-dimensional vectors. If we were to add any more vectors to our basis, we would
necessarily add vectors that are no longer independent of each other, and thus we wouldn’t have
a systematic and unique way to represent arbitrary vectors. If we were to remove any vectors (for
example, have 2 vectors in our basis), we wouldn’t be able to express an arbitrary vector as a linear
combination of the basis vectors, as we would be missing *building blocks*.

Formal definition
.................

In Linear Algebra, the basis of a vector space :math:`V` is formally defined as follows.

**Definition 2.2**.
A basis :math:`B` of a vector space :math:`V` over a field :math:`\F` is a linearly independent subset of :math:`V` that spans :math:`V`. This subset, therefore, has to satisfy the following conditions:

    1. **Linear independence**: For every finite subset :math:`\{\vv_1, . . . , \vv_m\}` of :math:`B`, neither of the m elements can be represented as a linear combination of the rest.
    2. **Spanning property**: For every vector :math:`\vv` in :math:`V`, one can choose scalars :math:`\lambda_1, \dots , \lambda_n` from the field :math:`\F` and :math:`\vv_1, \dots , \vv_n` such that :math:`\vv = \lambda_1 \vv_1 + \dots + \lambda_n \vv_n`.


The first condition states what we discussed above; if we wish to have a basis, we mustn’t be able to represent any of the basis elements by a linear combination of other basis elements. This is required if we wish to uniquely represent every vector using basis vectors. The second condition tells us that we must be able to represent any vector from the vector space using a linear combination of the basis vectors. These two conditions combined lead to the fact that for a basis of a :math:`n`-dimensional vector space we must have exactly :math:`n` linearly independent basis vectors.

Dot product
-----------

So far, we have only seen two operations we can do with vectors: addition of vectors and multiplication of vectors by a scalar.
The two operations combined allowed us to form a definition of a linear combination and basis.

A vanilla vector space does not have any other operations that involve two vectors. However, a
vector space can be equipped with an inner product to form an inner product space. [#f1]_
A dot product [#f2]_ between vectors :math:`v = [a_1, \dots , a_n]^T` [#f3]_ and :math:`\vw = [b_1, \dots , b_n]^T` is defined as follows:

.. math::
    \vv \cdot \vw = a_1 b_1 + \dots + a_n b_n = \sum_{i=1}^n a_i b_i

To see the benefit and the interpretation of the dot product, let’s take a closer look at a case when
we calculate a dot product of a vector with itself:

.. math::
    \vv \cdot \vv = \sum_{i=1}^n a_i a_i = \sum_{i=1}^n a_i^2

What we can see from this is that this corresponds to the squared norm/magnitude of the vector :math:`\vv`.
The usual notation for the norm of a vector is :math:`|| \cdot ||`, so we can write:

.. math::
    || \vv || = \sqrt{\sum_{i=1}^n a_i^2} \Longrightarrow || \vv || = \sqrt{\vv \cdot \vv}

As a simple example, let’s imagine that we have a 2D vector :math:`\vc = \va + \vb`, where :math:`\va = [a, 0]^T` and
:math:`\vb = [0, b]^T`, as shown in the figure below:

.. image:: /figures/linear_algebra/sum_vectors.png
    :alt: sum_of_vectors
    :align: center

If we calculate the dot product of the vector :math:`\vc` with itself, we get:

.. math::
    || \vc ||^2 = a^2 + b^2

which is exactly the Pythagorean theorem in 2D.

Besides being useful for calculating norms of vectors, dot product can be used as a measure of
similarity. If we imagine two :math:`n`-dimensional vectors :math:`\vv` and :math:`\vw`, the angle :math:`\theta` between them can be
calculated using the following formula:

.. math::
    \vv \cdot \vw = || \vv || \ || \vw || \cos \theta

We can divide both sides by the norms of both vectors to get the expression for the cosine of the
angle between the vectors:

.. math::
    \cos \theta = \frac{\vv \cdot \vw}{|| \vv || \ || \vw||}

When the cosine of the angle between two vectors is equal to :math:`1`, the vectors are perfectly aligned
(interpreted as being as similar as possible), and when it is equal to :math:`0`, the vectors are perpendicular
(interpreted as being as different as possible). This can be interpreted as a measure of similarity
(often called the *cosine similarity*), which is often used in many areas, such as Natural Language
Processing (more information with some examples can be found here).

To sum up, we have introduced a new operation we can use to manipulate vectors, the dot product.
It is a useful tool because it allows us to easily calculate the norms of vectors, and also the cosine
similarity between them.

Linear Operators
----------------

Change of Basis
---------------







.. rubric:: Footnotes

.. [#f1] An interested reader can find more information `here <https://en.wikipedia.org/wiki/Inner_product_space>`_.
.. [#f2] Inner product and a dot product are often used interchangeably, although there are subtle differences, refer `here <https://math.stackexchange.com/questions/476738/difference-between-dot-product-and-inner-product?rq=1>`_ for a brief discussion.
.. [#f3] Letter :math:`T` stands for the transpose operation, more information can be found `here <https://en.wikipedia.org/wiki/Transpose>`_.

