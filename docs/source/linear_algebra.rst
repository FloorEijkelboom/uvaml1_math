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

Mappings
........

In linear algebra, besides the operations that involve two vectors (vector addition, dot product),
there are functions (mappings) that take as an input a vector, and output a vector. Let’s denote
this mapping as :math:`f`. Formally, any mapping of this sort can be written as:

.. math::
    f: V \to W

This is a standard mathematical notation which means the following: a function :math:`f` takes as an
input a vector from the vector space :math:`V` and outputs a vector from a vector space :math:`W`. Now, you
might wonder why there are two vector spaces involved, and this will become more clear after a few
examples.

Let’s consider the following two mappings :math:`f` and :math:`g`:

.. math::
    f\left( \matrix{x \\ y} \right) = \matrix{x \\ y \\ x+y},
    g\left( \matrix{x \\ y} \right) = \matrix{x \\ y \\ xy}

This is a generalization of functions that we are used to; here our inputs are vectors, and so are the
outputs. If :math:`x` and :math:`y` are real numbers, then formally we can write this mapping as :math:`f : \R^2 \to \R^3`,
since the input to our mapping is a 2D vector, and the output is a 3D vector (therefore they *live*
in different vector spaces). You will work more with these types of functions in the Multivariate
Calculus section.

Now, which mappings can be called *linear* mappings? The conditions are intuitive, and quite similar
to the ones of the vector spaces, so we shall provide now a formal definition.

**Definition 2.3**.
Let :math:`V` and :math:`W` be vector spaces over the same field :math:`\F`. A function :math:`f : V \to W`,
said to be a *linear map* if for any two vectors :math:`\vv`, :math:`\vw` from :math:`V` and any scalar :math:`\lambda` from :math:`\F`, the following
two conditions are satisfied:

    1. **Additivity** : :math:`f(\vv +\vw) = f(\vv) + f(\vw)`
    2. **Homogeneity** : :math:`f(\lambda \vv) = \lambda f(\vv)`

The first condition states that the transformation of the sum of vectors has to be equal to the sum
of transformations of every vector individually.
The second condition simply states that it shouldn’t matter whether we first multiply the vector :math:`\vv`
by a scalar :math:`\lambda` and then transform it, or we first transform the vector :math:`\vv` and then multiply it by :math:`\lambda`.

Now, are the mappings :math:`f` and :math:`g` above linear or not? The way to check it is by testing whether they
satisfy the additivity and homogeneity conditions, which we leave as an exercise. [#f4]_

Matrix-vector multiplication
............................

If you’ve encountered linear algebra before, then you probably associate linear mappings/transformations/operators
with matrices. Let’s first discuss how and why matrix-vector multiplication works, and then we will
connect it to the concept of linear mappings discussed in the previous subsection.

To start, let’s imagine we have a very simple canonical basis :math:`B = \{\vb_1, \vb_2 \}` in :math:`\R^2`, where the basis vectors are:

.. math::

    \vb_1 = \matrix{1 \\ 0} , \vb_2 = \matrix{0 \\ 1}


The matrix representation of a linear transformation is *defined* to have the following form: :math:`n`-th
column of the matrix corresponds to a vector to which the :math:`n`-th canonical basis vector transforms.
For example, let’s observe the following matrix :math:`\mA`:

.. math::
    \mA = \matrix{-1 & -2 \\ 1 & -1}

This means that the matrix :math:`\mA` will transform the vectors :math:`\vb_1` and :math:`\vb_2` into :math:`\vb_1'` and :math:`\vb_2'` in the
following way:

.. math::
    \vb_1 = \matrix{1 \\ 0} \rightarrow
    \vb_1' = \matrix{-1 \\ 1}, \quad
    \vb_2 = \matrix{0 \\ 1} \rightarrow
    \vb_2' = \matrix{-2 \\ -1}

which is visualized in the figure below.

.. image:: /figures/linear_algebra/linear_map.png
    :alt: linear_map
    :align: center

Now that we know how a matrix transformation transforms our basis vectors, let’s see how this
applies to an arbitrary vector. Let’s consider a general matrix :math:`\mA` and a vector :math:`\vv` which have the
following form:

.. math::
    \mA = \matrix{A_{11} & A_{12} \\ A_{21} & A_{22}}, \quad \vv = \matrix{v_1 \\ v_2}

The vector produced by the matrix-vector multiplication shall be denoted as :math:`\vw`. Let’s try to calculate
it using the rules of vector spaces and linear operators that we have learned so far:

.. math::
   :nowrap:

   \[
   \begin{array}{
     r@{}% no padding
     l@{}% no padding
   }
   \vw
       &\stackrel{(1)}{=} \matrix{A_{11} & A_{12} \\ A_{21} & A_{22}} \matrix{v_1 \\ v_2} \\
       &\stackrel{(2)}{=} \matrix{A_{11} & A_{12} \\ A_{21} & A_{22}} \left( \matrix{v_1 \\ 0} + \matrix{0 \\ v_2} \right)\\
       &\stackrel{(3)}{=} \matrix{A_{11} & A_{12} \\ A_{21} & A_{22}} \left( v_1 \matrix{1 \\ 0} + v_2 \matrix{0 \\ 1} \right)\\
       &\stackrel{(4)}{=} \matrix{A_{11} & A_{12} \\ A_{21} & A_{22}} \left( v_1 \matrix{1 \\ 0} \right)
        + \matrix{A_{11} & A_{12} \\ A_{21} & A_{22}} \left( v_2 \matrix{0 \\ 1} \right)\\
       &\stackrel{(5)}{=} v_1 \matrix{A_{11} & A_{12} \\ A_{21} & A_{22}} \matrix{1 \\ 0}
        + v_2 \matrix{A_{11} & A_{12} \\ A_{21} & A_{22}}  \matrix{0 \\ 1} \\
       &\stackrel{(6)}{=} v_1 \matrix{A_{11} \\ A_{21} } + v_2 \matrix{A_{12} \\ A_{22}}   \\
       &\stackrel{(7)}{=} \matrix{A_{11} v_1 + A_{12} v_2 \\ A_{21} v_1 + A_{22} v_2}
   \end{array}
   \]

It is important to discuss all the properties used in the derivation above, as they serve as a backbone
to all calculations in linear algebra in general:

    1. We have decomposed the vector :math:`\vv` into its separate components.
    2. We have pulled out the scalar from each vector in order to easily recognize the basis vectors :math:`\vb_1` and :math:`\vb_2`.
    3. Since we are dealing with a linear operator, we use the *additivity* property defined above.
    4. Again, as we are dealing with a linear operator, we use *homogeneity* property defined above.
    5. We use the definition of what matrix columns represent, i.e. we transform the canonical basis vectors accordingly.
    6. We simply sum up the two remaining vectors.

Using known rules we have derived the elements of the transformed vector. This result is general,
and if we have a matrix-vector multiplication of the type :math:`\vw = \mA \vv`, then the :math:`i`-th element of the
output vector :math:`\vw` is given by:

.. math::
    w_i = \sum_k A_{ik} v_k
    :label: matrix-vector-multiplication

Note that :math:`ik`-th element of the matrix :math:`\mA` is simply the entry of the matrix at the :math:`i`-th row and :math:`k`-th
column. Using this formula, we can find every element of the output vector :math:`\vw`.

In the example above, we have assumed that the matrix :math:`\mA` is a square matrix, which resulted in
vectors :math:`\vv` and :math:`\vw` having the same dimension. Let’s take a look at another matrix, :math:`\mA'`. We will define
:math:`\mA'` as:

.. math::
    \mA' = \matrix{1 & 0 \\ 0 & 1 \\ 1 & 1}

Now, let’s try to interpret the meaning of this matrix. We have stated that the columns of the matrix
correspond to the vectors to which our basis vector will transform. So, this means the following:

.. math::
    \vb_1 = \matrix{1 \\ 0} \rightarrow
    \vb_1' = \matrix{1 \\ 0 \\ 1}, \quad
    \vb_2 = \matrix{0 \\ 1} \rightarrow
    \vb_2' = \matrix{0 \\ 1 \\ 1} ,

i.e. we have a transformation from :math:`\R^2 → \R^3`. To calculate how this matrix would transform an
arbitrary vector, we would use the procedure same as above, and would again retrieve equation :eq:`matrix-vector-multiplication`.
As a simple exercise, let’s calculate the output vector :math:`\vw = \mA' \vv`, where vector :math:`\vv = \matrix{x & y}^T` using
relation :eq:`matrix-vector-multiplication`

.. math::
   :nowrap:

   \[
   \begin{array}{
     r@{}% no padding
     l@{}% no padding
   }
       w_1 &= \sum_{k=1}^2 A_{1k}' v_k = A_{11}' v_1 + A_{12}' v_2 = x + 0 = x \\
       w_2 &= \sum_{k=1}^2 A_{2k}' v_k = A_{21}' v_1 + A_{22}' v_2 = 0 + y = y \\
       w_3 &= \sum_{k=1}^2 A_{3k}' v_k = A_{31}' v_1 + A_{32}' v_2 = x + y
   \end{array}
   \]

Therefore, the output vector :math:`\vw` is equal to:

.. math::
    \vw = \matrix{x \\ y \\ x+y}

This is exactly the mapping :math:`f` defined in 2.5! [#f5]_

Let’s summarize our current findings regarding matrix-vector multiplication:

- We have a general formula :eq:`matrix-vector-multiplication` for calculating how a matrix transforms a vector.
- The matrix-vector multiplication may or may not change the dimensionality of the input vector.
- If we have a :math:`n \times k` matrix (`n` rows, :math:`k` columns), then the input vector has to be :math:`k`-dimensional, while the output will be :math:`n`-dimensional.
- All linear transformations (in finite dimensions) can be written in the matrix form.


Matrix-matrix multiplication
............................

In the previous subsection, we have discussed how matrices (linear operators) transform vectors,
and how to calculate elements of the transformed vectors. Matrix-matrix multiplication can be
thought of as chaining two transformations one after another, and for this reason, we can calculate
the resulting matrix elements by analyzing how the two transformations act on the basis vectors.
For simplicity, let’s assume that we have two :math:`2 \times 2` matrices :math:`\mA` and :math:`\mB` of the following form:

.. math::
    \mA = \matrix{A_{11} & A_{12} \\ A_{21} & A_{22}}, \quad
    \mB = \matrix{B_{11} & B_{12} \\ B_{21} & B_{22}}

Now, we wish to calculate elements of the resulting matrix :math:`\mC = \mA \mB`. As we stated before, columns
of the matrix represent to what the canonical basis vectors transform to. Therefore, for example,
the first column of the matrix :math:`\mC` will be given by the vector to which the vector :math:`\vb_1 = \matrix{1 & 0}^T` will
transform to. Let’s calculate this by first acting with the matrix :math:`\mB` and then with matrix :math:`\mA` on the
vector :math:`\vb_1`:

.. math::
   :nowrap:

   \[
   \begin{array}{
     r@{}% no padding
     l@{}% no padding
   }
        \mC \vb_1 &= (\mA \mB) \vb_1 \\
                  &= \matrix{A_{11} & A_{12} \\ A_{21} & A_{22}} \matrix{B_{11} & B_{12} \\ B_{21} & B_{22}} \matrix{1 \\ 0} \\
                  &= \matrix{A_{11} & A_{12} \\ A_{21} & A_{22}} \matrix{B_{11} \\ B_{21} } \\
                  &= \matrix{A_{11} B_{11} + A_{12} B_{21} \\ A_{21} B_{11} + A_{22} B_{21}}
   \end{array}
   \]

where we have used identities and properties described in the matrix-vector multiplication section.
Now, if we write the matrix :math:`\mC` in the following form:

.. math::
    \mC = \matrix{C_{11} & C_{12} \\ C_{21} & C_{22}},

we can recognize that the elements of the first column are given by:

.. math::
   :nowrap:

   \[
   \begin{array}{
     r@{}% no padding
     l@{}% no padding
   }
        C_{11} &= A_{11} B_{11} + A_{12} B_{21} \\
        C_{21} &= A_{21} B_{11} + A_{22} B_{21} \\
   \end{array}
   \]

Similarly, we could calculate the elements of the second column of the matrix :math:`\mC` by observing how
the two transformations transform the vector :math:`\vb_2 = \matrix{0 & 1}^T`.

In general, if we have a matrix-matrix multiplication of the type :math:`\mC = \mA \mB`, the the :math:`ij`-th element of
the matrix :math:`\mC` is given by:

.. math:: C_{ij} = \sum_k A_{ik} B_{kj}
    :label: matrix-matrix-multiplication

It is important to note that we used a simple example where both matrices have the same dimensions.
A more general case would be if the matrix :math:`\mA \in \R^{n \times k}` and :math:`\mB \in \R^{k \times m}`.
Then, the matrix :math:`\mB` would take as the input a :math:`m`-dimensional vector and transform it to a :math:`k`-dimensional vector.
Afterward, the matrix :math:`\mA` would take as the input the transformed :math:`k`-dimensional vector, and output a :math:`n`-dimensional
vector. So, the total transformation :math:`\mC` would be a :math:`n \times n` matrix, i.e. :math:`\mC \in \R^{n \times m}`.
Note that the elements of the matrix :math:`\mC` would still be calculated using formula :eq:`matrix-matrix-multiplication`.

Next, let’s take a look at two special types of matrices:

- **Identity matrix** - Identity matrix is often denoted by :math:`\mI` or :math:`\mathbb{I}`, and it represents a matrix that leaves a vector unchanged, i.e. :math:`\mI \vv = \vv`. Such matrix has elements :math:`1` on the diagonal, and :math:`0` otherwise. For example, a :math:`3\times 3` identity matrix has the following form:

.. math::
    \mI = \matrix{1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1}

- **Inverse matrix** - An inverse of a matrix :math:`\mA` is denoted as :math:`\mA\inv`, and is defined by the following equation:

.. math::
    \mA\inv \mA = \mA \mA\inv = \mI

Intuitively, we can think of the inverse matrix :math:`\mA\inv` as a matrix that counteracts the operation
done by the matrix :math:`\mA`. Therefore, if we chain the two transformations together, it should be
the same as if we did nothing (i.e. the total transformation is equal to the identity matrix
:math:`\mI`). A matrix that has an inverse is called an invertible matrix, and only square matrices are
invertible. More information can be found `here <https://en.wikipedia.org/wiki/Invertible_matrix>`_.

Let’s briefly summarize important information regarding matrix-matrix multiplication:

- Using formula :eq:`matrix-matrix-multiplication` we can find elements of a matrix that is the result of matrix multiplication.
- Multiplying a :math:`n \times k` matrix with a :math:`k \times m` matrix will result in a :math:`n \times m` matrix.
- In general, matrix multiplication is not commutative, i.e. :math:`\mA \mB \neq \mB \mA`.
- An identity matrix :math:`\mI` leaves the vector unchanged.
- Some square matrices :math:`\mA` have an inverse, which is denoted by :math:`\mA\inv`.


Change of Basis
---------------

In the previous section, the elements of the matrix were determined by how they transform the basis
vectors. Let’s take a closer look at two different basis in :math:`\R^2`: a canonical basis :math:`\{\vb_1, \vb_2\}` and an
arbitrary non-canonical basis :math:`\{\vd_1, \vd_2\}` whose elements can be expressed in the canonical basis as:

.. math::
    \vd_1 = \matrix{3/5 \\ 1/3}, \quad \vd_2 = \matrix{1/3 \\ 1}
    :label: example-non-canonical-basis

The two bases are visualized in the figure below.

.. image:: /figures/linear_algebra/change_of_basis.png
    :alt: change_of_basis
    :align: center

We can think of a basis as a language we use to explicitly write vectors and operators as matrices.
However, the way an arbitrary operator :math:`\mA` transforms a vector :math:`\vv` shouldn’t depend on the basis we
use. Therefore, we must adjust the entries of the matrix depending on which basis we use, because as
described before, rows of the matrix correspond to the vectors to which the basis vectors transform
to. So let’s try to motivate intuitively how we can transform a matrix :math:`\mA` that is written in the
canonical basis :math:`\{\vb_1, \vb_2\}` into a matrix :math:`\mA'` which describes the same operation, but in the new basis
:math:`\{\vd_1, \vd_2\}`. The procedure is as follows:

1. We take a vector from the written using vectors from the new basis and translate [#f6]_ it into a language of the old basis using a transformation :math:`\mS`.
2. We act on this translated vector with the operator :math:`\mA` expressed in the canonical basis.
3. We convert the transformed vector back to the language of the new basis using the inverse transformation :math:`\mS\inv`.

So, in total, we can express the change of basis of a matrix as:

.. math::
    \mA' = \mS\inv \mA \mS
    :label: change-of-basis

Next question is, what does the transformation :math:`\mS` between languages correspond to? Well, if we speak
the language of the new basis, then we would express the vectors of the new basis as :math:`\vd_1 = \matrix{1 & 0}^T`
and :math:`\vd_2 = \matrix{0 & 1}^T`. However, if we want to express these new vectors in the old canonical basis, then
we would write them in the form of equation :eq:`example-non-canonical-basis`. Therefore, the transformation :math:`\mS` for
this example is equal to:

.. math::
    \mS = \matrix{3/5 & 1/3 \\ 1/3 & 1}

The inverse transformation can be found and is equal to:

.. math::
    \mS\inv = \matrix{45/22 & -15/22 \\ -15/22 & 27/22}

which is not the nicest expression, but we can transform **any** operator :math:`\mA` written in the canonical
basis :math:`\{\vb_1, \vb_2\}` into a matrix :math:`\mA'` written in the :math:`\{\vd_1, \vd_2\}` basis.

We can check whether the transformation :math:`\mS\inv` makes sense by for example applying it on the vector :math:`\vd_1` written
in the canonical basis:

.. math::
    \mS\inv \vd_1 = \matrix{45/22 & -15/22 \\ -15/22 & 27/22} \matrix{1/3 \\ 1} = \matrix{1 \\ 0}

which is exactly the expected result, because if we speak the language of the :math:`\{\vd_1, \vd_2\}` basis, we
would write the vector :math:`\vd_1` as :math:`\vd_1 = \matrix{1 & 0}^T`.





.. rubric:: Footnotes

.. [#f1] An interested reader can find more information `here <https://en.wikipedia.org/wiki/Inner_product_space>`_.
.. [#f2] Inner product and a dot product are often used interchangeably, although there are subtle differences, refer `here <https://math.stackexchange.com/questions/476738/difference-between-dot-product-and-inner-product?rq=1>`_ for a brief discussion.
.. [#f3] Letter :math:`T` stands for the transpose operation, more information can be found `here <https://en.wikipedia.org/wiki/Transpose>`_.
.. [#f4] You should find out that :math:`f` is indeed linear, while :math:`g` isn’t.
.. [#f5] This is actually a very general result, all linear mappings (in finite dimensional vector spaces) can be written as matrix multiplication. More info can be found `here <https://math.stackexchange.com/questions/2547357/is-a-linear-map-transformation-always-a-matrix-multiplication>`_.
.. [#f6] In this context, translation is meant in the context of the language, not as a spatial translation.
