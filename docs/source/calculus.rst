Calculus
========



.. toctree::
   :maxdepth: 2
   :caption: Contents


What are derivatives and why should we care?
--------------------------------------------

Before deep diving into derivatives, it is reasonable to ask ourselves what we mean when we talk about the derivative of some function with respect to some variable.
You may know that the derivative describes the **rate of change** of the function.
With 'rate of change' we refer to how quickly the function value increases at some point :math:`x` when we increase the value of :math:`x`.
A running metaphor we will use is the following. We can imagine a variable :math:`y` which is formed through applying function :math:`f` to :math:`x`, i.e. :math:`y = f(x)`.
In this case, we call :math:`x` an **input** and call :math:`y` an **output**.
We are often interested in studying how **sensitive** our outputs are to a change in the inputs, or how much our inputs **influence** our outputs, as we will get more into it soon.
This sensitivity is exactly what is captured by the derivative, e.g. if the derivative of the output with respect to the input is large in some point, we know that output is 'sensitive' to a small change increase around that point.
Now, you can picture this as a machine spitting out outputs :math:`y` controlled with many knobs, where each knob corresponds to a variable :math:`x`.
The derivative tells us how sensitive the value our function spits out is to any turn of the knobs.
Note that standard functions :math:`f: \R \to \R` are machines with one knob and spit out one value, but general functions :math:`f: \R^m \to \R^n` are machines with :math:`m` knobs and spit out :math:`n` different values.
As we will look at later in this section, we have :math:`m \times n` derivatives in the latter case, for we can look at the sensitivity of each output to any of the knobs.

More important, perhaps, is the question of why we care about derivatives at all.
In the context of machine learning, we are often very interested in a function that describes how well our model performs given our parameters.
What we mean with 'doing well' is reflected in \autoref{sec:statlearning}, but for now, we presume that we have some measure of 'doing well'.
It is common to instead of maximizing performance, minimize the error we make, which are equivalent views on the same thing.
Let us, for the sake of simplicity, say that our model parameter is given by :math:`x` and our error rate is given by :math:`f(x) = x^2 + 4x -2`:

.. figure:: /figures/calculus/example_loss.png
    :alt: example_loss_function
    :align: center

    Example loss function. Horizontal axis describes the model parameter value :math:`x`, the vertical axis describes the corresponding error :math:`f(x)`.

If this function describes our error given our model parameters, we would be very interested in finding the point where this error rate is minimum,
which is exactly why we want to use the derivative.
We notice that in our minimum (which soon enough will turn out to be given by :math:`x=-2`), the rate of change of our function is :math:`0`.
Please take your time to verify this, because this point is crucial.

As you might remember from a previous calculus course, the derivative of the function :math:`f(x)` is given by :math:`f'(x) = 2x + 4`:

.. figure:: /figures/calculus/example_loss_2.png
    :alt: example_loss_function_2
    :align: center

    Example loss function plus its derivative. Horizontal axis describes the model parameter value :math:`x`, the vertical axis describes the corresponding error :math:`f(x)` and derivative function :math:`f'(x)`.

Here we see that for all points less than :math:`-2`, indeed the derivative is negative (i.e. the function decreases) and for all points greater than :math:`x=-2`,
the derivative is positive (i.e. the function increases).
It is exactly the minimum point :math:`x=-2` where be function does from decreasing to increasing.
If we want to find this point :math:`x=-2` algebraically, we simply solve :math:`2x + 4 = 0`, which of course turns out to be for :math:`x=-2`.

The following sections will deep dive into how you can find these derivatives.
We will first review the univariate case such as the function we just covered.
We will then steadily work our way up to higher-dimensional derivatives with the aim of you being able to differentiate any ML/DL type of function.

We do now want to spend too much time on basic differentiation techniques and rather give you a general approach to differentiation from which things such as the sum rule, product rule,
chain rule, et cetera, will follow directly.
If you need a refresher on the basic derivative rules, we included them in \autoref{appendix:deriv_rules}.
Let us now dive into the actual derivatives!

Univariate derivatives
----------------------

Let us start nice and easy with our basic functions over the reals, i.e. functions :math:`f: \R \to \R`.
Though this initially may look superfluous, we will introduce a visual way of representing these functions.
This new approach will make it easier to consider multivariate functions and is commonplace in machine learning.
Consider the function :math:`f` such that :math:`f: x \mapsto x^2`, i.e. the functions that squares its input.
Again, our output is given by :math:`y = f(x) = x^2`.
In our example, we can visualize this function as follows:

.. figure:: /figures/calculus/f1.png
    :alt: example_function
    :align: center

The blue squares represent **values** and the yellow rectangles represent ways to **determine** a value.
The most important insight you should take away is that the sensitivity of :math:`y` to :math:`x` is given by the sum of influences of all the paths from :math:`x` to :math:`y`.
In this case, there is only one path, that is through the function :math:`x^2`.
Using basic differentiation techniques, we hence observe that:

.. math::
    \frac{dy}{dx} = \frac{dx^2}{dx} = 2x.


A slightly more *spicy* example if the function :math:`f: \R \to \R` such that :math:`f: x \mapsto \exp (\sin (x))`. [#f1]_
If we make a diagram of this function as above, we can represent it as follows:

.. figure:: /figures/calculus/f2.png
    :alt: example_function
    :align: center

Please note that we had to introduce a new variable :math:`u := \sin (x)` that represents the intermediate value found after applying the sine function to :math:`x`.
When finding the derivative of :math:`y` with respect to :math:`x`, we again count all the paths from :math:`x` to :math:`y`.
Again, there is only one path, now going through our intermediate value :math:`u`.
In this case, the effect of :math:`x` on :math:`y` is equal to the effect of :math:`x` on :math:`u` times the effect of :math:`u` on :math:`y`, i.e.

.. math::
    \frac{dy}{dx} = \frac{dy}{du} \frac{du}{dx}.

You may have encountered this separation of derivatives before as the **chain rule**.
These derivatives are quite simple, giving us

.. math::
    \frac{dy}{dx} = \frac{dy}{du} \frac{du}{dx} = \exp(u) \cdot \cos (x) = \exp (\sin (x)) \cos (x),

where we substituted :math:`u = \sin (x)` in the last step.
So, we **sum** all the paths from :math:`x` to :math:`y`, and we **multiply** the intermediate effects,
e.g. if :math:`x` influences :math:`u` which influences :math:`y`, the influence of :math:`x` on :math:`y` is the
influence of :math:`x` on :math:`u` times the influence of :math:`u` on :math:`y`.


Multivariate derivatives
------------------------

Let's go one step further, and consider a function :math:`f: \R^2 \to \R` such that :math:`f: \begin{bmatrix}x_1 \\ x_2\end{bmatrix} \mapsto x_1x_2^2`.
We can again draw this function:

.. figure:: /figures/calculus/f3.png
    :alt: example_function
    :align: center

In this case, we can consider two derivatives: we can look at the effect of :math:`x_1` on :math:`y` and the effect of :math:`x_2` on :math:`y`.
When we can consider multiple derivatives for different variables, we do not write :math:`\frac{dy}{dx_1}` but rather :math:`\frac{\partial y}{\partial x_1}`, to avoid confusion.
We call such a derivative a **partial derivative**.
Considering our earlier metaphor, a derivative in a real function is just the effect of turning a knob of a machine with one knob, whereas a partial derivative is an effect of turning one of the multiple knobs and keeping the other still.
Luckily for us, we can still apply our same tricks and count the paths from a variable to :math:`y`.
In this case, we have that there is only one path from :math:`x_1` to :math:`y`, and only one path from :math:`x_2` to :math:`y`, giving us:

.. math::
    \frac{\partial y}{\partial x_1} = \frac{\partial x_1x_2^2}{\partial x_1} = x_2^2,

and

.. math::
    \frac{\partial y}{\partial x_2} = \frac{\partial x_1x_2^2}{\partial x_2} = 2x_1x_2.

Please note that since we only consider the influence of one variable at the time, all the other variables are **constant** when taking derivatives.
What we sometimes do, is write the 'full' derivative :math:`\frac{d f}{d\vx}` as the following vector:

.. math::
    \frac{dy}{d\vx} = \begin{bmatrix} \frac{\partial f}{\partial x_1} & \frac{\partial f}{\partial x_2} \end{bmatrix} = \begin{bmatrix}x_2^2 & 2x_1x_2\end{bmatrix}.


We call this full derivative a **gradient** in the case we have functions :math:`f: \R^n \to \R`, denoted as :math:`\frac{dy}{d\vx} = \nabla y (\vx) = \text{grad } y(\vx)`.
However, in the the general case of functions :math:`f: \R^n \to \R^m` we call the resulting matrix a **Jacobian**, denoted as :math:`\frac{d\vy}{d\vx} = \mathbf{J}_\vy(\vx)`.
The Jacobian is just the matrix which has on its :math:`i` row all the partial derivatives of :math:`y_i` with respect to :math:`x_j`, i.e.
:math:`\mathbf{J}_{ij} = \frac{\partial y_i}{\partial x_j}`.
Hence, since we only have one output here, we have that the Jacobian has only one row. [#f2]_

We can also have a function :math:`\vf: \R \to \R^2` which maps :math:`\vf: x \mapsto \begin{bmatrix}x^2 \\ \sqrt{x} \end{bmatrix}`.
In this case, we have that :math:`\vy = \vf(x)` where :math:`\vy` is a vector (and hence is written in bold font), and thus we can consider :math:`y_1 = x^2` and :math:`y_2 = \sqrt{x}`.
Drawing this, we find:

.. figure:: /figures/calculus/f4.png
    :alt: example_function
    :align: center


When again looking at the paths, we see that

.. math::
    \frac{d y_1}{dx} = \frac{d x^2}{dx} = 2x,

and

.. math::
    \frac{d y_2}{dx} = \frac{d \sqrt{x}}{dx} = \frac{1}{2\sqrt{x}}.

Here we can also group the different derivatives into one matrix:

.. math::
    \frac{d\vy}{dx} = \begin{bmatrix} 2x \\ \frac{1}{2\sqrt{x}} \end{bmatrix}.

Please note that if we have a function :math:`f: \R^n \to \R^m` our Jacobian will be of the shape :math:`m \times n`.

Now we are finally ready to consider a function with multiple streams of influence.
Consider the :math:`y = g(\vh(x))`, where :math:`\vh(x) = (x^2, \ln (x))` and :math:`g(u, v) = uv`.
That is, :math:`y` is found by first calculating intermediate values :math:`u = x^2` and :math:`v = \ln (x)` and then finding :math:`y= uv`.
If we draw these functions, we see the following:

.. figure:: /figures/calculus/f5.png
    :alt: example_function
    :align: center


It is now very clear that the effect of :math:`x` of :math:`y` is twofold: both through :math:`u` and :math:`v`.
As mentioned earlier, we need to consider all streams of influence.
Specifically, we **sum** the different paths/effects, i.e.:

.. math::
    \frac{d y}{dx} = \frac{\partial y}{\partial u} \frac{du}{dx} + \frac{\partial y}{\partial v} \frac{dv}{dx}.

Plugging everything in, we find

.. math::
    \frac{d y}{dx} = \frac{dy}{du} \frac{du}{dx} + \frac{dy}{dv} \frac{dv}{dx} = v \cdot 2x + u \cdot \frac{1}{x} = \ln(x) \cdot 2x + x^2 \cdot \frac{1}{x} =  2x (\ln (x) + \frac{1}{2}).

You may recognize this as the product rule, now you know where that comes from!

Finishing up, we go over one big example.
Suppose :math:`\vf: \R^3 \to \R^3` such that :math:`f(x_1, x_2, x_3) = \vh(\vg(x_1, x_2, x_3))`,
where :math:`\vg(x_1, x_2, x_3) = (x_1^2x_2^2, \sqrt{x_2x_3})` and :math:`\vh(u, v) = (u^2, uv, v^2)`.
Try it for yourself!
Find :math:`\frac{\partial y_2}{\partial x_2}`. Hint: draw out what happens.

When visualizing this function, we get the following:

.. figure:: /figures/calculus/f6.png
    :alt: example_function
    :align: center


When counting the paths from :math:`x_2` to :math:`y_2`, we find two paths: one through :math:`u` and one through :math:`v`. We hence find

.. math::
    \frac{\partial y_2}{\partial x_2} = \frac{\partial y_2}{\partial u} \frac{\partial u}{\partial x_2} + \frac{\partial y_2}{\partial v} \frac{\partial v}{\partial x_2}.

Plugging our derivatives, we find

.. math::
    \frac{\partial y_2}{\partial x_2} = v \cdot 2x_1^2x_2 + u \cdot \frac{x_3}{2\sqrt{x_2x_3}} = 2x_1^2 x_2 \sqrt{x_2x_3} + \frac{x_1^2 x_2^2x_3}{2\sqrt{x_2x_3}}.

Sweet! We now know how to find derivatives in multivariate functions.
As you have seen, this approach is quite a time intensive, and sometimes (especially in deep learning) it is not necessary to write out everything by hand like this.
This will be the topic of the rest of this section.

Jacobians
---------

One of the most iconic functions in deep learning is the 'linear layer', which takes some input :math:`\vx \in \R^m` and takes :math:`n` linear combinations
(with different factors) of the inputs.
This linear layer can be considered a function :math:`\vf: \R^m \to \R^n` such that :math:`y_i = w_{i1}x_1 + \cdots +  w_{im}x_m = \sum_{j=1}^m w_{ij} x_j`,
where we still write :math:`\vy = \vf(\vx)`.
We call the :math:`\{w_{ik}\}_{k=1}^m` the **weights** of the function.
Notice that for the entire function :math:`\vf` we have :math:`n` of such sets of weights, i.e. in total :math:`n \times m` weights.
We can write this functions more compactly as

.. math::
    \vy = \mW \vx,

where

.. math::
    \mW = \begin{bmatrix}w_{11} & w_{12} & \cdots & w_{1m} \\ w_{21} & w_{22} & \cdots & w_{2m} \\ \vdots & \vdots & \ddots & \vdots \\ w_{n1} & w_{n2} & \cdots & w_{nm} \end{bmatrix} \in \R^{n \times m}.


When we now imagine all the streams of influence found between the :math:`\vy` and :math:`\vx`, we realize that each element :math:`y_i` is dependent on each variable :math:`x_j`.
If you do not see this immediately, please draw out the respective diagram.

A consequence of this is that we have a lot of derivatives, namely for each of the :math:`n` outputs :math:`y_i` we have :math:`m` different derivatives (for the :math:`m` inputs).
To make our lives a whole lot easier, we simply determine the derivative of the :math:`i`-th element for the :math:`j`-th variable and see if what we end up with generalizes.
We hence wanna find :math:`\frac{\partial y_i}{\partial x_j} = \frac{d}{dx_j} (\sum_{k=1}^m w_{ik}x_k)`.
We know that

.. math::
    \frac{d}{dx_j} (\sum_{k=1}^m w_{ik}x_k) = \sum_{k=1}^m \frac{d}{dx_j} w_{ik}x_k.

Let us know zoom in into one of the terms of the summation, i.e. we only consider :math:`\frac{d}{dx_j} w_{ik}x_k`.
If we have that :math:`x_k \neq x_j`, we will always have that :math:`\frac{d}{dx_j} w_{ik}x_k = 0`, because the entire term does not depend on :math:`x_j`.
When :math:`x_j = x_k`, however, we see that the derivative is given by :math:`w_{ik}`.
We can express this 'if-else' statement quite easily mathematically using something called the **Kronecker delta**.
The Kronecker delta over two variables :math:`i` and :math:`j` is equal to :math:`1` if :math:`i` is equal to :math:`j`, and equal to :math:`0` other, or:

.. math::
    \delta_{ij} = \begin{cases}1& \text{ if } i = j \\ 0& \text{ otherwise }\end{cases}

Sometimes this is written with so-called **Iverson brackets** as :math:`[i=j]`.
These brackets do the same thing, i.e. :math:`[\mathsf{S}] = 1` if :math:`\mathsf{S}` is true, else :math:`[\mathsf{S}] = 0` for any statement :math:`\mathsf{S}`.
The most important property (for us) of this Kronecker delta is that

.. math::
    \sum_j \delta_{ij} x_j = x_i,

i.e. when summing over elements :math:`x_j`, we can filter out :math:`x_i` by introducing :math:`\delta_{ij}`.
Please verify this carefully, for this will be our main workhorse throughout this section.

If we go back to our example, we see that hence our derivative is given by :math:`\frac{d}{dx_j} w_{ik}x_k = \delta_{jk} w_{ik}` for any combination of :math:`x_j` and :math:`x_k`.
That is, the derivative is equal to :math:`0` if :math:`x_j` and :math:`x_k` are different, and equal to :math:`w_{ik}` when :math:`x_j` and :math:`x_k` are the same.
Plugging this back in, we find

.. math::
    \sum_{k=1}^m \frac{d}{dx_j} w_{ik}x_k = \sum_{k=1}^m\delta_{jk} w_{ik}.

This we know how to evaluate using our workhorse, and hence we see that

.. math::
    \frac{df_i}{dx_j} = \sum_{k=1}^m\delta_{jk} w_{ik} = w_{ij}.

Neat! We just found a general approach to taking the derivative of the linear layer and concluded that the effect of the
:math:`j`-th variable on the :math:`i`-th output is given by the weight :math:`w_{ij}`.
We can write out the entire Jacobian (where the element in :math:`i`-th row, :math:`j`-th column is the derivative
of :math:`y_i` with respect to :math:`x_j`) again:

.. math::
    \frac{d\vy}{d\vx}  = \begin{bmatrix}w_{11} & w_{12} & \cdots & w_{1m} \\ w_{21} & w_{22} & \cdots & w_{2m} \\ \vdots & \vdots & \ddots & \vdots & \\ w_{n1} & w_{n2} & \cdots & w_{nm} \end{bmatrix} \in \R^{n \times m}.

But wait! This matrix we recognize from earlier, namely as our matrix :math:`\mW`.
This allows us to write

.. math::
    \frac{d\vy}{d\vx} = \mW.

We call this approach if finding a single entry of the derivative and then generalizing the 'index method'.

Please note that not only did we just derive the derivative of the linear layer, but we found that :math:`\frac{d}{d\vx} \mW\vx = \mW` for arbitrary matrices and vectors,
e.g. we also know now that

.. math::
    \frac{d}{d\vv}(\mA\mB + \mC)\vv = \mA\mB + \mC,

by simply remembering that :math:`\mA\mB + \mC = \mW'` for some matrix :math:`\mW'`.

Another very common derivative you will encounter is :math:`\frac{d}{d\vx} \va^T\vx`.
In this case, we have that :math:`\va^T\vw` is simply a scalar, and hence our Jacobian will be of the shape :math:`(1 \times m)` if :math:`\vx` is :math:`m`-dimensional.
Let us again use the index method, and aim to find

.. math::
    \frac{d}{dx_j} \va^T\vx = \frac{d}{dx_j} \sum_{k=1}^m a_k x_{k} = \sum_{k=1}^m \frac{d}{dx_j} a_k x_k.

As before, we see that the derivative is equal to :math:`a_k` when :math:`k = j`, and equal to zero otherwise, and thus

.. math::
    \sum_{k=1}^m \frac{d}{dx_j} a_k x_k = \delta_{jk} a_k = a_j,

where we find :math:`a_j` by applying our workhorse again.
This means that the :math:`j`-th element of our derivative is given by :math:`a_j`, or the entire derivative is given by :math:`\va`.

But... :math:`\va` is a column vector, where our Jacobian should be a row vector as we argued earlier.
Sadly, this problem cannot quite be overcome, and we just need to always check of our answer should be transposed or not.
In this case, we see that our Jacobian matches :math:`\va^T`.
This is slightly annoying, but luckily our answer is always either correct or needs to be only transposed, and checking it will become second nature soon enough!
Let this inconvenience not distract us from the fact that we did just find our new identity though, that is:

.. math::
    \frac{d}{d\vx} \va^T \vx = \va^T

Now it is your turn, please try and verify that :math:`\frac{d}{d\vx} \vy^T \mA \vx = \vy^T\mA,` where :math:`\vx \in \R^m`, :math:`\mA \in \R^{n \times m}`, and :math:`\vy \in \R^m`.
Please do this 1) using index notation, and 2) using our identity friends we have already found.


We know that :math:`\vy^T \mA \vx` is a scalar (why?), and hence the Jacobian will be again of the form :math:`(1 \times m)` if :math:`\vx` is a :math:`m`-dimensional vector.
Using index notation, we aim to find :math:`\frac{d}{dx_i} \vy^T\mA\vx`. We observe that

.. math::
    \frac{d}{dx_i} \vy^T \mA \vx = \frac{d}{dx_i} \sum_{k=1}^n \sum_{j=1}^m y_{k}A_{kj}x_j = \sum_{k=1}^n \sum_{j=1}^m \frac{d}{dx_i} y_{k}A_{kj}x_j.

Again, since :math:`y_kA_{kj}` are just scalars, we know that the derivative is simply found by

.. math::
    \frac{d}{dx_i} y_k A_{kj}x_j = \delta_{ij} y_k A_{kj}.

This gives us the following derivative:

.. math::
    \sum_{k=1}^n \sum_{j=1}^m \frac{d}{dx_i} y_{k}A_{kj}x_j = \sum_{k=1}^n \sum_{j=1}^m \delta_{ij} y_k A_{kj} = \sum_{k=1}^n  y_k A_{ki}.

But this term we recognize as :math:`[\vy^T\mA]_i`.
This means that our full derivative is simply given by :math:`\vy^T\mA`, which aligns with our desired shape so we are done.

So... That's quite a lot of work. And actually, we could have done way less work using our previous identities.
Observe that :math:`\vy^T\mA` is just a row vector, i.e. it can be written as :math:`\vv^T = \vy^T\mA` for some vector :math:`\vv`.
Thus, we can write :math:`\vy^T\mA\vx = \vv^T\vx`.
But this we know how to differentiate with our tricks, that is :math:`\frac{d}{d\vx} \vv^T\vx = \vv^T`, and hence we know that :math:`\frac{d}{d\vx} \vy^T\mA\vx = \vy^T \mA`.

This should cover the basics of vector calculus!
During the first week of the course, we will spend some more time on time on this and you will receive an excellent document written by two other TAs.
If you understand these basics, you are well on your way to doing machine learning soon enough!











.. rubric:: Footnotes

.. [#f1] If you are not familiar with the :math:`\exp(x)` function, it is just another way to write :math:`e^x`.
.. [#f2] For pedagogical reasons, we will call all such higher-order derivatives of :math:`y` Jacobians and denote them with :math:`\frac{d\vy}{d\vx}`, but in practice, most people will just use the word 'gradient' here anyway.