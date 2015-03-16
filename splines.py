from __future__ import division
import numpy as np

# following code is copied from nwilming's ocupy/spline_base.py
# see https://github.com/nwilming

def augknt(knots,order):
    """Augment knot sequence such that some boundary conditions
    are met."""
    a = []
    [a.append(knots[0]) for t in range(0,order)]
    [a.append(k) for k in knots]
    [a.append(knots[-1]) for t in range(0,order)]
    return np.array(a)


def spcol(x,knots,spline_order):
    """Computes the spline colocation matrix for knots in x.

    The spline collocation matrix contains all m-p-1 bases
    defined by knots. Specifically it contains the ith basis
    in the ith column.

    Input:
        x: vector to evaluate the bases on
        knots: vector of knots
        spline_order: order of the spline
    Output:
        colmat: m x m-p matrix
            The colocation matrix has size m x m-p where m
            denotes the number of points the basis is evaluated
            on and p is the spline order. The colums contain
            the ith basis of knots evaluated on x.
    """
    columns = len(knots) - spline_order - 1
    colmat = np.nan*np.ones((len(x), columns))
    for i in range(columns):
        colmat[:,i] = spline(x, knots, spline_order, i)
    return colmat

def spline(x,knots,p,i=0.0):
    """Evaluates the ith spline basis given by knots on points in x"""
    assert(p+1<len(knots))
    return np.array([N(float(u),float(i),float(p),knots) for u in x])

def N(u,i,p,knots):
    """Compute Spline Basis

    Evaluates the spline basis of order p defined by knots
    at knot i and point u.
    """
    if p == 0:
        if knots[int(i)] < u and u <=knots[int(i+1)]:
            return 1.0
        else:
            return 0.0
    else:
        try:
            k = (( float((u-knots[int(i)]))/float((knots[int(i+p)] - knots[int(i)]) ))
                    * N(u,i,p-1,knots))
        except ZeroDivisionError:
            k = 0.0
        try:
            q = (( float((knots[int(i+p+1)] - u))/float((knots[int(i+p+1)] - knots[int(i+1)])))
                    * N(u,i+1,p-1,knots))
        except ZeroDivisionError:
            q  = 0.0
        return float(k + q)

