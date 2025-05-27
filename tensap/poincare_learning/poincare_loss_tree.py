
import numpy as np
from tensap.tensor_algebra.tensors.dimension_tree import DimensionTree


class Cofnet:

    def __init__(self, tree, bases):

        self.tree = tree
        self.bases = bases
    
    def eval(self, x):

        is_leaf = self.tree.is_leaf[self.tree.root - 1]

        if is_leaf:
            out = self._eval_leaf(x)
        else:
            out = self._eval_node(x)

        return out

    def _eval_node(self, x):

        tree = self.tree
        bases = self.bases
        z_lst = []
        
        for a in tree.descendants(tree.root):

            sub_cofnet, _ = self.sub_cofnet(a)
            z_a = sub_cofnet.eval(x[:, tree.dims[a-1]])
            z_lst.append(z_a)
        
        z = np.hstack(z_lst)
        out = bases[tree.root - 1].eval(z)

        return out
    
    def _eval_leaf(self, x):
        
        out = self.bases[self.tree.root - 1].eval(x)
        return out


    def sub_cofnet(self, root):

        sub_tree, nodes = self.tree.sub_dimension_tree(root)
        sub_bases = self.bases[nodes]
        sub_cofnet = Cofnet(sub_tree, sub_bases)

        return sub_cofnet


if __name__ == '__main__':

    import tensap
    d = 5
    X = tensap.RandomVector(tensap.UniformRandomVariable(-1, 1), d)

    # %% build the dimension tree
    tree = DimensionTree.trivial()

    # %% build the bases

    bases = np.empty(shape=tree.nb_nodes, dtype=np.object_)
    poly_deg = 1
    p_norm = 0

    # leaf bases
    for i in X.ndim:
        poly = X.orthonormal_polynomials()[i]
        bases_i = tensap.PolynomialFunctionalBasis(poly, range(poly_deg + 1))
        bases[tree.dim2ind[i]] = bases_i
    
    for i in np.arange(tree.nb_nodes)[tree.is_leaf]:
        dim_i = np.prod(tree.children(i))
        


    

    I0 = tensap.MultiIndices.with_bounded_norm(X.ndim(), p_norm, poly_deg)
    I0 = I0.remove_indices(0)



    leaf_bases = [
        tensap.PolynomialFunctionalBasis(poly, range(poly_deg + 1))
        for poly in X.orthonormal_polynomials()
    ]
    
    tensap.FunctionalBases()
