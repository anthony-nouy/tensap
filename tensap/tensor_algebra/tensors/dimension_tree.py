'''
Module dimension_tree.

Copyright (c) 2020, Anthony Nouy, Erwan Grelier
This file is part of tensap (tensor approximation package).

tensap is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

tensap is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with tensap.  If not, see <https://www.gnu.org/licenses/>.

'''

import numpy as np


class DimensionTree:
    '''
    Class DimensionTree.

    Attributes
    ----------
    adjacency_matrix : numpy.ndarray
        Adjacency matrix. The nth row indicates the sons of the (n+1)th node.
        The nth column indicates the parents of the (n+1)th node.
    arity : int
        Maximal number of children.
    _child_number : numpy.ndarray
        _child_number[n] = k means that (n+1)th node is the kth child of its
        parent.
    _children : numpy.ndarray
        _children[:,n] is the set of indices of the children of (n+1)th node.
    dim2ind : numpy.ndarray
        dim2ind[k] is the index of the node (leaf) corresponding to
        dimension k.
    dims : list
        dims[n] is the set of dimensions associated with node n+1.
    internal_nodes : numpy.ndarray
        Indices of internal (non leaf) nodes.
    is_leaf : numpy.ndarray
        is_leaf[n] = True if the (n+1)th node is a leaf and False otherwise.
    level : numpy.ndarray
        level[n] is the level of the (n+1)th node.
    nb_nodes : int
        Number of nodes.
    nodes_parent_of_leaves : numpy.ndarray
        Indices of nodes which are parents of leaves.
    _parent : numpy.ndarray
        _parent[n] is the index of the parent of the (n+1)th node.
    root : numpy.int64
        Index of the root node.
    sibling : numpy.ndarray
        sibling[:,n] contains the indices of the children of the parent of
        the (n+1)th node.
    nodes_indices : numpy.ndarray
        Indices of the nodes of the tree.
    plot_options : dict
        Options for plotting the tree.

    '''

    def __init__(self, dim2ind, adjacency_matrix):
        '''
        Constructor of the class DimensionTree.

        Create a dimension partition tree over D = {1,...,d} from an
        adjacency matrix.

        Parameters
        ----------
        dim2ind : numpy.ndarray or list
            dim2ind[k] is the index of the node (leaf) corresponding to
            dimension k.
        adjacency_matrix : numpy.ndarray or list of a list
            The nth row indicates the sons of the (n+1)th node. The nth column
            indicates the parents of the (n+1)th node.

        Returns
        -------
        None.

        '''
        self.dim2ind = np.array(dim2ind)
        self.adjacency_matrix = np.array(adjacency_matrix)
        self.plot_options = {'level_alignment': False}

        self._precompute_attributes()
        self.update_dims_from_leaves()

    def __repr__(self):
        return ('<DimensionTree:{n}' +
                '{t}nb_nodes = {},{n}' +
                '{t}arity = {},{n}' +
                '{t}dim2ind = {},{n}' +
                '{t}is_leaf = {},{n}').format(self.nb_nodes,
                                              self.arity,
                                              self.dim2ind,
                                              self.is_leaf,
                                              t='\t', n='\n')

    def permute(self, sigma):
        '''
        Permute the dimensions of the DimensionTree.

        Parameters
        ----------
        sigma : list
            Indicates the permutation of the dimensions.

        Returns
        -------
        DimensionTree
            A DimensionTree with permuted dimensions.

        '''
        dim2ind = np.copy(self.dim2ind)
        dim2ind[sigma] = dim2ind.flatten()
        return DimensionTree(dim2ind, np.copy(self.adjacency_matrix))

    def ipermute(self, sigma):
        '''
        Inverse permutation of the dimensions of the DimensionTree.

        Parameters
        ----------
        sigma : list
            Indicates the permutation of the dimensions.

        Returns
        -------
        DimensionTree
            A DimensionTree with permuted dimensions.

        '''
        dim2ind = np.copy(self.dim2ind)
        dim2ind = dim2ind[sigma].flatten()
        return DimensionTree(dim2ind, np.copy(self.adjacency_matrix))

    def __eq__(self, T):
        is_equal = (len(self.dim2ind) == len(T.dim2ind)) and \
            (self.nb_nodes == T.nb_nodes) and \
            (np.max(self.level) == np.max(T.level))

        if is_equal:
            for level in np.arange(1, np.max(self.level)+1):
                nodes1 = self.nodes_with_level(level)
                nodes2 = T.nodes_with_level(level)
                if len(nodes1) != len(nodes2):
                    return is_equal
                dims1 = [self.dims[i] for i in nodes1-1]
                dims2 = [T.dims[i] for i in nodes2-1]
                for i, dim1 in enumerate(dims1):
                    for dim2 in dims2:
                        if np.array_equal(np.sort(dim1),
                                          np.sort(dim2)):
                            nodes1[i] = 0
                            break
                if any(nodes1 != 0):
                    is_equal = False
                    return is_equal
        return is_equal

    def __ne__(self, T):
        return not self == T

    def children(self, nod):
        '''
        Return the children of a given node.

        Parameters
        ----------
        nod : int
            Node for which to compute the children.

        Returns
        -------
        list
            List of children of nod.

        '''
        children = self._children[:, nod-1]
        return children[children != 0]

    def child_number(self, nod):
        '''
        Return the child number of nod.

        Parameters
        ----------
        nod : int
            Node for which to compute the child number.

        Returns
        -------
        int
            The child number.

        '''
        return self._child_number[nod-1]

    def parent(self, nod):
        '''
        Return the parent of a given node.

        Parameters
        ----------
        nod : int
            Node for which to compute the parent.

        Returns
        -------
        int
            The parent of nod.

        '''
        return self._parent[nod-1]

    def ascendants(self, nod):
        '''
        Return the ascendants of a given node.

        Parameters
        ----------
        nod : int
            Node for which to compute the ascendants.

        Returns
        -------
        anod : list
            List of ascendants of nod.

        '''
        anod = []
        pnod = self.parent(nod)
        while pnod:
            anod.append(pnod)
            pnod = self.parent(pnod)
        return np.array(anod)

    def descendants(self, nod):
        '''
        Return the descendants of a given node.

        Parameters
        ----------
        nod : int
            Node for which to compute the descendants.

        Returns
        -------
        dnod : list
            List of descendants of nod.

        '''
        dnod = np.array([], dtype=int)
        chnod = self.children(nod)
        chnod = chnod[chnod != 0]
        if len(chnod) != 0:
            for children in chnod:
                dnod = np.concatenate((dnod, [children],
                                       self.descendants(children)))
        return np.array(dnod)

    def sub_dimension_tree(self, root):
        '''
        Extract a sub dimension tree.

        The attribute dim2ind of the sub dimension tree gives the nodes indices
        corresponding to the dimensions in T.dims[r] (not sorted).

        Parameters
        ----------
        root : int
            Index of the node which is the root of the sub dimension tree.

        Returns
        -------
        DimensionTree
            Sub dimension tree.
        nod : np.array
            Extracted nodes from T.

        '''
        dims = self.dims[root-1]
        nod = np.concatenate(([root], self.descendants(root)))
        adj_mat = np.copy(self.adjacency_matrix[nod[:, None]-1, nod-1])
        dim2ind = np.nonzero(np.isin(nod, self.dim2ind[dims]))[0] + 1
        return DimensionTree(dim2ind, adj_mat), nod

    def node_with_dims(self, dims):
        '''
        Return the index of the node with given set of dimensions.

        Return the index of the node corresponding to dimensions dims or an
        empty array if no node contains these dimensions.

        Parameters
        ----------
        dims : list or numpy.ndarray
            List of dimensions.

        Returns
        -------
        numpy.ndarray
            Index of the node with the given set of dimensions dims.

        '''
        dims = np.sort(dims)
        nod = [np.array_equal(dims, np.sort(x)) for x in self.dims]
        return np.nonzero(nod)[0]+1

    def update_dims_from_leaves(self):
        '''
        Update the dimensions of all nodes from the dimensions of the leaves
        given in T.dim2ind.

        Returns
        -------
        DimensionTree
            The DimensionTree object with updated attribute dims.

        '''
        _dims = np.zeros(self.nb_nodes, dtype=int)
        _dims[self.dim2ind-1] = np.arange(len(self.dim2ind))
        _dims = np.split(_dims, len(_dims))

        for level in np.arange(np.max(self.level), -1, -1):
            nod_level = self.nodes_with_level(level)
            for nod in nod_level:
                if not self.is_leaf[nod-1]:
                    children = self._children[:, nod-1]
                    _dims[nod-1] = np.hstack([_dims[i-1] for i in
                                              children[children != 0]])

        self.dims = _dims
        return self

    def nodes_with_level(self, level):
        '''
        Return the indices of the nodes at a given level.

        Parameters
        ----------
        level : int
            Level.

        Returns
        -------
        numpy.ndarray
            Nodes with given level.

        '''
        ind_lvl = np.repeat(False, self.nb_nodes)
        ind_lvl[self.level == level] = True
        return np.nonzero(ind_lvl)[0]+1

    def tree_layout(self):
        '''
        Return the layout of a tree.

        Returns
        -------
        dict
            The layout of the tree, used for plotting.

        '''
        def post_order(T, root=None):
            '''
            Return the post-ordering of a tree.

            Parameters
            ----------
            T : tensap.DimensionTree
            The dimension tree.
            root : int, optional
                The root of the current subtree. The default is None,
                indicating to use T.root.

            Returns
            -------
            post : numpy.ndarray
                The post-ordering of the tree or of one of its subtrees.

            '''
            if root is None:
                root = T.root
            post = np.array([], dtype=int)

            ch = T.children(root)
            if ch.size:
                for nod in ch:
                    post = np.concatenate((post, post_order(T, nod)))
            post = np.concatenate((post, ch))
            return post

        post = post_order(self)

        xmin = np.full(self.nb_nodes, self.nb_nodes)
        xmax = np.zeros(self.nb_nodes)
        y = np.zeros(self.nb_nodes)
        n_leaves = 0
        for nod in post:
            if self.is_leaf[nod-1]:
                n_leaves += 1
                xmin[nod-1] = n_leaves
                xmax[nod-1] = n_leaves
                y[nod-1] = 0
            pa = self.parent(nod)
            xmin[pa-1] = np.min((xmin[pa-1], xmin[nod-1]))
            xmax[pa-1] = np.max((xmax[pa-1], xmax[nod-1]))
            y[pa-1] = np.max((y[pa-1], y[nod-1]+1))

        x = (xmin + xmax) / 2

        x /= (self.dim2ind.size + 1)
        y = (y+1) / (np.max(y)+2)

        if self.plot_options['level_alignment']:
            height = np.max(y) - np.min(y)
            for alpha in np.arange(1, self.nb_nodes+1):
                y[alpha-1] = np.max(y) - self.level[alpha-1] * \
                    height / np.max(self.level)

        pos = [np.array([xx, yy]) for xx, yy in zip(x, y)]
        return dict(zip(range(self.nb_nodes), pos))

    def plot(self, **args):
        '''
        Plot the tree with the nodes indices.

        This method requires the package igraph.

        Parameters
        ----------
        node_color : str, optional
            Color for the colored nodes. The default is 'red'.
        colored_nodes : list or numpy.ndarray, optional
            Colored nodes. The default is [].

        Returns
        -------
        None.

        '''
        self.plot_with_labels_at_nodes(self.nodes_indices, **args)

    def plot_with_labels_at_nodes(self, labels, node_color='red',
                                  colored_nodes=None, title=None):
        '''
        Plot the tree with labels at nodes.

        This method requires the package igraph.

        Parameters
        ----------
        labels : list or numpy.ndarray
            Nodes labels.
        node_color : str, optional
            Color for the colored nodes. The default is 'red'.
        colored_nodes : list or numpy.ndarray, optional
            Colored nodes. The default is [].
        title : str, optional
            The title of the plot. The default is None.

        Returns
        -------
        None.

        '''
        import networkx as nx
        import matplotlib.pyplot as plt

        plt.figure()
        g = nx.convert_matrix.from_numpy_matrix(self.adjacency_matrix)
        pos = self.tree_layout()
        nx.draw_networkx_nodes(g, pos,
                               nodelist=np.setdiff1d(
                                   np.arange(1, self.nb_nodes+1),
                                   colored_nodes)-1,
                               node_color='w',
                               edgecolors='k',
                               node_size=100)

        if colored_nodes is not None:
            nx.draw_networkx_nodes(g, pos,
                                   nodelist=colored_nodes-1,
                                   node_color=node_color,
                                   edgecolors='k',
                                   node_size=100)

        nx.draw_networkx_edges(g, pos)
        labels = dict(zip(range(self.nb_nodes), [x if x is not None
                                                 else '' for x in labels]))
        y = [x[1] for x in pos.values()]
        min_height = np.min(y)
        shift = (np.max(y)-np.min(y))/np.max(self.level)/2
        pos_labels = {}
        for k, v in pos.items():
            if v[1] == min_height:
                pos_labels[k] = (v[0], v[1]-shift)
            else:
                pos_labels[k] = (v[0], v[1]+shift)
        nx.draw_networkx_labels(g, pos_labels, labels)

        plt.axis('off')
        plt.title(title)
        axes = plt.gca()
        axes.set_ylim([0, 1])
        plt.show()

    def plot_dims(self, nodes=None, **args):
        '''
        Plot the dimensions associated with the nodes of the tree.

        This method requires the package igraph.

        Parameters
        ----------
        nodes : list or numpy.ndarray, optional
            List of leaf nodes for which to display the dimensions. The
            default is None, the dimensions of all the leaf nodes are
            displayed.
        Returns
        -------
        None.

        '''
        if nodes is None:
            nodes = self.dim2ind
        labels = [np.nonzero(nodes == i+1)[0][0] if self.is_leaf[i] else None
                  for i in range(self.nb_nodes)]
        self.plot_with_labels_at_nodes(labels=labels, **args)

    def _precompute_attributes(self):
        '''
        Precompute the attributes of the DimensionTree from the attributes
        adjacency_matrix and dim2ind.

        Returns
        -------
        DimensionTree
            A DimensionTree with updated attributes.

        '''
        self.nb_nodes = self.adjacency_matrix.shape[0]
        self.nodes_indices = np.arange(1, self.nb_nodes+1)
        self.arity = np.max(np.sum(self.adjacency_matrix, 1))
        self._parent = np.matmul(range(1, self.nb_nodes+1),
                                 self.adjacency_matrix)
        self.root = self.nodes_indices[self._parent == 0][0]
        self.is_leaf = np.repeat(False, self.nb_nodes)
        self.is_leaf[self.dim2ind-1] = True

        _children = np.zeros([self.arity, self.nb_nodes], dtype=int)
        for i in range(self.nb_nodes):
            ind = np.nonzero(self.adjacency_matrix[i, :])[0]
            _children[0:len(ind), i] = ind+1
        self._children = _children

        _child_number = np.zeros(self.nb_nodes, dtype=int)
        for i in range(self.nb_nodes):
            if self._parent[i]:
                ind1 = self._children[:, self._parent[i]-1]
                ind2 = np.nonzero(ind1 == i+1)
                if ind2[0].size:
                    _child_number[i] = ind2[0][0]+1
        self._child_number = _child_number

        _sibling = np.zeros([self.arity, self.nb_nodes], dtype=int)
        for i in range(self.nb_nodes):
            if self._parent[i]:
                _sibling[:, i] = self._children[:, self._parent[i]-1]
        self.sibling = _sibling

        _level = np.zeros(self.nb_nodes, dtype=int)
        ind = self._children[:, self.root-1]
        ind = ind[ind != 0]
        level = 1
        while ind.size:
            _level[ind-1] = level
            ind = np.reshape(self._children[:, ind-1], [1, -1])
            ind = ind[np.nonzero(ind)]
            level += 1
        self.level = _level

        self.internal_nodes = np.setdiff1d(self.nodes_indices, self.dim2ind)
        self.nodes_parent_of_leaves = np.unique(self._parent[self.dim2ind-1])
        return self

    @staticmethod
    def trivial(order):
        '''
        Create a dimension tree with one level.

        Parameters
        ----------
        order : int
            Order of the tensor (dimension).

        Returns
        -------
        DimensionTree
            Trivial dimension tree.

        '''
        d2i = np.arange(2, order+2)
        adj_mat = np.zeros([order+1]*2, dtype=int)
        adj_mat[0, 1:] = 1
        return DimensionTree(d2i, adj_mat)

    @staticmethod
    def linear(order):
        '''
        Create a linear dimension tree.

        Parameters
        ----------
        order : int
            Order of the tensor (dimension).

        Returns
        -------
        DimensionTree
            Linear dimension tree.

        '''
        order = np.array(order)
        if order.size == 1:
            dim = order
            order = np.arange(dim)
        else:
            dim = order.size
            order -= 1

        d2i = np.concatenate(([2*dim-2], 2*np.arange(dim-1, 0, -1)+1))
        adj_mat = np.zeros([2*dim-1]*2, dtype=int)
        adj_mat[0, 1:3] = 1
        for level in range(dim-2):
            adj_mat[2*level+1, 2*level+3] = 1
            adj_mat[2*level+1, 2*level+4] = 1
        d2i[order] = d2i.flatten()
        return DimensionTree(d2i, adj_mat)

    @staticmethod
    def balanced(order):
        '''
        Create a balanced dimension tree.

        Parameters
        ----------
        order : int
            Order of the tensor (dimension).

        Returns
        -------
        DimensionTree
            Balanced dimension tree.

        '''
        order = np.array(order)
        if order.size == 1:
            dim = order
            order = range(dim)
        else:
            dim = order.size
            order -= 1

        d2i = np.arange(dim, 2*dim)
        adj_mat = np.zeros([2*dim-1]*2, dtype=int)
        adj_mat[0, 1:3] = 1
        for i in range(1, dim-1):
            adj_mat[i, 2*i+1:2*i+3] = 1
        d2i[order] = d2i.flatten()
        return DimensionTree(d2i, adj_mat)

    @staticmethod
    def random(order, arity=2):
        '''
        Create a random dimension tree over {1,...,order}.

        If arity is an interval [amin,amin], then the number of children of a
        node is randomly drawn from the uniform distribution over
        {amin,...,amax}.

        Parameters
        ----------
        order : int
            Order of the tensor (dimension).
        arity : int, list or numpy.ndarray, optional
            Arity or interval for the arity. The default is 2.

        Returns
        -------
        DimensionTree
            Random dimension tree.

        '''
        arity = np.array(arity)
        if arity.size == 1:
            arity = np.repeat(arity, 2)
        arity[1] += 1
        dims_nodes = [np.arange(order)]
        nb_nodes = 1
        new_nodes = [1]
        adj_mat = np.zeros([2*order]*2, dtype=int)
        dim2ind = np.zeros(order, dtype=int)

        while len(new_nodes) != 0:
            parent_nodes = new_nodes
            new_nodes = []
            for pnod in parent_nodes:
                dims = dims_nodes[pnod-1]
                if len(dims) == 1:
                    dim2ind[dims-1] = pnod
                else:
                    pnod_arity = np.min(np.append(np.random.randint(*arity),
                                                  dims.size))
                    for k in range(pnod_arity):
                        nb_nodes += 1
                        new_nodes.append(nb_nodes)
                        adj_mat[pnod-1, nb_nodes-1] = 1
                        if k == pnod_arity-1:
                            dims_nodes.append(dims)
                            dims = []
                        else:
                            n_children = np.max(np.append(np.random.randint(
                                len(dims)-pnod_arity+k+2), 1))
                            perm = np.random.\
                                permutation(len(dims))[:n_children]
                            dims_nodes.append(dims[perm])
                            dims = np.delete(dims, perm)
                        if len(dims) == 0:
                            break
        adj_mat = adj_mat[:nb_nodes, :nb_nodes]
        return DimensionTree(dim2ind, adj_mat)
