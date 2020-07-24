'''
Tutorial on DimensionTree.

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

import tensap

# %% Linear dimension tree
ORDER = 5
TREE = tensap.DimensionTree.linear(ORDER)
TREE.plot(title='Nodes indices')
TREE.plot_dims(title='Nodes dimensions')

# %% Random dimension tree
ORDER = 10
ARITY_INTERVAL = [2, 3]
TREE = tensap.DimensionTree.random(ORDER, ARITY_INTERVAL)
TREE.plot(title='Nodes indices')
TREE.plot_dims(title='Nodes dimensions')

# %% Balanced dimension tree
ORDER = 10
TREE = tensap.DimensionTree.balanced(ORDER)
TREE.plot(title='Nodes indices')
TREE.plot_dims(title='Nodes dimensions at the leaves')

TREE.plot(colored_nodes=TREE.ascendants(4), node_color='b',
          title='Ascendants of node 4')

TREE.plot(colored_nodes=TREE.descendants(4), node_color='b',
          title='Descendants of node 4')

# %% Extraction of a subtree
[SUB_TREE, NOD] = TREE.sub_dimension_tree(4)
TREE.plot(colored_nodes=NOD, node_color='b', title='Sub-tree to extract')
SUB_TREE.plot(title='Sub-tree')

# %% Plot with and without the same level at the same height
ORDER = 8
TREE = tensap.DimensionTree.random(ORDER)

TREE.plot_options['level_alignment'] = False
TREE.plot_with_labels_at_nodes(TREE.level, title='No level alignment')
TREE.plot_options['level_alignment'] = True
TREE.plot_with_labels_at_nodes(TREE.level, title='Level alignment')
