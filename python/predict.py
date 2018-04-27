from collections import deque
from copy import deepcopy

###### Generate prob assignment using context tree weighting #####
# sequences - N x T matrix of binary observations (N=num seqs,T=num obs)
# target - which sequence to try and predict (can be 0,1,...,N-1)
# depth - depth of context tree
##############################
def ctw(sequences,target=0,depth=3,debug=False,alphabet=2):
    (N,T) = sequences.shape
    tree = Tree(max_depth=depth,nsymbols=alphabet**N,debug=debug)
    probs = []
    pes = []
    context = deque([0]*depth)

    for t in range(T):

        probs.append(tree.get_probability(context))

        obs = sequences[:,t]
        tree.add_count(context,obs[target])

        symbol = get_symbol(obs)
        context.popleft()
        context.append(symbol)

        tree.update_weights()
        pes.append(tree.root.pe)

    return probs, pes, tree


# turn an observation into a single integer symbol
def get_symbol(obs):
    symbol = 0
    for n in range(len(obs)):
        symbol += obs[n]*(2**n)
    return int(symbol)

class Tree():
    def __init__(self,max_depth,nsymbols,debug):
        self.max_depth = max_depth
        self.nsymbols = nsymbols
        self.root = Node(s='',nsymbols=nsymbols,parent=None)
        self.nodes = {'0':[self.root]}
        self.debug = debug
        depth = 1
        while depth <= max_depth:
            self.nodes[str(depth)] =[]
            for node in self.nodes[str(depth-1)]:
                for symbol in range(nsymbols):
                    s = str(symbol) + node.get_string()
                    new_node = Node(s=s,nsymbols=nsymbols,parent=node)
                    node.add_child(new_node)
                    self.nodes[str(depth)].append(new_node)
            depth += 1

    def add_count(self,context,count):
        self.root.add_count(count)
        cur = self.root

        if self.debug: print(context)
        for symbol in reversed(context):
            cur = cur.get_child(symbol)
            cur.add_count(count)

    def update_weights(self):
        for depth in reversed(range(self.max_depth+1)):
            for node in self.nodes[str(depth)]:
                if depth == self.max_depth:
                    node.pw = node.pe
                else:
                    pw = 1
                    for child in node.get_children():
                        pw = pw * child.pw
                    node.pw = 0.5*pw + 0.5*node.pe

    def get_probability(self,context):
        future = deepcopy(self)
        future.add_count(context,1)
        future.update_weights()
        future_pw = future.root.get_prob()
        pw = self.root.get_prob()
        return future_pw/pw

    def print_tree(self):
        for depth in range(self.max_depth+1):
            for node in self.nodes[str(depth)]:
                print('%i zeros and %i ones for context %s'\
                    %(node.zeros,node.ones,node.s))

class Node():
    def __init__(self,s,nsymbols,parent):
        self.s = s
        self.zeros = 0
        self.ones = 0
        self.children = []
        self.parent = parent
        self.pe = 1
        self.pw = 1

    def add_child(self,child):
        self.children.append(child)

    def add_count(self,count):
        if count == 0:
            self.pe = ((self.zeros+0.5)/ \
                (self.zeros+self.ones+1))*self.pe
            self.zeros += 1
        else:
            self.pe = ((self.ones+0.5)/ \
                (self.zeros+self.ones+1))*self.pe
            self.ones += 1

    def get_child(self,symbol):
        return self.children[symbol]

    def get_children(self):
        return self.children

    def get_prob(self):
        return self.pw

    def get_string(self):
        return self.s
