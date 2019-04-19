import numpy as np

class Figure:
    n_genes = 50
    mutation_rate = 0.2
    mutation_prop = 0.1

    def __init__(self, genes=None):
        if genes == None:
            self.genes = np.zeros((self.n_genes, 7))
            # center
            self.genes[:, [0, 1]] = np.random.rand(n_genes, 2)
            # radius
            self.genes[:, [2]] = np.random.exponential(0.1, (n_genes, 1))
            # color
            self.genes[:, [3, 4, 5]] = np.random.rand(n_genes, 3)
            # zorder
            self.genes[:, [6]] = np.random.randn(n_genes, 1)
        else:
            self.genes = genes

    def clone_and_mutate(self):
        clone = Figure(self.genes)
        clone.genes += (np.random.randint(0, 2, (self.n_genes, 7)) *
                        np.randn(self.n_genes, 7) * self.mutation_rate)
        return clone
    
    def breed(self, other):

        new_genes = np.where(np.random.randint(0, 2, (self.n_genes, 7)),
                             self.genes,
                             other.genes)
        return Figure(new_genes)
    
    def draw(self, size):
        pass