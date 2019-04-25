import numpy as np
from PIL import Image, ImageDraw

class Figure:
    n_genes = 50
    mutation_rate = 0.1
    mutation_prop = 0.1
    size = 224

    def __init__(self, genes=None):
        if genes is None:
            self.genes = np.zeros((self.n_genes, 7))
            # center
            self.genes[:, [0, 1]] = np.random.rand(self.n_genes, 2)
            # radius
            self.genes[:, [2]] = np.random.exponential(0.1, (self.n_genes, 1))
            # color
            self.genes[:, [3, 4, 5]] = np.random.rand(self.n_genes, 3)
            # zorder
            self.genes[:, [6]] = np.random.randn(self.n_genes, 1)
        else:
            self.genes = genes

    def clone_and_mutate(self):
        clone = Figure(self.genes.copy())
        clone.genes += (np.random.randint(0, 2, (self.n_genes, 7)) *
                        np.random.randn(self.n_genes, 7) * self.mutation_rate)
        return clone
    
    def breed(self, other):

        new_genes = np.where(np.random.randint(0, 2, (self.n_genes, 7)),
                             self.genes,
                             other.genes)
        return Figure(new_genes)
    
    def draw(self):
        im = Image.new('RGB', (self.size, self.size))
        draw = ImageDraw.Draw(im)
        sorted_genes = self.genes[np.argsort(self.genes[:, 6])]
        base_coords = self.size * np.array([[-1, -1], [1, 1]])
        for gene in sorted_genes:
            coords = gene[2] * base_coords
            coords = (coords + self.size * gene[[0,1]]).flatten().tolist()
            color = tuple((gene[[3,4,5]] * 255).astype(int))
            #print(color)
            #print(coords)
            #print()              
            draw.ellipse(coords, color)
        return im

    def transform(self, model):
        im = self.draw()
        return model.predict(np.array(im)[None, ...]).flatten()
    
    def score(self, model, target):
        '''
        Assume target is normalized already
        '''
        transformed = self.transform(model)
        return transformed @ target / np.linalg.norm(transformed) 
            