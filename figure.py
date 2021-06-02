import numpy as np
from scipy import stats
import random
from PIL import Image, ImageDraw


class Population:
    n_figures = 50

    def __init__(self, model, target_image):
        self.figures = []
        for i in range(self.n_figures):
            self.figures.append(Figure())
        self.model = model
        target_array = np.array(target_image)[None, :, :, 0:3]
        self.target = model.predict(target_array).flatten()
        self.target /= np.linalg.norm(self.target)
        self.scores = None

    def score_and_sort(self):
        '''Score figures against a target using a model, and sort
        from best to worst'''
        self.scores = []
        for figure in self.figures:
            self.scores.append(-figure.score(self.model, self.target))
            print(".", end='')
        self.figures = [f for _, f in sorted(zip(self.scores, self.figures),
                                             key=lambda x:x[0])]

    def purge_and_mutate(self):
        # first, choose random numbers to mutate/breed
        weights = 1/(np.arange(20, 50))
        weights /= weights.sum()
        mutaters = np.random.choice(30, size=15, p=weights, replace=False)
        breeders = np.random.choice(30, size=10, p=weights, replace=False)

        for i, mutater in enumerate(mutaters):
            self.figures[i + 30] = self.figures[mutater].clone_and_mutate()

        for i, b1, b2 in zip(range(5), breeders[:5], breeders[5:]):
            self.figures[i + 45] = self.figures[b1].breed(self.figures[b2])


class Figure:
    n_genes = 50
    mutation_rate = 0.02
    mutation_prob = 0.1
    size = 224

    def __init__(self, genes=None):
        if genes is None:
            self.genes = np.zeros((self.n_genes, 7))
            # center
            self.genes[:, [0, 1]] = np.random.rand(self.n_genes, 2)
            # radius
            self.genes[:, [2]] = np.random.exponential(0.5, (self.n_genes, 1))
            self.genes[:, 2] = np.clip(self.genes[:, 2], None, 0.1)
            # color
            self.genes[:, [3, 4, 5]] = np.random.rand(self.n_genes, 3)
            # zorder
            self.genes[:, [6]] = np.random.randn(self.n_genes, 1)
        else:
            self.genes = genes
        self.score_ = None

    def clone_and_mutate(self):
        clone = Figure(self.genes.copy())
        clone.genes += (stats.bernoulli(self.mutation_prob).rvs((self.n_genes, 7)) *
                        stats.norm(self.mutation_rate).rvs((self.n_genes, 7)))
        self.genes[:, 2] = np.clip(self.genes[:, 2], None, 0.1)

        # don't let it get too big
        
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
            coords = (coords + self.size * gene[[0, 1]]).flatten().tolist()
            color = tuple((gene[[3, 4, 5]] * 255).astype(int)) + (10,)
            draw.ellipse(coords, color)
        return im

    def transform(self, model):
        im = self.draw()
        return model.predict(np.array(im)[None, ...]).flatten()

    def score(self, model, target):
        '''
        Assume target is normalized already
        '''
        if self.score_ is None:
            transformed = self.transform(model)
            self.score_ = transformed @ target / np.linalg.norm(transformed)
        return self.score_
