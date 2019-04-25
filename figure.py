import numpy as np
from scipy import stats
import random
from PIL import Image, ImageDraw


class Population:
    n_figures = 50

    def __init__(self):
        self.figures = []
        for i in range(self.n_figures):
            self.figures.append(Figure())

    def score_and_sort(self, model, target):
        '''Score figures against a target using a model, and sort
        from best to worst'''
        scores = []
        for figure in self.figures:
            scores.append(-figure.score(model, target))
            print(".", end='')
        self.figures = [f for _, f in sorted(zip(scores, self.figures),
                                             key=lambda x:x[0])]
        print(sorted(scores))

    def purge_and_mutate(self):
        # first, choose random numbers to mutate/breed
        weights = 1/(np.arange(20, 50))
        mutaters = random.choices(range(30), k=10, weights=weights)
        breeders = random.choices(range(30), k=20, weights=weights)

        for i, mutater in enumerate(mutaters):
            self.figures[i + 30] = self.figures[mutater].clone_and_mutate()

        for i, b1, b2 in zip(range(10), breeders[:10], breeders[10:]):
            self.figures[i + 40] = self.figures[b1].breed(self.figures[b2])


class Figure:
    n_genes = 50
    mutation_rate = 0.05
    mutation_prob = 0.2
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
        self.score_ = None

    def clone_and_mutate(self):
        clone = Figure(self.genes.copy())
        clone.genes += (stats.bernoulli(self.mutation_prob).rvs((self.n_genes, 7)) *
                        stats.norm(self.mutation_rate).rvs((self.n_genes, 7)))
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
            color = tuple((gene[[3, 4, 5]] * 255).astype(int))
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
