from . import SentenceEvaluator, SimilarityFunction
import logging
import os
import csv
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
import numpy as np
from typing import List
from ..readers import InputExample
import numpy as np
import torch
#from .util import AverageMeter


logger = logging.getLogger(__name__)


class AlignmentandUniformityEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on the similarity of the embeddings by calculating the Spearman and Pearson rank correlation
    in comparison to the gold standard labels.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the Spearman correlation with a specified metric.

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """
    def __init__(self, sentences1: List[str], sentences2: List[str], scores: List[float], batch_size: int = 16, main_similarity: SimilarityFunction = None, name: str = '', show_progress_bar: bool = False, write_csv: bool = True):
        """
        Constructs an evaluator based for the dataset

        The labels need to indicate the similarity between the sentences.

        :param sentences1:  List with the first sentence in a pair
        :param sentences2: List with the second sentence in a pair
        :param scores: Similarity score between sentences1[i] and sentences2[i]
        :param write_csv: Write results to a CSV file
        """
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.scores = scores
        self.write_csv = write_csv

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.scores)

        self.main_similarity = main_similarity
        self.name = name

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file = "similarity_evaluation"+("_"+name if name else '')+"_results.csv"
        self.csv_headers = ["epoch", "steps", "Cosine-Similarity", "Manhattan-Distance", "Euclidean-Distance", "Dot-Product-Similarity", "align_meter", "unif_meter", "loss_meter"]

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        sentences1 = []
        sentences2 = []
        scores = []

        for example in examples:
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])
            scores.append(example.label)
        return cls(sentences1, sentences2, scores, **kwargs)


    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"
        print("EmbeddingSimilarityEvaluator的__call__")
        logger.info("EmbeddingSimilarityEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)

        embeddings1 = model.encode(self.sentences1, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        print("embedding_type:",type(embeddings1))
        print("embedding_shape:", embeddings1.shape)
        #print("embedding1:"+ str(embeddings1))
        embeddings2 = model.encode(self.sentences2, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        #print("embeddings2:", embeddings2)


        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        print("!!cosine_scores:", cosine_scores)
        manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
        print("!!manhattan_distances:", manhattan_distances)
        euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
        print("!!euclidean_distances:", euclidean_distances)
        dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]
        print("!!dot_products:", dot_products)

        cosine_scores_mean = np.mean(cosine_scores)
        manhattan_distances_mean = np.mean(manhattan_distances)
        euclidean_distances_mean = np.mean(euclidean_distances)
        dot_products_mean = np.mean(dot_products)
        x = torch.Tensor(embeddings1)
        y = torch.Tensor(embeddings2)
        #alignment_loss = self.align_loss(embeddings1, embeddings2)
        alignment_loss = self.align_loss(x, y)
        #uniform_loss = self.uniform_loss(embeddings1)
        uniform_loss = (self.uniform_loss(x) + self.uniform_loss(y)) / 2
        loss = alignment_loss * 0.9 + uniform_loss * 0.1

        '''
        align_meter = AverageMeter('align_loss')
        unif_meter = AverageMeter('uniform_loss')
        loss_meter = AverageMeter('total_loss')
        for emb1, emb2 in zip(embeddings1, embeddings2):
            emb1 = torch.Tensor(emb1)
            emb2 = torch.Tensor(emb2)
            print('type_emb1:', type(emb1))
            print('shape:', emb1.shape)
            align_loss_val = self.align_loss(emb1, emb2)
            unif_loss_val = (self.uniform_loss(emb1) + self.uniform_loss(emb2)) / 2
            loss = align_loss_val * 0.75 + unif_loss_val * 0.5
            align_meter.update(align_loss_val, emb1.shape[0])
            unif_meter.update(unif_loss_val)
            loss_meter.update(loss, emb1.shape[0])
'''

        logger.info("Cosine-Similarity :\t {:.4f}".format(cosine_scores_mean))
        logger.info("Manhattan-Distance:\t {:.4f}".format(manhattan_distances_mean))
        logger.info("Euclidean-Distance:\t {:.4f}".format(euclidean_distances_mean))
        logger.info("Dot-Product-Similarity:\t {:.4f}".format(dot_products_mean))
        logger.info("alignment_loss:\t {:.4f}".format(alignment_loss))
        logger.info("uniform_loss:\t {:.4f}".format(uniform_loss))
        logger.info("loss:\t {:.4f}".format(loss))
        #logger.info("alignment_loss:\t {:.4f}".format(align_meter))
        #logger.info("uniform_loss:\t {:.4f}".format(unif_meter))
        #logger.info("loss:\t {:.4f}".format(loss_meter))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, newline='', mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                #writer.writerow([epoch, steps, cosine_scores_mean, -manhattan_distances_mean, -euclidean_distances_mean, dot_products_mean, align_meter, unif_meter, loss_meter])
                writer.writerow([epoch, steps, cosine_scores_mean, -manhattan_distances_mean, -euclidean_distances_mean, dot_products_mean, uniform_loss, loss])

        if self.main_similarity == SimilarityFunction.COSINE:
            return cosine_scores_mean
        elif self.main_similarity == SimilarityFunction.EUCLIDEAN:
            return manhattan_distances
        elif self.main_similarity == SimilarityFunction.MANHATTAN:
            return euclidean_distances
        elif self.main_similarity == SimilarityFunction.DOT_PRODUCT:
            return dot_products
        elif self.main_similarity is None:
            # return max(cosine_scores_mean, manhattan_distances_mean, euclidean_distances_mean, dot_products_mean)
            #return -loss_meter
            return -loss
        else:
            raise ValueError("Unknown main_similarity value")

    @classmethod
    def align_loss(cls, x, y, alpha=2):
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    @classmethod
    def uniform_loss(cls, x, t=2):
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


