import numpy as np
from numpy import ndarray
from typing import List, Tuple
from opdynamics.model.model import Model
from opdynamics.statistics.abstract_statistic import Statistic


class Entropy(Statistic):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def compute(self, model: Model, *args, **kwargs) -> float:
        """
        Returns the mean entropy of the model.

        Parameters:
            model (Model): The model to compute the entropy.

        Returns:
            float: The mean entropy of the model.
        """
        return model.H

    def get_rep_mean(
        self,
        statistics: ndarray,
        *args,
        **kwargs
    ) -> ndarray:
        """
        Compute the mean of the entropy evolution (array of measures for
        iterations) along repetitions.

        Parameters:
            statistics (ndarray): A numpy array of shape (n, m) where n is
                the number of repetitions and m is the number of measures in
                each repetition.

        Returns:
            ndarray: A numpy array of shape (m,) containing the mean of the
                entropy evolution along repetitions.
        """
        if len(statistics.shape) == 1:
            return statistics
        return statistics.mean(axis=0)


class Polarity(Statistic):
    def compute(self, model: Model, *args, **kwargs) -> float:
        """
        Compute the mean polarity of the model.

        Parameters:
            model (Model): The model to compute the mean polarity.

        Returns:
            float: The mean polarity of the model.
        """
        return model.pi

    def get_rep_mean(
        self,
        statistics: ndarray,
        *args,
        **kwargs
    ) -> ndarray:
        """
        Compute the mean of the polarity evolution (array of measures for
        iterations) along repetitions.

        Parameters:
            statistics (ndarray): A numpy array of shape (n, m) where n is
                the number of repetitions and m is the number of measures in
                each repetition.

        Returns:
            ndarray: A numpy array of shape (m,) containing the mean of the
                polarity evolution along repetitions.
        """
        if len(statistics.shape) == 1:
            return statistics
        return statistics.mean(axis=0)


class Proximity(Statistic):
    def compute(self, model: Model, *args, **kwargs) -> float:
        """
        Compute the mean proximity of the model.

        Parameters:
            model (Model): The model to compute the mean proximity.

        Returns:
            float: The mean proximity of the model.
        """
        return model.S

    def get_rep_mean(
        self,
        statistics: np.ndarray,
        *args,
        **kwargs
    ) -> np.ndarray:
        """
        Compute the mean of the proximity evolution (array of measures for
        iterations) along repetitions.

        Parameters:
            statistics (ndarray): A numpy array of shape (n, m) where n is
                the number of repetitions and m is the number of measures in
                each repetition.

        Returns:
            ndarray: A numpy array of shape (m,) containing the mean of the
                proximity evolution along repetitions.
        """
        if len(statistics.shape) == 1:
            return statistics
        return statistics.mean(axis=0)


class Delta(Statistic):
    def compute(self, model: Model, *args, **kwargs) -> float:
        """
        Compute the mean delta of the model.

        Parameters:
            model (Model): The model to compute the mean delta.

        Returns:
            float: The mean delta of the model.
        """
        return np.asarray(
            [model.ind_vertex_objects[node].delta for node in model.G]
        ).mean()

    def get_rep_mean(
        self,
        statistics: ndarray,
        *args,
        **kwargs
    ) -> ndarray:
        """
        Compute the mean of the delta evolution (array of measures for
        iterations) along repetitions.

        Parameters:
            statistics (ndarray): A numpy array of shape (n, m) where n is
                the number of repetitions and m is the number of measures in
                each repetition.

        Returns:
            ndarray: A numpy array of shape (m,) containing the mean of the
                delta evolution along repetitions.
        """
        if len(statistics.shape) == 1:
            return statistics
        return statistics.mean(axis=0)


class Transmissions(Statistic):
    def compute(self, model: Model, *args, **kwargs) -> ndarray:
        """
        Counts the number of transmissions made by each node in the model
        and associates it with the degree of the node.

        Parameters:
            model (Model): The model to count the transmissions.

        Returns:
            ndarray: A numpy array of shape (n, 2) where n is the number of
                nodes in the model. The first column contains the number of
                transmissions made by the node and the second column contains
                the degree of the node.
        """
        transmissions_by_degree: List[Tuple[int, int]] = [
            (model.ind_vertex_objects[node].transmissions,
             model.G.degree[node])
            for node in model.G
        ]
        transmissions_by_degree = np.array(transmissions_by_degree)

        return transmissions_by_degree

    def get_rep_mean(
        self,
        statistics: ndarray,
        *args,
        **kwargs
    ) -> ndarray:
        """
        Computes the mean evalutions of transmissions made by each node in the
        model along repetitions.

        Parameters:
            statistics (ndarray): A numpy array of shape (n, m, k, 2) where
                n is the number of repetitions, m is the number of measures in
                each repetition, k is the number of nodes in the model.
                For each (n, m, k) triplet the first column contains the
                number of transmissions made by the node and the second column
                contains the degree of the node.

        Returns:
            ndarray: A numpy array of shape (m, k, 2) where m is the number
                of measures in each repetition, k is the number of nodes in the
                model. For each (m, k) pair the first column contains the mean
                number of transmissions made by the node and the second column
                contains the degree of the node.

                Each entry in the array will be the mean transmissions made in
                iteration "m" by a node of a certain degree "d".
        """
        if len(statistics.shape) == 3:
            return statistics
        return statistics.mean(axis=0)


class Acceptances(Statistic):
    def compute(self, model: Model, *args, **kwargs) -> ndarray:
        """
        Counts the number of acceptances made by each node in the model
        and associates it with the degree of the node.

        Parameters:
            model (Model): The model to count the transmissions.

        Returns:
            ndarray: A numpy array of shape (n, 2) where n is the number of
                nodes in the model. The first column contains the number of
                acceptances made by the node and the second column contains
                the degree of the node.
        """
        acceptances_by_degree: List[Tuple[int, int]] = [
            (model.ind_vertex_objects[node].acceptances,
             model.G.degree[node])
            for node in model.G
        ]
        acceptances_by_degree = np.array(acceptances_by_degree)

        return acceptances_by_degree

    def get_rep_mean(
        self,
        statistics: ndarray,
        *args,
        **kwargs
    ) -> ndarray:
        """
        Computes the mean evalutions of acceptances made by each node in the
        model along repetitions.

        Parameters:
            statistics (ndarray): A numpy array of shape (n, m, k, 2) where
                n is the number of repetitions, m is the number of measures in
                each repetition, k is the number of nodes in the model.
                For each (n, m, k) triplet the first column contains the
                number of acceptances made by the node and the second column
                contains the degree of the node.

        Returns:
            ndarray: A numpy array of shape (m, k, 2) where m is the number
                of measures in each repetition, k is the number of nodes in the
                model. For each (m, k) pair the first column contains the mean
                number of acceptances made by the node and the second column
                contains the degree of the node.

                Each entry in the array will be the mean acceptances made in
                iteration "m" by a node of a certain degree "d".
        """
        if len(statistics.shape) == 3:
            return statistics
        return statistics.mean(axis=0)


class InformationDistribution(Statistic):
    def compute(self, model: Model, *args, **kwargs) -> ndarray:
        """
        Computes the probability distribution of informations in the system.

        Parameters:
            model (Model): The model to compute the probability distribution.

        Returns:
            ndarray: A numpy array of shape (n,) where n is the number of
                informations in the model. Each entry in the array will be the
                probability of the information "i" in the system.
        """
        P = [model.ind_vertex_objects[node].P_array for node in model.G]
        P = np.array(P)
        P = P.sum(axis=0)/model.N

        return P

    def get_rep_mean(self, statistics: ndarray, *args, **kwargs) -> ndarray:
        """
        Parameters:
            statistics (ndarray): A numpy array of shape (n, m, k) where n is
                the number of repetitions, m is the number of measures in
                each repetition, and k is the number of informations.
        """
        if len(statistics.shape) == 2:
            return statistics
        return statistics.mean(axis=0)
