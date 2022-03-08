from graph_tool.generation import price_network

class Model:
    def __init__(self, N: int, pa: int,
                       mu: int, m: int,
                       kappa: float, lambd: float,
                       alpha: float, omega: float,
                       gamma: float,
                       initialize: bool = True):
        self.N = N
        self.pa = pa
        
        self.mu = mu
        self.m  = m
        self.kappa = kappa
        self.alpha = alpha
        self.omega = omega
        self.lambd = lambd
        self.gamma = gamma
        
        self.seed = seed
        
        self.create_graph()
        if initialize: self.initialize_model_info()
            
    def create_graph(self):
        self._G = price_network(self.N, gamma = self.pa, directed = False)