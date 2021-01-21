import common.config as cfg


class Arguments():
    def __init__(
            self,
            args,
    ):
        # expt config
        self.dataset = args.dataset
        self.clf = args.clf
        self.paradigm = args.paradigm
        self.num_train = cfg.num_trains[self.dataset]*args.repeat
        self.num_test = cfg.num_tests[self.dataset]
        self.input_size = cfg.input_sizes[self.dataset]
        self.output_size = cfg.output_sizes[self.dataset]

        # worker clustering config
        self.num_workers = args.num_workers
        self.num_clusters = args.num_clusters
        self.uniform_clusters = args.uniform_clusters
        self.shuffle_workers = args.shuffle_workers

        # training config
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        if not self.test_batch_size:
            self.test_batch_size = self.num_test
        self.epochs = args.epochs
        self.lr = args.lr
        self.decay = args.decay
        self.no_cuda = args.no_cuda
        self.seed = args.seed
        self.stratify = args.stratify
        self.uniform_data = args.uniform_data
        self.shuffle_data = args.shuffle_data
        self.non_iid = args.non_iid
        self.repeat = args.repeat

        # training config
        self.factor = args.factor
        self.topology = args.topology
        self.const_graph = args.const_graph
        if self.const_graph:
            self.graphs = ['topology_rgg_degree_3.2_rho_0.7500.pkl']
        else:
            self.graphs = [
                'topology_rgg_degree_2.0_rho_0.8750.pkl',
                'topology_rgg_degree_2.0_rho_0.8750.pkl',
                'topology_rgg_degree_3.2_rho_0.7500.pkl',
                'topology_rgg_degree_4.0_rho_0.3750.pkl',
            ]
        self.eut_range = args.eut_range

        # logging config

        # constants
        self.delta = args.delta
        self.zeta = args.zeta
        self.beta = args.beta
        self.mu = args.mu
        self.phi = args.phi

        # derived
        self.omega = self.zeta/(2.0*self.beta)
        self.kappa = self.mu/(1.0*self.beta)

        # logging and debug
        self.log_intv = args.log_intv
        self.save_model = args.save_model
        self.dry_run = args.dry_run
        if self.dry_run:
            self.save_model = False
            self.log_interval = 1
            self.epochs = 2
