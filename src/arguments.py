import config as cfg


class Arguments():
    def __init__(
            self,
            args,
    ):
        # data config
        self.dataset = args.dataset
        self.clf = args.clf
        self.paradigm = args.paradigm
        self.num_train = cfg.num_trains[self.dataset]*args.repeat
        self.num_test = cfg.num_tests[self.dataset]
        self.input_size = cfg.input_sizes[self.dataset]
        self.output_size = cfg.output_sizes[self.dataset]
        self.stratify = args.stratify
        self.uniform_data = args.uniform_data
        self.shuffle_data = args.shuffle_data
        self.non_iid = args.non_iid
        self.repeat = args.repeat

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
        self.nesterov = args.nesterov
        self.decay = args.decay
        self.no_cuda = args.no_cuda
        self.seed = args.seed

        # logging config
        self.log_interval = args.log_interval
        self.save_model = args.save_model

        # laplacian consensus
        self.shuffle_worker_data = args.shuffle_worker_data
        self.rounds = args.rounds
        self.const_graph = args.const_graph
        self.radius = args.radius
        if self.radius == 'multi':
            self.graphs = [
                'topology_rgg_degree_2.0_rho_0.8750.pkl',
                'topology_rgg_degree_2.0_rho_0.8750.pkl',
                'topology_rgg_degree_3.2_rho_0.7500.pkl',
                'topology_rgg_degree_4.0_rho_0.3750.pkl',
            ][-len(self.num_clusters):]
        else:
            self.graphs='topology_rgg_degree_3.2_rho_0.7500.pkl'
        self.d2d = args.d2d
        self.factor = args.factor

        # graph topology erdos renyi or rgg
        self.topology = args.topology
        self.eut_int = args.eut_int
        self.lut_int = args.lut_int

        # dry run
        self.dry_run = args.dry_run
        if self.dry_run:
            self.save_model = False
            self.log_interval = 1
            self.epochs = 2
