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

        # stop criterion
        self.accuracy = args.accuracy
        self.patience = args.patience

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

        # distributed training config
        self.factor = args.factor
        self.eut_range = args.eut_range
        self.eut_seed = args.eut_seed
        self.lut_intv = args.lut_intv
        self.rounds = args.rounds
        self.channel = args.channel
        self.eut_round = 1
        self.lut_round = 1

        # logging config

        # constants
        self.beta = args.beta
        self.mu = args.mu
        self.fp_mu = 1.0 if args.paradigm == 'fp' else 0.0
        self.delta = args.delta
        self.zeta = args.zeta
        self.phi = args.phi
        self.sigma = args.sigma
        self.xi = args.xi
        self.tau_max = args.tau_max
        self.e_frac = args.e_frac
        self.d_frac = args.d_frac
        self.cs = args.cs

        # derived
        if args.paradigm == 'hl' and not self.lut_intv:
            self.omega = self.zeta/(2.0*self.beta)
            self.kappa = self.mu/(1.0*self.beta)

        # logging and debug
        self.log_intv = args.log_intv
        self.save_model = args.save_model
        self.dry_run = args.dry_run
        if self.dry_run:
            self.save_model = False
            self.log_interval = 1
            self.epochs = 3
