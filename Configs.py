class NetConfig:
    def __init__(self):
        self.if_transskip = True
        self.if_convskip = True
        self.patch_size = 2
        self.in_chans = 2
        self.embed_dim = 12
        self.depths = (2, 4, 2, 2)
        self.num_heads = (3, 6, 12, 24)
        self.window_size = (3, 3, 3)
        self.mlp_ratio = 4
        self.pat_merg_rf = 4
        self.qkv_bias = False
        self.drop_rate = 0
        self.drop_path_rate = 0.3
        self.ape = True
        self.spe = False
        self.patch_norm = True
        self.use_checkpoint = False
        self.out_indices = (0, 1, 2, 3)
        self.seg_head_chan = self.embed_dim // 2
        self.img_size = (32, 224, 224)
        self.pos_embed_method = 'relative'
