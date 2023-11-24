import torch


class EtC:
    """EtC encryption"""
    def __init__(self, n, bs, seed=42):
        self.n = n
        self.bs = bs
        # self.s = 2
        self.skey, self.rkey, self.ikey, self.nkey, self.ckey = \
            self.generate_keys(seed)

        # imagenet
        self.mean = torch.Tensor([0.485, 0.456, 0.406])
        self.std = torch.Tensor([0.229, 0.224, 0.225])
        # cifar10
        # self.mean = torch.Tensor([0.4914, 0.4822, 0.4465])
        # self.std = torch.Tensor([0.2471, 0.2435, 0.2616])

    def generate_keys(self, seed):
        torch.manual_seed(seed)
        svec = torch.randperm(self.n*self.n)
        # svec = torch.randperm(self.n//self.s*self.n//self.s)
        rvec = torch.empty(self.n*self.n).random_(4)
        ivec = torch.empty(self.n*self.n).random_(4)
        nvec = torch.empty(self.n*self.n).random_(2)
        cvec = torch.empty(self.n*self.n).random_(6)
        return svec, rvec, ivec, nvec, cvec

    def rotate(self, block, k):
        r_block = block.clone()
        if k == 0:
            r_block = r_block.rot90(k=1, dims=[2, 3])
        elif k == 1:
            r_block = r_block.rot90(k=2, dims=[2, 3])
        elif k == 2:
            r_block = r_block.rot90(k=3, dims=[2, 3])
        return r_block

    def invert(self, block, k):
        i_block = block.clone()
        if k == 0:
            i_block = i_block.flip(2)
        elif k == 1:
            i_block = i_block.flip(3)
        elif k == 2:
            i_block = i_block.flip(2).flip(3)
        return i_block

    def negate(self, block, k):
        n_block = block.clone()
        n_block = self.denormalize(n_block)
        if k == 1:
            n_block = 1 - n_block
        return self.normalize(n_block)
        # return n_block

# Integer | R | G | B
# 0       | R | B | G
# 1       | G | R | B
# 2       | G | B | R
# 3       | B | R | G
# 4       | B | G | R
    def c_shuffle(self, block, k):
        c_block = block.clone()
        if k == 0:
            c_block[:, 1, :, :] = block[:, 2, :, :]
            c_block[:, 2, :, :] = block[:, 1, :, :]
        elif k == 1:
            c_block[:, 0, :, :] = block[:, 1, :, :]
            c_block[:, 1, :, :] = block[:, 0, :, :]
        elif k == 2:
            c_block[:, 0, :, :] = block[:, 1, :, :]
            c_block[:, 1, :, :] = block[:, 2, :, :]
            c_block[:, 2, :, :] = block[:, 0, :, :]
        elif k == 3:
            c_block[:, 0, :, :] = block[:, 2, :, :]
            c_block[:, 1, :, :] = block[:, 0, :, :]
            c_block[:, 2, :, :] = block[:, 1, :, :]
        elif k == 4:
            c_block[:, 0, :, :] = block[:, 2, :, :]
            c_block[:, 2, :, :] = block[:, 0, :, :]
        return c_block


    def normalize(self, X):
        return (X - self.mean.type_as(X)[None, :, None, None]) / self.std.type_as(X)[
            None, :, None, None
        ]

    def denormalize(self, X):
        return (X * self.std.type_as(X)[None, :, None, None]) + self.mean.type_as(X)[
            None, :, None, None
        ]

    def encrypt(self, images):
        enc_images = images.clone()
        # for i in range(self.n//self.s * self.n//self.s):
        #     a = i // (self.n//self.s)
        #     b = i % (self.n//self.s)
        #     aa = self.skey[i].item() // (self.n//self.s)
        #     bb = self.skey[i].item() % (self.n//self.s)
        #     enc_images[:, :, a*self.bs*self.s:(a+1)*self.bs*self.s, b*self.bs*self.s:(b+1)*self.bs*self.s]\
        #         = images[:, :, aa*self.bs*self.s:(aa+1)*self.bs*self.s,
        #                  bb*self.bs*self.s:(bb+1)*self.bs*self.s]

        for i in range(self.n * self.n):
            a = i // self.n
            b = i % self.n
            aa = self.skey[i].item() // self.n
            bb = self.skey[i].item() % self.n
            enc_images[:, :, a*self.bs:(a+1)*self.bs, b*self.bs:(b+1)*self.bs]\
                = images[:, :, aa*self.bs:(aa+1)*self.bs,
                         bb*self.bs:(bb+1)*self.bs]

            # combine rotation and inversion
            enc_images[:, :, a*self.bs:(a+1)*self.bs, b*self.bs:(b+1)*self.bs]\
                = self.rotate(enc_images[:, :, a*self.bs:(a+1)*self.bs,
                              b*self.bs:(b+1)*self.bs], self.rkey[i])
            enc_images[:, :, a*self.bs:(a+1)*self.bs, b*self.bs:(b+1)*self.bs]\
                = self.invert(enc_images[:, :, a*self.bs:(a+1)*self.bs,
                              b*self.bs:(b+1)*self.bs], self.rkey[i])

            enc_images[:, :, a*self.bs:(a+1)*self.bs, b*self.bs:(b+1)*self.bs]\
                = self.negate(enc_images[:, :, a*self.bs:(a+1)*self.bs,
                              b*self.bs:(b+1)*self.bs], self.nkey[i])
            enc_images[:, :, a*self.bs:(a+1)*self.bs, b*self.bs:(b+1)*self.bs]\
                = self.c_shuffle(enc_images[:, :, a*self.bs:(a+1)*self.bs,
                                 b*self.bs:(b+1)*self.bs], self.ckey[i])
        return enc_images

