
class DomainBus(object):

    def __init__(self, domainloaders, train_samplers=None, iter_num=-1):

        self.domainloaders = domainloaders
        self.train_samplers = train_samplers
        self.domainiters = [iter(dataloader) for dataloader in self.domainloaders]
        self.domain_sizes = [len(dataloader) for dataloader in self.domainloaders]

        self.max_iter_num = iter_num if iter_num > 0 else max(self.domain_sizes)
        #print(self.max_iter_num)
        self.current_iter = 0

    def get_samples(self):
        batch_split = []

        for i in range(len(self.domainloaders)):

            try:
               
                imgs, trgs = next(self.domainiters[i])

            except StopIteration:

                self.domainiters[i] = iter(self.domainloaders[i])
                imgs, trgs = next(self.domainiters[i])

            batch_split.append((imgs, trgs))
            

        self.current_iter += 1

        return batch_split

    def __len__(self):
        return self.max_iter_num

    def reset(self):
        self.current_iter = 0

    def __next__(self):

        if self.current_iter >= self.max_iter_num:
            raise StopIteration

        return self.get_samples()

    def __iter__(self):
        return self

    def __str__(self):
        return "\n".join([domainloader.__str__() for domainloader in self.domainloaders])


    def set_epoch(self, epoch):
        if self.train_samplers:
            for sampler in self.train_samplers:
                sampler.set_epoch(epoch)

