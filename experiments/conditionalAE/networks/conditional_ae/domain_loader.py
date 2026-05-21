class DomainLoaderWrapper:
    """Sets model domain condition from batch[3] filenames before each batch."""

    def __init__(self, loader, model):
        self.loader = loader
        self.model = model

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __iter__(self):
        for batch in self.loader:
            names = batch[3]
            if isinstance(names, str):
                names = [names]
            self.model.set_domain_from_names(names)
            yield batch

    def __len__(self):
        return len(self.loader)
