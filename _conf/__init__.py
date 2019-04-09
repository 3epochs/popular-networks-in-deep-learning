import _conf.global_setting as settings


class Settings:

    def __init__(self, settings):

        for attr in dir(settings):
            if attr.isupper():
                setattr(self, attr, getattr(settings, attr))


settings = Settings(settings)
