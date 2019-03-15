from . import register_plugin

@register_plugin
class CantReach(Exception):
    '''
        Raises when cant find path
    '''
    def __init__(self, message):

      super().__init__(message)
