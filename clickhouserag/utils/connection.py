import functools


def ensure_connection(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self.client:
            raise ConnectionError("Client is not connected. Call `connect` first.")
        return method(self, *args, **kwargs)
    return wrapper
