class RetrievalError(Exception):
    """
    Custom exception for retrieval or re-ranking failures.
    """


class GpuApiError(Exception):
    """
    Custom exception for errors related to the GPU API service.
    """


class ModelNotReadyError(Exception):
    """
    Custom exception for when the model is not ready.
    """
