class ExtractionError(Exception):
    """An extraction could not do its work at all.

    Raised where an extractor would otherwise log a problem and return as if it had
    succeeded — which makes the scheduler record the task as successful, continue the
    pipeline, and report a clean run. Individual items failing is normal and stays
    tolerated; this is for a step that produced nothing.
    """
