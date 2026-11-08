"""The smallest valid policy: looks for the part, then waits out the episode. Replace policy() with your own.

A policy is a Python file defining policy(api); the methods of `api` are listed by

    PYTHONPATH=.:benchmark python -c "import api; print(api.spec())"

api.EpisodeOver is raised when the episode ends (success, safety stop or the time limit): let it propagate.
"""


def policy(api):
    pos, quat, conf = api.locate_part()          # None, None, conf until a camera has seen the part
    api.wait(1.0)
    api.wait(60.0)                               # raises api.EpisodeOver at the time limit
