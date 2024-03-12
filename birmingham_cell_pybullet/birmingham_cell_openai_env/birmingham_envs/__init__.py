from gymnasium.envs.registration import register

register(
    id="BirminghamCellEnv-v0",
    entry_point="birmingham_envs.envs:BirminghamCellEnv",
)
register(
    id="ConnectionEnv-v0",
    entry_point="birmingham_envs.envs:ConnectionEnv",
)

register(
    id="FakeEnv-v0",
    entry_point="birmingham_envs.envs:FakeEnv",
)

register(
    id="EasyEnv-v0",
    entry_point="birmingham_envs.envs:EasyEnv",
)